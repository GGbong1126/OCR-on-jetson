"""CPU-only image inventory; no detection, recognition, engine load or benchmark."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform

import cv2
import numpy as np

from ocr_det import preprocess

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def detector_profile(report):
    if (report.get("stage") != "benchmark" or report.get("status") not in (
            "completed_consistent_not_accuracy_validated",
            "completed_policy_consistent_not_accuracy_validated")):
        raise ValueError("Expected a completed live pipeline benchmark report")
    meta = report.get("detector", {})
    sha = report.get("det_engine_sha256", "")
    if (len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
            or sha != meta.get("engine_sha256") or meta.get("backend") != "tensorrt"):
        raise ValueError("Invalid or inconsistent detector engine identity")
    profile = meta.get("profile")
    if not isinstance(profile, list) or len(profile) != 3:
        raise ValueError("Missing min/opt/max detector profile")
    for shape in profile:
        if (not isinstance(shape, list) or len(shape) != 4 or shape[:2] != [1, 3]
                or any(type(v) is not int or v < 1 for v in shape)):
            raise ValueError("Expected positive NCHW detector profile, N=1 C=3")
    if any(not lo <= opt <= hi for lo, opt, hi in zip(*profile)):
        raise ValueError("Invalid min/opt/max order")
    return profile


def discover(root):
    """List every regular file; do not silently follow links outside the dataset."""
    files, excluded = [], []

    def fail(error):
        raise error

    for current, dirs, names in os.walk(root, followlinks=False, onerror=fail):
        for name in sorted(dirs[:]):
            path = Path(current) / name
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                dirs.remove(name)
                excluded.append({"path": path.relative_to(root).as_posix(), "reason": "linked_directory"})
        dirs.sort()
        for name in sorted(names):
            path = Path(current) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                excluded.append({"path": relative, "reason": "linked_file"})
            elif not path.is_file():
                excluded.append({"path": relative, "reason": "not_regular_file"})
            elif path.suffix.lower() not in EXTENSIONS:
                excluded.append({"path": relative, "reason": "unsupported_extension"})
            else:
                files.append(path)
    return sorted(files), sorted(excluded, key=lambda x: x["path"])


def decode(path):
    data = path.read_bytes()
    if not data:
        raise ValueError("Empty image file")
    # Same decoding semantics as ocr_det.read_image, from the hashed bytes.
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image")
    pixel_hash = digest(str(tuple(image.shape)).encode("ascii") + image.tobytes(order="C"))
    return image, digest(data), pixel_hash, len(data)


def inventory(root, files, excluded, profile, known_files=(), known_pixels=()):
    rows, file_groups, pixel_groups = [], defaultdict(list), defaultdict(list)
    shapes = Counter()
    for path in files:
        row = {"path": path.relative_to(root).as_posix()}
        try:
            image, sha, pixels, size = decode(path)
            row.update(image_sha256=sha, decoded_pixels_sha256=pixels, file_bytes=size,
                       image_shape=list(image.shape),
                       matches_known_used_image=sha in known_files or pixels in known_pixels)
            file_groups[sha].append(row["path"])
            pixel_groups[pixels].append(row["path"])
            tensor = preprocess(image)  # Execute the existing preprocessing, no copied resize formula.
            shape = list(tensor.shape)
            within = all(lo <= n <= hi for lo, n, hi in zip(profile[0], shape, profile[2]))
            row.update(det_input_shape=shape, det_input_sha256=digest(tensor.tobytes(order="C")),
                       within_declared_det_profile=within, status="preprocessed")
            shapes["x".join(map(str, shape))] += 1
        except (OSError, ValueError, cv2.error) as error:
            row.update(status="error", error=f"{type(error).__name__}: {error}")
        rows.append(row)
    errors = [r["path"] for r in rows if r["status"] == "error"]
    outside = [r["path"] for r in rows if r.get("within_declared_det_profile") is False]
    known = [r["path"] for r in rows if r.get("matches_known_used_image")]
    duplicates = [v for v in pixel_groups.values() if len(v) > 1]
    status = ("invalid_inventory" if errors or not rows else
              "completed_with_findings_not_inference_validated" if outside or known or duplicates or excluded else
              "completed_not_inference_validated")
    return {"status": status, "records": rows, "excluded_files": excluded,
            "summary": {"image_file_count": len(rows), "preprocessed_count": len(rows)-len(errors),
                        "error_paths": errors, "outside_declared_det_profile_paths": outside,
                        "known_used_image_paths": known, "det_input_shape_counts": dict(sorted(shapes.items())),
                        "duplicate_file_groups": [v for v in file_groups.values() if len(v) > 1],
                        "duplicate_decoded_pixel_groups": duplicates, "excluded_count": len(excluded)},
            "recognition_widths": {"status": "not_measured", "reason": "Requires detection boxes; no model inference performed"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Recursively inventory this directory")
    parser.add_argument("--det-report", type=Path, required=True, help="Audited live pipeline report, NOT an engine")
    parser.add_argument("--used-image", type=Path, action="append", default=[], help="Previously used image; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root, output = args.images.resolve(), args.output.resolve()
    if not root.is_dir():
        parser.error("Image directory does not exist")
    if args.output.exists() or args.output.is_symlink():
        parser.error("Report exists; use a new output filename")
    if output.is_relative_to(root):
        parser.error("Place output outside the image directory")
    try:
        source_bytes = args.det_report.read_bytes()
        source = json.loads(source_bytes)
        profile = detector_profile(source)
        known_files = {source["image_sha256"]}
        known_pixels = set()
        for path in args.used_image:
            _, sha, pixels, _ = decode(path)
            known_files.add(sha)
            known_pixels.add(pixels)
        files, excluded = discover(root)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, cv2.error) as error:
        parser.error(str(error))
    cv2.setNumThreads(1)
    report = inventory(root, files, excluded, profile, known_files, known_pixels)
    report.update(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                  image_root=str(root), detector_report_sha256=digest(source_bytes),
                  declared_det_engine_sha256=source["det_engine_sha256"], declared_det_profile=profile,
                  known_used_file_sha256=sorted(known_files), known_used_pixels_sha256=sorted(known_pixels),
                  python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
                  opencv_threads=cv2.getNumThreads(), supported_extensions=sorted(EXTENSIONS),
                  preprocessing_source_sha256=digest(Path(__file__).with_name("ocr_det.py").read_bytes()),
                  limitations="CPU decode/preprocess only. Profile copied from supplied report; engine not loaded. "
                  "No rec widths, text, GT accuracy, latency or VRAM measured. No image resized on disk. "
                  "Known-image and exact decoded-pixel matching cannot certify independence or detect all near-duplicates. "
                  "TIFF/PDF/other extensions and links are listed as excluded, never silently accepted.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "summary": report["summary"],
                      "recognition_widths": report["recognition_widths"]}, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")
    return 2 if report["status"] == "invalid_inventory" else 0


if __name__ == "__main__":
    raise SystemExit(main())
