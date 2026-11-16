"""Manifest-bound batch detector comparison + CPU crop-width inventory, NOT OCR accuracy."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from inventory_images import decode, digest, discover
from ocr_det import preprocess, sha256_file
from ocr_rec import crop_text_region, preprocess_rec
from compare_rec_buckets import pad_white, validate_buckets


def check_manifest(root, manifest):
    rows = manifest.get("records", [])
    if (manifest.get("status") not in ("completed_not_inference_validated",
                                      "completed_with_findings_not_inference_validated")
            or not rows or manifest.get("excluded_files")):
        raise ValueError("Need a nonempty, complete image inventory without excluded files")
    files, excluded = discover(root)
    expected = [r.get("path") for r in rows]
    actual = [p.relative_to(root).as_posix() for p in files]
    if excluded or len(set(expected)) != len(expected) or set(expected) != set(actual):
        raise ValueError("Dataset file set changed or contains excluded files; inventory again")
    for row in rows:
        if row.get("status") != "preprocessed":
            raise ValueError("Inventory contains incomplete image rows")
        image, sha, _, _ = decode(root / row["path"])
        tensor = preprocess(image)
        if (sha != row["image_sha256"] or list(image.shape) != row["image_shape"]
                or list(tensor.shape) != row["det_input_shape"]
                or digest(tensor.tobytes(order="C")) != row["det_input_sha256"]):
            raise ValueError(f"Image/preprocessing differs from inventory: {row['path']}")
    return rows


def crop_inventory(image, boxes, buckets, rec_profile):
    rows = []
    for index, box in enumerate(boxes):
        row = {"index": index, "box": box}
        try:
            crop = crop_text_region(image, box)
            if crop is None:
                raise ValueError("Empty clipped crop")
            ideal_width = int(math.ceil(48 * (crop.shape[1] / float(crop.shape[0]))))
            tensor = preprocess_rec(crop)
            shape = list(tensor.shape)
            target = next((w for w in buckets if w >= shape[3]), None)
            inside = all(lo <= n <= hi for lo, n, hi in zip(rec_profile[0], shape, rec_profile[2]))
            row.update(status="inventoried", crop_shape=list(crop.shape),
                       aspect_width_before_legacy_cap=ideal_width, legacy_width_cap_applied=ideal_width > 3200,
                       input_shape=shape, input_sha256=digest(tensor.tobytes(order="C")),
                       within_declared_rec_profile=inside, bucket_width=target,
                       padded_within_declared_rec_profile=None, padded_input_sha256=None)
            if target is not None:
                padded = pad_white(tensor, target)
                row["padded_within_declared_rec_profile"] = all(
                    lo <= n <= hi for lo, n, hi in zip(rec_profile[0], padded.shape, rec_profile[2]))
                row["padded_input_sha256"] = digest(padded.tobytes(order="C"))
        except (ValueError, TypeError, OverflowError, cv2.error) as error:
            row.update(status="invalid_crop", error=f"{type(error).__name__}: {error}")
        rows.append(row)
    valid = [r for r in rows if r["status"] == "inventoried"]
    return {"records": rows, "summary": {
        "box_count": len(rows), "valid_crop_count": len(valid),
        "invalid_crop_indices": [r["index"] for r in rows if r["status"] != "inventoried"],
        "width_counts": dict(sorted(Counter(r["input_shape"][3] for r in valid).items())),
        "bucket_counts": dict(sorted(Counter(r["bucket_width"] for r in valid if r["bucket_width"] is not None).items())),
        "outside_rec_profile_indices": [r["index"] for r in valid if not r["within_declared_rec_profile"]],
        "no_bucket_indices": [r["index"] for r in valid if r["bucket_width"] is None],
        "padded_outside_profile_indices": [r["index"] for r in valid if r["padded_within_declared_rec_profile"] is False],
        "legacy_width_cap_indices": [r["index"] for r in valid if r["legacy_width_cap_applied"]]}}


def analyze_comparison(result, image, row, onnx_sha, engine_sha, buckets, rec_profile):
    if (result.get("status") != "finite_outputs_compared"
            or not all(result.get(k, {}).get("all_finite") for k in ("ort_output", "trt_output"))):
        raise ValueError("Detector comparison not finite/complete")
    if (result.get("image_sha256") != row["image_sha256"] or result.get("input_shape") != row["det_input_shape"]
            or result.get("ort_metadata", {}).get("model_sha256") != onnx_sha
            or result.get("trt_metadata", {}).get("engine_sha256") != engine_sha):
        raise ValueError("Detector report provenance mismatch")
    branches = {}
    for branch in ("ort", "trt"):
        boxes = result[branch+"_boxes"]
        if len(boxes) != result[branch+"_box_count"]:
            raise ValueError("Detector box count mismatch")
        branches[branch] = crop_inventory(image, boxes, buckets, rec_profile)
    if (result["ort_boxes"] == result["trt_boxes"]) != result["ordered_boxes_exactly_equal"]:
        raise ValueError("Detector equality summary mismatch")
    metrics = {k: result[k] for k in ("probability_mae", "probability_max_abs_error",
                "allclose_rtol_1e_3_atol_1e_4", "mask_disagreement_fraction_at_0_1",
                "ort_box_count", "trt_box_count", "ordered_boxes_exactly_equal")}
    return {"path": row["path"], "image_sha256": row["image_sha256"], "det_input_shape": row["det_input_shape"],
            "detection": metrics, "branches": branches}


def summarize(images):
    result = {"compared_image_count": len(images),
              "det_allclose_failed_paths": [r["path"] for r in images if not r["detection"]["allclose_rtol_1e_3_atol_1e_4"]],
              "det_box_difference_paths": [r["path"] for r in images if not r["detection"]["ordered_boxes_exactly_equal"]]}
    for branch in ("ort", "trt"):
        totals, widths, buckets, review, zero = 0, Counter(), Counter(), [], []
        for image in images:
            s = image["branches"][branch]["summary"]
            totals += s["box_count"]
            widths.update({int(k): v for k, v in s["width_counts"].items()})
            buckets.update({int(k): v for k, v in s["bucket_counts"].items()})
            if not s["box_count"]:
                zero.append(image["path"])
            if any(s[k] for k in ("invalid_crop_indices", "outside_rec_profile_indices", "no_bucket_indices",
                                   "padded_outside_profile_indices", "legacy_width_cap_indices")):
                review.append(image["path"])
        result[branch] = {"box_count": totals, "width_counts": dict(sorted(widths.items())),
                          "bucket_counts": dict(sorted(buckets.items())), "review_paths": review, "zero_box_paths": zero}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "inventory", "onnx", "engine", "rec-baseline", "output-dir"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--buckets", nargs="+", type=int, default=[64,128,160,256,512,960,1024])
    args = parser.parse_args(argv)
    root, out = args.images.resolve(), args.output_dir.resolve()
    if not root.is_dir() or out.exists() or args.output_dir.is_symlink() or out.is_relative_to(root):
        parser.error("Need existing images and a NEW output directory outside the dataset")
    cv2.setNumThreads(1)
    try:
        validate_buckets(args.buckets)
        if len(args.buckets) > 7:
            raise ValueError("Keep the current at-most-seven bucket constraint")
        manifest = json.loads(args.inventory.read_text(encoding="utf-8"))
        rows = check_manifest(root, manifest)
        baseline = json.loads(args.rec_baseline.read_text(encoding="utf-8"))
        rec = baseline["recognizer"]
        profile = rec["profile"]
        if (baseline.get("status") != "completed_consistent_not_accuracy_validated"
                or rec.get("backend") != "tensorrt" or profile != [[1,3,48,w] for w in (32,160,1024)]):
            raise ValueError("Expected the audited original TRT rec baseline with W=32..1024")
        onnx_sha, engine_sha = sha256_file(args.onnx), sha256_file(args.engine)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "inventory_sha256": sha256_file(args.inventory), "onnx_sha256": onnx_sha, "engine_sha256": engine_sha,
              "rec_baseline_sha256": sha256_file(args.rec_baseline), "declared_rec_engine_sha256": rec["engine_sha256"],
              "declared_rec_profile": profile, "buckets": args.buckets, "expected_image_count": len(rows), "images": [],
              "source_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in
                                ("scan_det_rec_shapes.py", "compare_trt.py", "ocr_det.py", "ocr_rec.py", "trt_backend.py")},
              "limitations": "One fresh child process per image; CPU ORT reference then TRT detector, not a benchmark. "
              "Rec crops/padding only: rec engine is NOT loaded and no text/field/GT accuracy is measured. "
              "Both detector branches inventoried independently; no index-based page policy. "
              "Legacy rec preprocessing caps aspect width at 3200; cap cases explicitly flagged, not silently accepted."}
    code = 0
    try:
        for index, row in enumerate(rows):
            report["last_started_image"] = row["path"]
            path = root / row["path"]
            image, sha, _, _ = decode(path)
            if sha != row["image_sha256"]:
                raise ValueError("Dataset changed during scan")
            target = out / f"det_{index:04d}.json"
            with (out / f"det_{index:04d}.log").open("x", encoding="utf-8") as log:
                process = subprocess.run([sys.executable, str(Path(__file__).with_name("compare_trt.py")),
                    "--onnx", str(args.onnx.resolve()), "--engine", str(args.engine.resolve()),
                    "--image", str(path), "--output", str(target)], stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError(f"Detector child failed ({process.returncode}); see {target.with_suffix('.log').name}")
            result = json.loads(target.read_text(encoding="utf-8"))
            sample = analyze_comparison(result, image, row, onnx_sha, engine_sha, args.buckets, profile)
            sample.update(comparison_file=target.name, comparison_sha256=sha256_file(target))
            report["images"].append(sample)
            print(f"Compared {index+1}/{len(rows)}: {row['path']}; ORT/TRT boxes "
                  f"{result['ort_box_count']}/{result['trt_box_count']}", flush=True)
        report["status"] = "completed_inventory_not_rec_validated"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="invalid_scan", error=f"{type(error).__name__}: {error}")
        code = 2
    report["summary"] = summarize(report["images"])
    with (out / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "summary": report["summary"], "error": report.get("error")}, indent=2))
    print(f"Saved: {out / 'summary.json'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
