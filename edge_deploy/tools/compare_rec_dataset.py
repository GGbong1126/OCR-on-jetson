"""Scanned dataset: original CPU/TRT and TRT original/padded strict regressions.

One CPU call and two serial TRT calls per crop. No benchmark or page exception.
Checkpoint each image; save no recognized text. Existing outputs are never replaced.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2

from compare_rec import compare_rec_outputs
from compare_rec_buckets import bucket_width, decode_call, edit_distance, pad_white, validate_buckets
from compare_rec_expanded import inside, make_sample, summary as original_summary, validated_scan_rows
from inventory_images import decode, discover
from ocr_det import sha256_file
from ocr_rec import Recognizer, TensorRTRecognizer


SUCCESS = "completed_strict_consistent_not_accuracy_validated"
CHANGED = "completed_with_text_changes_not_accepted"
PROFILE = [[1, 3, 48, w] for w in (32, 160, 1536)]


def validate_acceptance(report, hashes):
    if report.get("status") != SUCCESS or report.get("error") or report.get("cleanup_error"):
        raise ValueError("Need successful expanded-profile original-width regression")
    for key in ("onnx", "engine", "dict", "scan_report", "baseline_report"):
        if report.get(key + "_sha256") != hashes[key + "_sha256"]:
            raise ValueError(f"Expanded acceptance {key} hash mismatch")
    count = report.get("expected_count", 0)
    computed = original_summary(report.get("records", []), count)
    if (computed["status"] != SUCCESS or report.get("summary") != computed
            or report.get("reference_count", 0) < 1 or report.get("overflow_count", 0) < 1
            or count != report["reference_count"] + report["overflow_count"]
            or report.get("candidate_metadata", {}).get("profile") != PROFILE):
        raise ValueError("Incomplete/inconsistent expanded acceptance records/profile")


def load_checked_image(root, entry):
    relative = Path(entry["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Scan path escapes image directory")
    image, sha, _, _ = decode(path)
    if sha != entry["image_sha256"]:
        raise ValueError("Scan image changed")
    return image


def samples_for_image(image, entry):
    for row in entry["branches"]["trt"]["records"]:
        yield make_sample(image, row, f"scan:{entry['path']}#{row['index']}",
                          expected_hash=row["input_sha256"])


def preflight(root, scan, buckets):
    files, excluded = discover(root)
    if ({p.relative_to(root).as_posix() for p in files} != {r["path"] for r in scan["images"]}
            or any(r["reason"] != "unsupported_extension" for r in excluded)):
        raise ValueError("Image file set/links differ from scan")
    if len(buckets) > 7 or any(not inside([1, 3, 48, w], PROFILE) for w in buckets):
        raise ValueError("Candidate needs at most seven buckets within expanded profile")
    counts = Counter()
    widths = []
    for entry in scan["images"]:
        image = load_checked_image(root, entry)
        for sample in samples_for_image(image, entry):
            shape = sample["input_shape"]
            if not inside(shape, PROFILE):
                raise ValueError("Original tensor outside expanded profile; no truncation")
            target = bucket_width(shape[3], buckets)
            counts[str(target)] += 1
            widths.append(shape[3])
    if not widths:
        raise ValueError("No recognition crops")
    return {"expected_count": len(widths), "bucket_counts": dict(counts),
            "padding_width_expansion_ratio": sum(int(k)*v for k, v in counts.items()) / sum(widths),
            "zero_box_paths": [e["path"] for e in scan["images"] if not e["branches"]["trt"]["records"]]}


def compare_sample(cpu, gpu, sample, buckets):
    row = {k: v for k, v in sample.items() if k not in ("tensor", "expected_text")}
    tensor = sample["tensor"]
    target = bucket_width(tensor.shape[3], buckets)
    padded = pad_white(tensor, target)
    row.update(padded_input_shape=list(padded.shape), padding_columns=target-tensor.shape[3],
               padded_input_sha256=hashlib.sha256(padded.tobytes(order="C")).hexdigest())
    try:
        reference = cpu.run_tensor(tensor)
        candidate = gpu.run_tensor(tensor)
        result = compare_rec_outputs(reference, candidate, cpu.characters)
        # Consume original borrowed GPU output before the padded call.
        ref_text = result.pop("reference_text", None)
        orig_text = result.pop("candidate_text", None)
        if result["status"] == "finite_outputs_compared":
            result.update(reference_text_length=len(ref_text), candidate_text_length=len(orig_text),
                          reference_empty=not ref_text, candidate_empty=not orig_text,
                          reference_matches_recorded=None)
        row["original"] = result
        del reference, candidate
        if result["status"] != "finite_outputs_compared":
            row["status"] = "invalid_original_output"
            return row
        row["padding"] = {}
        padded_text = decode_call(gpu, padded, row["padding"])
        # Different timestep lengths are expected. Decode full output; never trim.
        row["padding"].update(status="finite_outputs_compared", text_equal=padded_text == orig_text,
                              cpu_text_equal=padded_text == ref_text,
                              edit_count=edit_distance(orig_text, padded_text),
                              ascii_space_only_change=(padded_text != orig_text and
                                  padded_text.replace(" ", "") == orig_text.replace(" ", "")))
        row["status"] = "finite_outputs_compared"
    except Exception as error:
        row.update(status="error", error=f"{type(error).__name__}: {error}")
    return row


def summarize(records, expected):
    original = original_summary([{"id": r["id"], **r.get("original", {"status": "missing"})}
                                 for r in records], expected)
    valid = [r for r in records if r["status"] == "finite_outputs_compared"]
    changed = [r["id"] for r in valid if not r["padding"]["text_equal"]]
    incomplete = len(valid) != expected or len(records) != expected or not expected
    padded_status = "invalid_comparison" if incomplete else CHANGED if changed else SUCCESS
    status = ("invalid_comparison" if incomplete or original["status"] == "invalid_comparison"
              else CHANGED if original["status"] != SUCCESS or padded_status != SUCCESS else SUCCESS)
    return {"status": status, "expected_count": expected, "compared_count": len(records),
            "original_cpu_trt": original,
            "padding_trt": {"status": padded_status, "finite_count": len(valid),
                            "text_equal_count": sum(r["padding"]["text_equal"] for r in valid),
                            "changed_text_ids": changed,
                            "cpu_changed_text_ids": [r["id"] for r in valid if not r["padding"]["cpu_text_equal"]],
                            "ascii_space_only_change_ids": [r["id"] for r in valid if r["padding"]["ascii_space_only_change"]],
                            "empty_ids": [r["id"] for r in valid if r["padding"]["empty_text"]]},
            "invalid_ids": [r["id"] for r in records if r["status"] != "finite_outputs_compared"]}


def save_new(path, report):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("onnx", "engine", "dict", "images", "scan-report", "baseline-report", "acceptance-report", "output-dir"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--buckets", nargs="+", type=int, default=[64,128,160,256,512,960,1536])
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error("Need positive threads and a new output directory")
    validate_buckets(args.buckets)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    gpu = None
    report = {"status": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
              "buckets": args.buckets, "threads": args.threads, "expected_count": 0,
              "images": [], "records": [],
              "source_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in
                  ("compare_rec_dataset.py", "compare_rec_expanded.py", "compare_rec.py", "compare_rec_buckets.py",
                   "compare_trt.py", "inventory_images.py", "ocr_det.py", "ocr_rec.py", "trt_backend.py")},
              "note": "Strict text regression only; no GT accuracy, page exception or benchmark. "
                      "ASCII-space-only differences remain failures. No recognized text saved. "
                      "Fixed recorded crops; detection/layout/NER not rerun."}
    try:
        for name in ("onnx", "engine", "dict", "scan_report", "baseline_report", "acceptance_report"):
            report[name+"_sha256"] = sha256_file(getattr(args, name))
        load = lambda p: json.loads(p.read_text(encoding="utf-8"))
        scan, baseline, acceptance = load(args.scan_report), load(args.baseline_report), load(args.acceptance_report)
        validate_acceptance(acceptance, report)
        rows = validated_scan_rows(scan, baseline, report["baseline_report_sha256"], report["onnx_sha256"], report["dict_sha256"])
        root = args.images.resolve()
        report.update(preflight(root, scan, args.buckets))
        if len(rows) != report["expected_count"]:
            raise ValueError("Preflight crop count mismatch")
        print(f"Preflight: {len(scan['images'])} images, {len(rows)} crops; starting CPU/TRT + padding checks", flush=True)
        cpu = Recognizer(args.onnx, args.dict, args.threads)
        report["reference_metadata"] = cpu.metadata()
        gpu = TensorRTRecognizer(args.engine, args.dict)
        report["candidate_metadata"] = gpu.metadata()
        if gpu.profile != PROFILE or cpu.characters != gpu.characters:
            raise ValueError("Candidate profile/dictionary mismatch")
        for number, entry in enumerate(scan["images"], 1):
            report["last_started_image"] = entry["path"]
            image = load_checked_image(root, entry)
            page_records = []
            try:
                for sample in samples_for_image(image, entry):
                    row = compare_sample(cpu, gpu, sample, args.buckets)
                    page_records.append(row)
                    report["records"].append(row)
                    if len(page_records) % 10 == 0:
                        print(f"Image {number}/{len(scan['images'])}: {len(page_records)} crops; total {len(report['records'])}/{len(rows)}", flush=True)
            finally:
                filename = f"image_{number:04d}.json"
                save_new(args.output_dir / filename, {"path": entry["path"], "image_sha256": entry["image_sha256"],
                    "scan_report_sha256": report["scan_report_sha256"], "expected_count": len(entry["branches"]["trt"]["records"]),
                    "records": page_records})
                report["images"].append({"path": entry["path"], "report_file": filename,
                                         "sha256": sha256_file(args.output_dir / filename)})
            print(f"Completed image {number}/{len(scan['images'])}", flush=True)
        report["status"] = "comparison_finished"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="invalid_comparison", error=f"{type(error).__name__}: {error}")
    finally:
        if gpu is not None:
            try:
                gpu.close()
            except Exception as error:
                report.update(status="invalid_comparison", cleanup_error=f"{type(error).__name__}: {error}")
    report["summary"] = summarize(report["records"], report["expected_count"])
    if report["status"] == "comparison_finished":
        report["status"] = report["summary"]["status"]
    else:
        report["summary"]["status"] = "invalid_comparison"
    save_new(args.output_dir / "summary.json", report)
    print(json.dumps({k: report.get(k) for k in ("status", "summary", "error", "cleanup_error")}, indent=2))
    print(f"Saved: {(args.output_dir / 'summary.json').resolve()}")
    return 0 if report["status"] == SUCCESS else 2


if __name__ == "__main__":
    raise SystemExit(main())
