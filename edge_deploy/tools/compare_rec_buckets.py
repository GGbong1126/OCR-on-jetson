"""Compare original vs right-white-padded recognition using ONE TRT context.

Correctness diagnostic only: all recorded crops, original order, two calls/crop.
No extra contexts, benchmark, output rewriting/trimming or GT claims.
Optional explicit page-number annotations allow ASCII-space-only differences.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from benchmark_rec import prepare_rows, validate_profile
from ocr_det import read_image, sha256_file
from ocr_rec import TensorRTRecognizer, crop_text_region, preprocess_rec, decode_ctc
from profile_rec_shapes import verify_baseline
from ocr_text_policy import page_number_equal, validate_annotations


SUCCESS = "completed_consistent_not_accuracy_validated"
CHANGED = "completed_with_text_changes_not_accepted"
POLICY_SUCCESS = "completed_policy_consistent_not_accuracy_validated"


def validate_buckets(buckets):
    if not buckets or any(type(w) is not int or w < 32 for w in buckets):
        raise ValueError("Bucket widths must be integers >= 32")
    if list(buckets) != sorted(set(buckets)):
        raise ValueError("Bucket widths must be strictly increasing without duplicates")


def bucket_width(width, buckets):
    for target in buckets:
        if target >= width:
            return target
    raise ValueError(f"Width {width} exceeds largest bucket {buckets[-1]}; no truncation/fallback")


def pad_white(tensor, width):
    if (tensor.dtype != np.float32 or tensor.ndim != 4 or tensor.shape[:3] != (1, 3, 48)
            or tensor.shape[3] < 32 or not np.isfinite(tensor).all()):
        raise ValueError("Expected finite float32 [1,3,48,W], W >= 32")
    if type(width) is not int or width < tensor.shape[3]:
        raise ValueError("Padding cannot shrink or truncate the input")
    if width == tensor.shape[3]:
        return tensor
    padded = np.full((1, 3, 48, width), 1.0, dtype=np.float32)
    padded[:, :, :, :tensor.shape[3]] = tensor
    return padded


def make_plan(rows, buckets, profile):
    validate_buckets(buckets)
    validate_profile(rows, profile)
    # Reject even unused invalid configured buckets before any inference.
    validate_profile([{"index": f"bucket_{w}", "input_shape": [1, 3, 48, w]}
                      for w in buckets], profile)
    return [{"index": row["index"], "original_input_shape": row["input_shape"],
             "padded_input_shape": [1, 3, 48, bucket_width(row["input_shape"][3], buckets)]}
            for row in rows]


def edit_distance(a, b):
    """Raw Python-character edit count; neither normalized text nor GT CER."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def decode_call(rec, tensor, result):
    start = perf_counter()
    raw = rec.run_tensor(tensor)
    result["call_ms"] = (perf_counter() - start) * 1000
    result["output_shape"] = list(raw.shape)
    result["all_finite"] = bool(np.isfinite(raw).all())
    if raw.dtype != np.float32:
        raise ValueError("Expected FP32 engine output")
    # Decode the COMPLETE borrowed output before the next run overwrites it.
    text, score = decode_ctc(raw, rec.characters)
    if not np.isfinite(score):
        raise ValueError("Non-finite decoded score")
    result.update(text_length=len(text), empty_text=(text == ""), score=float(score))
    return text


def compare_rows(rec, image, rows, plan, report, roles=None):
    if len(rows) != len(plan) or not rows:
        raise ValueError("Expected a complete nonempty crop plan")
    report["records"] = []
    roles = roles or {}
    for position, (row, item) in enumerate(zip(rows, plan)):
        result = {**item, "status": "started", "original": {}, "padded": {}}
        try:
            if row["index"] != item["index"]:
                raise ValueError("Source and plan index differ")
            tensor = preprocess_rec(crop_text_region(image, row["box"]))
            if list(tensor.shape) != item["original_input_shape"]:
                raise ValueError("Original preprocessed shape differs from source")
            padded = pad_white(tensor, item["padded_input_shape"][3])
            result.update(padding_columns=padded.shape[3] - tensor.shape[3],
                          width_expansion_ratio=padded.shape[3] / tensor.shape[3],
                          original_tensor_sha256=hashlib.sha256(tensor.tobytes(order="C")).hexdigest(),
                          padded_tensor_sha256=hashlib.sha256(padded.tobytes(order="C")).hexdigest())
            original_text = decode_call(rec, tensor, result["original"])
            padded_text = decode_call(rec, padded, result["padded"])
            result.update(original_equals_source=(original_text == row["text"]),
                          padded_equals_original=(padded_text == original_text),
                          padded_equals_source=(padded_text == row["text"]),
                          original_to_padded_edit_count=edit_distance(original_text, padded_text),
                          source_to_original_edit_count=edit_distance(row["text"], original_text),
                          status="finite_pair_compared")
            role = roles.get(row["index"], "strict")
            if role not in ("strict", "page_number"):
                raise ValueError("Unsupported comparison role")
            equal = page_number_equal if role == "page_number" else lambda a, b: a == b
            result.update(field_role=role,
                          policy_padded_equals_original=equal(original_text, padded_text),
                          policy_padded_equals_source=equal(row["text"], padded_text))
            result["tolerated_page_number_space_change"] = (
                role == "page_number" and result["original_equals_source"]
                and not result["padded_equals_original"]
                and result["policy_padded_equals_original"]
                and result["policy_padded_equals_source"])
        except BaseException as error:
            result.update(status="error", error=f"{type(error).__name__}: {error}")
            report["records"].append(result)
            raise  # CUDA/invalid-output failures stop: don't continue a broken context.
        report["records"].append(result)
        if (position + 1) % 10 == 0 or position + 1 == len(rows):
            print(f"Compared {position + 1}/{len(rows)} crops (original + padded)", flush=True)
    records = report["records"]
    summary = {
        "region_count": len(rows), "finite_pair_count": len(records),
        "original_source_changed_indices": [r["index"] for r in records if not r["original_equals_source"]],
        "padded_original_changed_indices": [r["index"] for r in records if not r["padded_equals_original"]],
        "padded_source_changed_indices": [r["index"] for r in records if not r["padded_equals_source"]],
        "source_empty_indices": [r["index"] for r in rows if r["text"] == ""],
        "original_empty_indices": [r["index"] for r in records if r["original"]["empty_text"]],
        "padded_empty_indices": [r["index"] for r in records if r["padded"]["empty_text"]],
        "unchanged_width_indices": [r["index"] for r in records if not r["padding_columns"]],
        "bucket_counts": dict(sorted(Counter(r["padded_input_shape"][3] for r in records).items())),
        "padded_original_text_equal_count": sum(r["padded_equals_original"] for r in records),
        "policy_padded_original_equal_count": sum(r["policy_padded_equals_original"] for r in records),
        "policy_changed_indices": [r["index"] for r in records
                                   if not (r["policy_padded_equals_original"] and r["policy_padded_equals_source"])],
        "tolerated_page_number_space_change_indices": [r["index"] for r in records
                                                       if r["tolerated_page_number_space_change"]],
        "padding_width_expansion_ratio": sum(r["padded_input_shape"][3] for r in records) /
                                         sum(r["original_input_shape"][3] for r in records),
        "note": "Text consistency including shared empty strings is not ground-truth accuracy. No CER, numeric allclose or benchmark is claimed. Different output timestep lengths are allowed; full outputs are decoded without trimming.",
    }
    if summary["original_source_changed_indices"]:
        report["status"] = "completed_with_reference_drift_not_accepted"
    elif summary["padded_original_changed_indices"]:
        report["status"] = CHANGED
    else:
        report["status"] = SUCCESS
    report["strict_status"] = report["status"]
    summary["strict_consistency_passed"] = report["strict_status"] == SUCCESS
    summary["policy_consistency_passed"] = not (
        summary["original_source_changed_indices"] or summary["policy_changed_indices"])
    if report["status"] == CHANGED and summary["policy_consistency_passed"]:
        report["status"] = POLICY_SUCCESS
    report["summary"] = summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("engine", "onnx", "image", "source-report", "baseline-report"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--dict", dest="dictionary", required=True)
    parser.add_argument("--branch", choices=["reference", "candidate"], default="reference")
    parser.add_argument("--buckets", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--field-annotations", type=Path,
                        help="Hash-bound manual page-number annotations; default is strict everywhere")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        validate_buckets(args.buckets)
    except ValueError as error:
        parser.error(str(error))
    if args.output.exists():
        parser.error("Report exists; use a new output filename")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    report = {
        "status": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
        "buckets": args.buckets, "records": [], "batch_size": 1, "context_count": 1,
        "padding": "Preserve original preprocessed tensor; right pad +1.0 (normalized white). No stretch, truncation, sorting, output rewriting or CTC timestep trimming.",
        "text_policy": {"default": "strict", "page_number": "Only explicitly annotated crops: ignore U+0020; preserve all other code points/order. Original vs source remains strict. No output rewriting or GT accuracy claim."},
        "timing_scope": "One original then one padded call per crop, no warmup: diagnostic call_ms only; excludes crop/preprocess/padding/hashes/CTC, det, layout, NER, HTTP. NOT benchmark or speedup.",
        "privacy": "No recognized text stored; only lengths, empty/equality flags, raw edit counts, scores, shapes and input hashes.",
    }
    rec = None
    try:
        source = json.loads(Path(args.source_report).read_text(encoding="utf-8"))
        baseline = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
        provenance = dict(source_branch=args.branch, source_report_sha256=sha256_file(args.source_report),
                          image_sha256=sha256_file(args.image), reference_rec_onnx_sha256=sha256_file(args.onnx),
                          dictionary_sha256=sha256_file(args.dictionary))
        image = read_image(args.image)
        rows = prepare_rows(source, image, provenance["image_sha256"],
                            provenance["reference_rec_onnx_sha256"], provenance["dictionary_sha256"], args.branch)
        crop_set = [{k: r[k] for k in ("index", "box", "input_shape")} for r in rows]
        provenance["crop_set_sha256"] = hashlib.sha256(json.dumps(crop_set, sort_keys=True).encode()).hexdigest()
        report.update(provenance, engine_sha256=sha256_file(args.engine),
                      baseline_report_sha256=sha256_file(args.baseline_report), expected_region_count=len(rows))
        verify_baseline(baseline, provenance, report["engine_sha256"])
        roles = {}
        if args.field_annotations is not None:
            annotation = json.loads(args.field_annotations.read_text(encoding="utf-8"))
            roles = validate_annotations(annotation, provenance, rows)
            report["field_annotations"] = annotation
            report["field_annotations_sha256"] = sha256_file(args.field_annotations)
        rec = TensorRTRecognizer(args.engine, args.dictionary)
        report["metadata"] = rec.metadata()
        report["plan"] = make_plan(rows, args.buckets, rec.profile)
        compare_rows(rec, image, rows, report["plan"], report, roles)
    except KeyboardInterrupt:
        report.update(status="interrupted", error="KeyboardInterrupt")
    except Exception as error:
        report.update(status="invalid_comparison", error=f"{type(error).__name__}: {error}")
    finally:
        try:
            if rec is not None:
                rec.close()
        except Exception as error:
            report.update(status="invalid_comparison", cleanup_error=f"{type(error).__name__}: {error}")
        finally:
            if report["status"] in ("invalid_comparison", "interrupted"):
                report.pop("summary", None)
                report.pop("strict_status", None)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            print(f"Saved: {args.output.resolve()}", flush=True)
    print(json.dumps({k: report[k] for k in ("status", "strict_status", "error", "cleanup_error", "summary") if k in report}, indent=2))
    print("Text-only regression diagnostic, not ground-truth accuracy or performance benchmark.")
    if report["status"] not in (SUCCESS, POLICY_SUCCESS):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
