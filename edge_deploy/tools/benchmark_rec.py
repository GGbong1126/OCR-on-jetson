"""One backend per process; warmed fixed-crop recognition, not full OCR/HTTP."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import cv2
import numpy as np

from compare_rec import verify_source
from ocr_det import read_image, sha256_file
from ocr_rec import Recognizer, TensorRTRecognizer, crop_text_region, preprocess_rec, decode_ctc


METRICS = ("crop_ms", "preprocess_ms", "inference_ms", "decode_ms", "crop_rec_ms")


def stats(values):
    values = list(values)
    if not values or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite timing samples")
    return {"count": len(values), "mean": float(np.mean(values)),
            **{f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)}}


def prepare_rows(source, image, image_sha, onnx_sha, dict_sha, branch):
    rows = verify_source(source, image_sha,
                         {"model_sha256": onnx_sha, "dictionary_sha256": dict_sha}, branch)
    for row in rows:
        if not isinstance(row.get("text"), str):
            raise ValueError("Source record must include baseline text")
        if (row["status"] == "empty_text") != (row["text"] == ""):
            raise ValueError("Source empty status and text disagree")
        crop = crop_text_region(image, row["box"])
        if crop is None:
            raise ValueError(f"Empty source crop at index {row['index']}")
        if list(preprocess_rec(crop).shape) != row.get("input_shape"):
            raise ValueError(f"Source input shape mismatch at index {row['index']}")
    return rows


def validate_profile(rows, profile):
    """Reject the whole dataset before spending time on partial GPU runs."""
    low, _, high = profile
    for row in rows:
        shape = row["input_shape"]
        if len(shape) != len(low) or any(not a <= s <= b for s, a, b in zip(shape, low, high)):
            raise ValueError(f"Region {row['index']} shape {shape} outside engine profile {profile}")


def run_cycle(rec, image, rows, phase, cycle, records):
    """Decode borrowed output immediately. Keep partial evidence on failure."""
    cycle_start = perf_counter()
    totals = dict.fromkeys(METRICS, 0.0)
    empty_indices = []
    for row in rows:
        result = {"phase": phase, "cycle": cycle, "index": row["index"],
                  "input_shape": row["input_shape"], "width": row["input_shape"][3]}
        try:
            start = perf_counter()
            crop = crop_text_region(image, row["box"])
            cropped = perf_counter()
            if crop is None:
                raise ValueError("Empty source crop")
            tensor = preprocess_rec(crop)
            prepared = perf_counter()
            if list(tensor.shape) != row["input_shape"]:
                raise ValueError("Preprocessed shape differs from source report")
            call_start = perf_counter()
            raw = rec.run_tensor(tensor)
            inferred = perf_counter()
            # Includes shape/finite checks; do not add a second large scan to the timer.
            text, score = decode_ctc(raw, rec.characters)
            decoded = perf_counter()
            result.update(crop_ms=(cropped-start)*1000,
                          preprocess_ms=(prepared-cropped)*1000,
                          inference_ms=(inferred-call_start)*1000,
                          decode_ms=(decoded-inferred)*1000,
                          crop_rec_ms=(decoded-start)*1000,
                          output_shape=list(raw.shape), all_finite=True,
                          text_equal=(text == row["text"]), empty_text=(text == ""))
            del raw, tensor, crop
            if not np.isfinite(score):
                raise ValueError("Non-finite decoded score")
            if not result["text_equal"]:
                raise ValueError("Decoded text differs from recorded CPU baseline; inspect with compare_rec.py")
            result["status"] = "consistent"
            if result["empty_text"]:
                empty_indices.append(row["index"])
            for key in METRICS:
                totals[key] += result[key]
        except Exception as error:
            result.update(status="error", error=f"{type(error).__name__}: {error}")
            records.append(result)
            raise
        records.append(result)
    return {"cycle": cycle, "region_count": len(rows), **totals,
            "cycle_wall_ms": (perf_counter()-cycle_start)*1000,
            "empty_indices": empty_indices}


def measure(rec, image, rows, warmup, repeat, report):
    if warmup < 1 or repeat < 1:
        raise ValueError("warmup and repeat must both be positive full-sequence cycles")
    report.update(warmup_records=[], warmup_cycles=[], records=[], cycles=[])
    for phase, count, record_key, cycle_key in (
            ("warmup", warmup, "warmup_records", "warmup_cycles"),
            ("measured", repeat, "records", "cycles")):
        for cycle in range(count):
            result = run_cycle(rec, image, rows, phase, cycle, report[record_key])
            report[cycle_key].append(result)
            # Print only between cycles, outside cycle timers.
            print(f"{phase.capitalize()} cycle {cycle+1}/{count}: "
                  f"{len(rows)} crops, inference sum {result['inference_ms']:.3f} ms", flush=True)
    report["cycle_summary_ms"] = {
        key: stats(c[key] for c in report["cycles"]) for key in (*METRICS, "cycle_wall_ms")}
    report["per_width_inference_ms"] = {
        str(width): stats(r["inference_ms"] for r in report["records"] if r["width"] == width)
        for width in sorted({r["width"] for r in report["records"]})}
    report["validation"] = {
        "warmup_checked_count": len(report["warmup_records"]),
        "measured_checked_count": len(report["records"]),
        "all_outputs_finite": True, "all_texts_equal_to_source": True,
        "source_empty_indices": [r["index"] for r in rows if r["text"] == ""],
        "note": "Includes shared empty strings; consistency is not ground-truth accuracy."}
    report["status"] = "completed_consistent_not_accuracy_validated"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["ort", "trt"], required=True)
    parser.add_argument("--model", required=True, help="ORT ONNX or TRT engine to measure")
    parser.add_argument("--onnx", required=True, help="Reference rec ONNX for source provenance; not loaded on TRT runs")
    parser.add_argument("--dict", dest="dictionary", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--det-report", required=True, help="Existing compare_ocr.py JSON with CPU rec text")
    parser.add_argument("--branch", choices=["reference", "candidate"], default="reference")
    parser.add_argument("--threads", type=int, default=2, help="ORT intra-op threads; ignored by TRT")
    parser.add_argument("--warmup", type=int, default=2, help="Complete crop-sequence warmup cycles")
    parser.add_argument("--repeat", type=int, default=10, help="Complete crop-sequence measured cycles")
    parser.add_argument("--run-note", default="", help="Optional power mode / competing load / run order note")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.threads, args.warmup, args.repeat) < 1:
        parser.error("threads, warmup and repeat must be positive")
    if args.output.exists():
        parser.error("Report exists; use a new output filename")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    report = {
        "status": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend, "source_branch": args.branch,
        "warmup_requested": args.warmup, "repeat_requested": args.repeat,
        "run_note": args.run_note, "python": platform.python_version(),
        "platform": platform.platform(), "opencv_threads": cv2.getNumThreads(),
        "timing_scope": {
            "inference_ms": "Completed run_tensor call; TRT includes host packing, shape setup, allocation if needed, H2D, enqueue, D2H, synchronization. Not kernel-only.",
            "decode_ms": "CTC including output shape and finite checks",
            "crop_rec_ms": "Per-region crop + preprocess + input shape check + completed call + CTC; sum per full crop sequence",
            "cycle_wall_ms": "Full crop sequence including per-region validation, record bookkeeping and buffer release; excludes between-cycle print",
            "excluded": "Image read/decode, model load/hash, source preflight, det, layout, NER, HTTP, JSON writing. Image is already in RAM.",
            "warmup": "Full original-order crop sequences, retained separately and excluded from measured summaries. No sorting/padding/batching changes.",
        },
        "limitations": "Single-image repeated fixed crops, not multi-image accuracy or full OCR. Width stats pool crops of that width and are diagnostic, not isolated operator benchmarks. Percentiles are exploratory; correlated repeats do not establish stable tail latency. No automatic speedup claim.",
    }
    rec = None
    try:
        source = json.loads(Path(args.det_report).read_text(encoding="utf-8"))
        report.update(source_report_sha256=sha256_file(args.det_report),
                      image_sha256=sha256_file(args.image),
                      reference_onnx_sha256=sha256_file(args.onnx),
                      dictionary_sha256=sha256_file(args.dictionary))
        if args.backend == "ort" and sha256_file(args.model) != report["reference_onnx_sha256"]:
            raise ValueError("ORT benchmark model differs from reference ONNX")
        image = read_image(args.image)
        rows = prepare_rows(source, image, report["image_sha256"], report["reference_onnx_sha256"],
                            report["dictionary_sha256"], args.branch)
        crop_set = [{k: r[k] for k in ("index", "box", "input_shape")} for r in rows]
        report.update(crop_set=crop_set,
                      crop_set_sha256=hashlib.sha256(json.dumps(crop_set, sort_keys=True).encode()).hexdigest(),
                      region_count=len(rows),
                      width_counts=dict(sorted(Counter(r["input_shape"][3] for r in rows).items())))
        rec = (Recognizer(args.model, args.dictionary, args.threads) if args.backend == "ort"
               else TensorRTRecognizer(args.model, args.dictionary))
        report["metadata"] = rec.metadata()
        if args.backend == "trt":
            validate_profile(rows, rec.profile)
            report["engine_provenance_note"] = "Reference ONNX hash is checked against source report; engine hash is recorded separately. This does not cryptographically prove engine was built from that ONNX; retain build log and compare_rec report."
        measure(rec, image, rows, args.warmup, args.repeat, report)
    except KeyboardInterrupt:
        report.update(status="interrupted", error="KeyboardInterrupt")
    except Exception as error:
        report.update(status="invalid_benchmark", error=f"{type(error).__name__}: {error}")
    finally:
        try:
            if rec is not None and hasattr(rec, "close"):
                rec.close()
        except Exception as error:
            report.update(status="invalid_benchmark", cleanup_error=f"{type(error).__name__}: {error}")
        finally:
            if report["status"] != "completed_consistent_not_accuracy_validated":
                report.pop("cycle_summary_ms", None)
                report.pop("per_width_inference_ms", None)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            print(f"Saved: {args.output.resolve()}", flush=True)
    print(json.dumps({k: report[k] for k in ("status", "error", "cleanup_error", "validation", "cycle_summary_ms")
                      if k in report}, indent=2))
    print("Single-image repeats: p95/p99 exploratory, not stable tail-latency estimates.")
    if report["status"] != "completed_consistent_not_accuracy_validated":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
