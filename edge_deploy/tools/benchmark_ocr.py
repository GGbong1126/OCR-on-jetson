"""Live det -> crop -> rec benchmark with both models resident.

Fixed detector artifact and CPU text reference isolate the rec backend change.
Not a complete OCR service, accuracy evaluation or peak-memory measurement.
"""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import cv2
import numpy as np

from benchmark_rec import prepare_rows, run_cycle, stats, validate_profile
from ocr_det import create_detector, read_image, sha256_file
from ocr_rec import Recognizer, TensorRTRecognizer


METRICS = ("det_preprocess_ms", "det_inference_ms", "det_postprocess_ms", "det_pipeline_ms",
           "rec_crop_ms", "rec_preprocess_ms", "rec_inference_ms", "rec_decode_ms",
           "rec_crop_rec_ms", "rec_cycle_wall_ms", "pipeline_wall_ms")
MEMORY_QUERY = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits"]


def parse_memory_csv(output):
    devices = []
    for fields in csv.reader(io.StringIO(output), skipinitialspace=True):
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError("Unexpected nvidia-smi CSV columns")
        device = {"nvidia_smi_index": fields[0].strip(), "name": fields[1].strip()}
        for key, value in zip(("total_mib", "used_mib", "free_mib"), fields[2:]):
            try:
                number = float(value)
                device[key] = number if np.isfinite(number) and number >= 0 else None
            except ValueError:
                device[key] = None  # N/A is unknown, never zero.
        devices.append(device)
    if not devices:
        raise ValueError("No GPU rows returned")
    return devices


def memory_snapshot(label, enabled=True):
    sample = {"label": label, "created_utc": datetime.now(timezone.utc).isoformat()}
    if not enabled:
        return {**sample, "available": False, "reason": "disabled or CPU-only run"}
    try:
        result = subprocess.run(MEMORY_QUERY, capture_output=True, text=True, check=True,
                                timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        devices = parse_memory_csv(result.stdout)
        return {**sample, "available": True, "devices": devices,
                "memory_metrics_available": all(d[k] is not None for d in devices
                                                for k in ("total_mib", "used_mib", "free_mib"))}
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {**sample, "available": False, "reason": f"{type(error).__name__}: {error}"}


def verify_detector(source_branch, backend, artifact_sha):
    key, expected_backend = (("engine_sha256", "tensorrt") if backend == "trt"
                             else ("model_sha256", "ort_cpu"))
    meta = source_branch.get("detector", {})
    if meta.get("backend") != expected_backend or meta.get(key) != artifact_sha:
        raise ValueError("Detector backend/artifact differs from source branch; keep detector fixed")
    shape = source_branch.get("input_shape")
    if not isinstance(shape, list) or len(shape) != 4 or shape[:2] != [1, 3]:
        raise ValueError("Invalid source detection input shape")
    return shape


def run_pipeline(detector, rec, image, source_rows, expected_det_shape, phase, cycle, sample):
    sample.update(phase=phase, cycle=cycle, records=[])
    try:
        start = perf_counter()
        boxes, det_timing, det_shape = detector.predict(image)
        actual_boxes = [np.asarray(box).tolist() for box in boxes]
        sample.update(det_input_shape=det_shape, boxes=actual_boxes)
        if list(det_shape) != expected_det_shape:
            raise ValueError("Live detector input shape differs from source")
        if len(boxes) != len(source_rows):
            raise ValueError("Live detector region count differs from fixed-detector baseline")
        changed = [r["index"] for box, r in zip(actual_boxes, source_rows)
                   if not np.array_equal(box, r["box"])]
        sample["changed_box_indices"] = changed
        if changed:
            raise ValueError("Live detector boxes/order changed; controlled backend benchmark invalid, not an accuracy judgement")
        # Use newly detected boxes, not a cached detection result from the report.
        live_rows = [{**row, "box": box} for row, box in zip(source_rows, actual_boxes)]
        rec_timing = run_cycle(rec, image, live_rows, phase, cycle, sample["records"])
        end = perf_counter()
        sample["timings"] = {
            **{"det_" + key: value for key, value in det_timing.items()},
            **{"rec_" + key: rec_timing[key] for key in
               ("crop_ms", "preprocess_ms", "inference_ms", "decode_ms", "crop_rec_ms", "cycle_wall_ms")},
            "pipeline_wall_ms": (end-start)*1000,
        }
        if not all(np.isfinite(sample["timings"][key]) and sample["timings"][key] >= 0 for key in METRICS):
            raise ValueError("Invalid pipeline timings")
        sample.update(status="consistent", region_count=len(boxes),
                      empty_indices=rec_timing["empty_indices"])
    except Exception as error:
        sample.update(status="error", error=f"{type(error).__name__}: {error}")
        raise


def measure_pipeline(detector, rec, image, rows, det_shape, warmup, repeat, report, snapshot=None):
    if not rows or min(warmup, repeat) < 1:
        raise ValueError("Need nonempty source crops and positive warmup/repeat")
    report.update(validation_run={}, warmup_cycles=[], cycles=[])
    print("Validating live det + crop + rec with both models resident ...", flush=True)
    run_pipeline(detector, rec, image, rows, det_shape, "validation", 0, report["validation_run"])
    print(f"Validation passed: {len(rows)} regions, text consistent with CPU source", flush=True)
    if snapshot:
        snapshot("after_validation")  # Outside timers; subsequent warmups run before measurement.
    for phase, count, key in (("warmup", warmup, "warmup_cycles"), ("measured", repeat, "cycles")):
        for cycle in range(count):
            sample = {}
            report[key].append(sample)  # Keep partial current cycle on error or Ctrl+C.
            run_pipeline(detector, rec, image, rows, det_shape, phase, cycle, sample)
            print(f"{phase.capitalize()} {cycle+1}/{count}: {len(rows)} regions, "
                  f"pipeline {sample['timings']['pipeline_wall_ms']:.3f} ms", flush=True)
    report["summary_ms"] = {
        key: stats(s["timings"][key] for s in report["cycles"]) for key in METRICS}
    report["validation"] = {
        "validation_rec_calls": len(rows), "warmup_rec_calls": len(rows)*warmup,
        "measured_rec_calls": len(rows)*repeat,
        "live_det_calls": 1+warmup+repeat,
        "all_boxes_equal_to_source": True, "all_outputs_finite": True,
        "all_texts_equal_to_source": True,
        "source_empty_indices": [r["index"] for r in rows if r["text"] == ""],
        "note": "Same detector artifact, exact boxes/order and CPU text baseline. Shared empties are not accuracy.",
    }
    report["status"] = "completed_consistent_not_accuracy_validated"


def close_models(rec, detector):
    errors = []
    for label, model in (("rec", rec), ("det", detector)):
        try:
            if model is not None and hasattr(model, "close"):
                model.close()
        except Exception as error:
            errors.append({"model": label, "error": f"{type(error).__name__}: {error}"})
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--det-model", required=True)
    parser.add_argument("--det-backend", choices=["trt", "ort"], default="trt")
    parser.add_argument("--rec-model", required=True)
    parser.add_argument("--rec-backend", choices=["trt", "ort"], required=True)
    parser.add_argument("--rec-onnx", required=True, help="CPU reference ONNX hash; not loaded in TRT mode")
    parser.add_argument("--dict", dest="dictionary", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-report", required=True, help="compare_ocr.py report with CPU rec baseline")
    parser.add_argument("--branch", choices=["reference", "candidate"], default="reference")
    parser.add_argument("--threads", type=int, default=2, help="ORT threads only")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--skip-memory", action="store_true", help="Skip optional device-wide nvidia-smi snapshots")
    parser.add_argument("--run-note", default="", help="Actual power/load/order conditions; avoid private data")
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
        "python": platform.python_version(), "platform": platform.platform(),
        "opencv_threads": cv2.getNumThreads(), "batch_size": 1,
        "det_backend": args.det_backend, "rec_backend": args.rec_backend,
        "source_branch": args.branch, "warmup_requested": args.warmup,
        "repeat_requested": args.repeat, "run_note": args.run_note,
        "memory_snapshots": [], "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "memory_scope": "Optional nvidia-smi snapshots of all visible device rows, outside timers, not per-process/peak memory. Other processes/driver/OS may contribute; N/A is unknown. CUDA device numbering can differ. No queries between measured cycles.",
        "timing_scope": {
            "pipeline_wall_ms": "Live det preprocess + completed det call + postprocess + box checks + live crop/preprocess + completed rec calls + CTC/finite/text checks + record bookkeeping. Both models resident; sequential, not concurrent kernels.",
            "rec_cycle_wall_ms": "Same crop/rec loop as benchmark_rec, including text validation and record handling",
            "inference_ms": "Host completed calls; TRT includes shape setup, required allocation/packing, H2D, execution, D2H and sync. Not kernel-only.",
            "excluded": "Read/decode image, model load, hash, preflight, memory probes, between-cycle print, JSON, layout/NER/HTTP/queueing",
            "validation_run": "First live correctness pass after model load, excluded from warmup/measured summaries",
        },
        "limitations": "Single image in RAM, one pipeline per process; not full server, ground-truth accuracy or stable tail latency. Exact boxes constrain this fixed-detector experiment only. JSON omits recognized text; review errors/notes before sharing.",
    }
    memory_enabled = not args.skip_memory and "trt" in (args.det_backend, args.rec_backend)

    def snapshot(label):
        report["memory_snapshots"].append(memory_snapshot(label, memory_enabled))

    detector = rec = None
    try:
        source = json.loads(Path(args.source_report).read_text(encoding="utf-8"))
        report.update(source_report_sha256=sha256_file(args.source_report),
                      image_sha256=sha256_file(args.image),
                      reference_rec_onnx_sha256=sha256_file(args.rec_onnx),
                      dictionary_sha256=sha256_file(args.dictionary),
                      det_artifact_sha256=sha256_file(args.det_model))
        if args.rec_backend == "ort" and sha256_file(args.rec_model) != report["reference_rec_onnx_sha256"]:
            raise ValueError("CPU rec model differs from reference ONNX")
        image = read_image(args.image)
        rows = prepare_rows(source, image, report["image_sha256"], report["reference_rec_onnx_sha256"],
                            report["dictionary_sha256"], args.branch)
        det_shape = verify_detector(source[args.branch], args.det_backend, report["det_artifact_sha256"])
        crop_set = [{k: r[k] for k in ("index", "box", "input_shape")} for r in rows]
        report.update(region_count=len(rows), image_shape=list(image.shape),
                      crop_set_sha256=hashlib.sha256(json.dumps(crop_set, sort_keys=True).encode()).hexdigest())
        snapshot("before_model_load")
        detector = create_detector(args.det_backend, args.det_model, args.threads)
        report["detector"] = detector.metadata()
        snapshot("after_detector_load")
        rec = (Recognizer(args.rec_model, args.dictionary, args.threads) if args.rec_backend == "ort"
               else TensorRTRecognizer(args.rec_model, args.dictionary))
        report["recognizer"] = rec.metadata()
        report["engine_provenance_note"] = "Detector artifact matches source report. Rec engine hash recorded separately; retain its build log and compare_rec result, ONNX path alone does not prove engine origin."
        snapshot("after_both_models_load")
        if args.rec_backend == "trt":
            validate_profile(rows, rec.profile)
        measure_pipeline(detector, rec, image, rows, det_shape, args.warmup, args.repeat, report, snapshot)
        snapshot("after_measured_cycles")
    except KeyboardInterrupt:
        report.update(status="interrupted", error="KeyboardInterrupt")
    except Exception as error:
        report.update(status="invalid_benchmark", error=f"{type(error).__name__}: {error}")
    finally:
        errors = close_models(rec, detector)
        rec = detector = None
        gc.collect()
        if errors:
            report.update(status="invalid_benchmark", cleanup_errors=errors)
        snapshot("after_model_close")
        if report["status"] != "completed_consistent_not_accuracy_validated":
            report.pop("summary_ms", None)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"Saved: {args.output.resolve()}", flush=True)
    print(json.dumps({k: report[k] for k in ("status", "error", "cleanup_errors", "validation", "summary_ms", "memory_snapshots")
                      if k in report}, indent=2))
    print("Single-image repeats: p95/p99 exploratory. Memory snapshots are not process peak VRAM.")
    if report["status"] != "completed_consistent_not_accuracy_validated":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
