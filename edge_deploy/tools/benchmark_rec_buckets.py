"""One mode/process: full crop-sequence FP32 benchmark, NOT detection/full OCR."""
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

from benchmark_rec import prepare_rows, stats
from compare_rec_buckets import make_plan, pad_white, validate_buckets, SUCCESS, POLICY_SUCCESS
from ocr_det import read_image, sha256_file
from ocr_rec import TensorRTRecognizer, crop_text_region, preprocess_rec, decode_ctc
from ocr_text_policy import validate_annotations, page_number_equal
from profile_rec_shapes import verify_baseline
from trt_host_profile import HostProfiler
from trt_bucket_contexts import BucketContexts

MODES = ("original-single", "padded-single", "padded-fixed")
TIMINGS = ("crop_preprocess_ms", "padding_ms", "inference_ms", "decode_ms", "crop_rec_ms")


def verify_acceptance(accepted, provenance, engine_sha, annotation_sha, buckets, rows):
    if accepted.get("status") not in (SUCCESS, POLICY_SUCCESS):
        raise ValueError("Expected successful bucket regression report")
    for key, value in {**provenance, "engine_sha256": engine_sha,
                       "field_annotations_sha256": annotation_sha, "buckets": buckets}.items():
        if accepted.get(key) != value:
            raise ValueError(f"Bucket regression provenance mismatch: {key}")
    records = accepted.get("records", [])
    if [r.get("index") for r in records] != [r["index"] for r in rows]:
        raise ValueError("Bucket regression incomplete or wrong crop order")
    for row, result in zip(rows, records):
        if (result.get("status") != "finite_pair_compared" or
                not result.get("original_equals_source") or
                not result.get("policy_padded_equals_original") or
                not result.get("policy_padded_equals_source") or
                result.get("original_input_shape") != row["input_shape"] or
                not result.get("original", {}).get("all_finite") or
                not result.get("padded", {}).get("all_finite")):
            raise ValueError("Bucket regression row not accepted")


def verify_inputs(image, rows, plan, accepted):
    evidence = []
    for row, item, previous in zip(rows, plan, accepted["records"]):
        tensor = preprocess_rec(crop_text_region(image, row["box"]))
        padded = pad_white(tensor, item["padded_input_shape"][3])
        hashes = {"original_tensor_sha256": hashlib.sha256(tensor.tobytes(order="C")).hexdigest(),
                  "padded_tensor_sha256": hashlib.sha256(padded.tobytes(order="C")).hexdigest()}
        if previous.get("padded_input_shape") != item["padded_input_shape"] or any(
                previous.get(k) != v for k, v in hashes.items()):
            raise ValueError("Preflight tensor differs from accepted bucket regression")
        evidence.append({"index": row["index"], **hashes})
    return evidence


def run_cycle(profiler, characters, image, rows, plan, roles, mode, phase, cycle, records):
    start_cycle = perf_counter()
    totals = dict.fromkeys(TIMINGS, 0.0)
    changed, empty = [], []
    for row, item in zip(rows, plan):
        result = {"index": row["index"], "cycle": cycle, "phase": phase}
        profiler.last_record = None
        try:
            started = perf_counter()
            tensor = preprocess_rec(crop_text_region(image, row["box"]))
            if list(tensor.shape) != row["input_shape"]:
                raise ValueError("Original crop shape changed")
            prepared = perf_counter()
            if mode != "original-single":
                tensor = pad_white(tensor, item["padded_input_shape"][3])
            padded = perf_counter()
            raw, details = profiler.run_tensor(tensor)
            inferred = perf_counter()
            if raw.dtype != np.float32:
                raise ValueError("Expected FP32 engine output")
            text, score = decode_ctc(raw, characters)
            decoded = perf_counter()
            strict = text == row["text"]
            policy = (page_number_equal(row["text"], text)
                      if mode != "original-single" and roles.get(row["index"]) == "page_number" else strict)
            result.update(host=details, input_shape=list(tensor.shape), output_shape=list(raw.shape),
                          all_finite=True, strict_equal=strict, policy_equal=policy, empty_text=text == "",
                          crop_preprocess_ms=(prepared-started)*1000, padding_ms=(padded-prepared)*1000,
                          inference_ms=(inferred-padded)*1000, decode_ms=(decoded-inferred)*1000,
                          crop_rec_ms=(decoded-started)*1000)
            if not np.isfinite(score) or not policy:
                raise ValueError("Non-finite score or field regression; benchmark invalid")
            if mode == "padded-fixed" and phase == "measured" and (
                    details.get("context_width") != tensor.shape[3] or details["shape_changed"] or
                    details["malloc_count"] or details["free_count"]):
                raise ValueError("Measured fixed-context route/shape/allocation invariant failed")
            if not strict:
                changed.append(row["index"])
            if not text:
                empty.append(row["index"])
            for key in TIMINGS:
                totals[key] += result[key]
            result["status"] = "consistent"
            del raw, tensor
        except BaseException as error:
            if profiler.last_record is not None:
                result["host"] = profiler.last_record
            result.update(status="error", error=f"{type(error).__name__}: {error}")
            records.append(result)
            raise
        records.append(result)
    return {"cycle": cycle, **totals, "cycle_wall_ms": (perf_counter()-start_cycle)*1000,
            "strict_changed_indices": changed, "empty_indices": empty, "region_count": len(rows)}


def measure(profiler, characters, image, rows, plan, roles, mode, warmup, repeat, report, snapshot):
    if not rows or len(rows) != len(plan) or min(warmup, repeat) < 1 or mode not in MODES:
        raise ValueError("Invalid measurement schedule")
    report.update(warmup_records=[], records=[], warmup_cycles=[], cycles=[])
    for phase, count, rk, ck in (("warmup", warmup, "warmup_records", "warmup_cycles"),
                                ("measured", repeat, "records", "cycles")):
        for cycle in range(count):
            result = run_cycle(profiler, characters, image, rows, plan, roles, mode, phase, cycle, report[rk])
            report[ck].append(result)
            snapshot(f"{phase}_{cycle+1}")  # Outside wall timers; may still affect next cycle.
            print(f"{mode} {phase} {cycle+1}/{count}: crop-rec wall {result['cycle_wall_ms']:.3f} ms", flush=True)
    report["cycle_summary_ms"] = {k: stats(r[k] for r in report["cycles"])
                                  for k in (*TIMINGS, "cycle_wall_ms")}
    report["measured_host_totals"] = {k: sum(r["host"][k] for r in report["records"])
                                     for k in ("shape_changed", "malloc_count", "free_count")}
    report["measured_host_ms"] = {k: stats(r["host"][k] for r in report["records"])
                                  for k in ("prepare_ms", "enqueue_api_ms", "call_wall_ms")}
    report["validation"] = {"all_outputs_finite": True, "all_policy_equal": True,
                            "warmup_checked_count": len(report["warmup_records"]),
                            "measured_checked_count": len(report["records"]),
                            "strict_changed_indices": sorted({r["index"] for r in report["records"] if not r["strict_equal"]}),
                            "source_empty_indices": [r["index"] for r in rows if not r["text"]]}
    report["status"] = POLICY_SUCCESS if any(not r["strict_equal"] for r in
                                            report["records"] + report["warmup_records"]) else SUCCESS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("engine", "onnx", "image", "source-report", "baseline-report", "acceptance-report", "field-annotations"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--dict", dest="dictionary", required=True)
    parser.add_argument("--branch", choices=["reference", "candidate"], default="reference")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--buckets", type=int, nargs="+", default=[64, 128, 160, 256, 512, 960, 1024])
    parser.add_argument("--max-contexts", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--run-note", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        validate_buckets(args.buckets)
        if not 1 <= args.max_contexts <= 7 or min(args.warmup, args.repeat) < 1:
            raise ValueError("Require positive cycles and context cap 1..7")
    except ValueError as error:
        parser.error(str(error))
    if args.output.exists():
        parser.error("Report exists; use a new output filename")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    report = {"status": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
              "mode": args.mode, "buckets": args.buckets, "max_contexts": args.max_contexts,
              "warmup_requested": args.warmup, "repeat_requested": args.repeat,
              "python": platform.python_version(), "platform": platform.platform(), "opencv_threads": 1,
              "batch_size": 1, "run_note": args.run_note,
              "timing_scope": "One mode/process, original crop order. inference_ms includes routing, host instrumentation, packing, prepare/allocation, H2D, enqueue, D2H and synchronization; NOT device kernel time. crop_rec_ms includes crop/preprocess/padding/full CTC. cycle_wall_ms also includes validation/bookkeeping. Image decode, hashes, model/context load, JSON, prints and memory snapshots excluded. Warmup kept separately. Host allocation times are nested in prepare, not additive.",
              "limitations": "Single-image fixed crops, no live det/layout/NER/HTTP, no GT accuracy or automatic speedup claim. p95/p99 exploratory. Device-wide memory snapshots are NOT process peak and may perturb subsequent cycles. Engine hash matches baseline, not proof of build provenance. No recognized text stored."}
    rec = pool = None
    def snapshot(label):
        if args.memory:
            from benchmark_ocr import memory_snapshot
            report.setdefault("memory_snapshots", []).append(memory_snapshot(label))
    try:
        def read(path):
            return json.loads(Path(path).read_text(encoding="utf-8"))
        source, baseline, accepted, annotation = map(read, (args.source_report, args.baseline_report,
                                                          args.acceptance_report, args.field_annotations))
        provenance = dict(source_branch=args.branch, source_report_sha256=sha256_file(args.source_report),
                          image_sha256=sha256_file(args.image), reference_rec_onnx_sha256=sha256_file(args.onnx),
                          dictionary_sha256=sha256_file(args.dictionary))
        image = read_image(args.image)
        rows = prepare_rows(source, image, provenance["image_sha256"], provenance["reference_rec_onnx_sha256"],
                            provenance["dictionary_sha256"], args.branch)
        crop_set = [{k: r[k] for k in ("index", "box", "input_shape")} for r in rows]
        provenance["crop_set_sha256"] = hashlib.sha256(json.dumps(crop_set, sort_keys=True).encode()).hexdigest()
        report.update(provenance, engine_sha256=sha256_file(args.engine),
                      acceptance_report_sha256=sha256_file(args.acceptance_report),
                      baseline_report_sha256=sha256_file(args.baseline_report),
                      field_annotations_sha256=sha256_file(args.field_annotations), field_annotations=annotation)
        verify_baseline(baseline, provenance, report["engine_sha256"])
        roles = validate_annotations(annotation, provenance, rows)
        verify_acceptance(accepted, provenance, report["engine_sha256"], report["field_annotations_sha256"], args.buckets, rows)
        snapshot("before_load")
        tick = perf_counter()
        rec = TensorRTRecognizer(args.engine, args.dictionary)
        report["model_load_ms"] = (perf_counter()-tick)*1000
        report["metadata"] = rec.metadata()
        plan = make_plan(rows, args.buckets, rec.profile)
        report["plan"] = plan
        report["verified_inputs"] = verify_inputs(image, rows, plan, accepted)
        widths = sorted({r["padded_input_shape"][3] for r in plan})
        report["bucket_counts"] = dict(Counter(r["padded_input_shape"][3] for r in plan))
        profiler = HostProfiler(rec.runner)
        report["context_count"] = 1
        if args.mode == "padded-fixed":
            tick = perf_counter()
            pool = BucketContexts(rec.runner, widths, args.max_contexts)
            pool.initialize()
            report["extra_context_create_ms"] = (perf_counter()-tick)*1000
            profiler = pool
            report["context_count"] = len(widths)
        snapshot("contexts_created")
        measure(profiler, rec.characters, image, rows, plan, roles, args.mode,
                args.warmup, args.repeat, report, snapshot)
        if pool is not None:
            report["resources"] = pool.resources()
    except KeyboardInterrupt:
        report.update(status="interrupted", error="KeyboardInterrupt")
    except Exception as error:
        report.update(status="invalid_benchmark", error=f"{type(error).__name__}: {error}")
    finally:
        try:
            if pool is not None:
                pool.close()
            if rec is not None:
                rec.close()
            snapshot("after_cleanup")
        except Exception as error:
            report.update(status="invalid_benchmark", cleanup_error=f"{type(error).__name__}: {error}")
        if report["status"] not in (SUCCESS, POLICY_SUCCESS):
            for key in ("cycle_summary_ms", "validation", "measured_host_totals", "measured_host_ms"):
                report.pop(key, None)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Saved: {args.output.resolve()}")
    print(json.dumps({k: report[k] for k in ("status", "mode", "error", "cleanup_error", "context_count",
                     "validation", "measured_host_totals", "cycle_summary_ms") if k in report}, indent=2))
    if report["status"] not in (SUCCESS, POLICY_SUCCESS):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
