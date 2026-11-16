"""Controlled live TRT det -> crop -> rec, with bounded fixed rec contexts.

Smoke first; one mode per process. This is not the HTTP/layout/NER service.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from benchmark_ocr import memory_snapshot, verify_detector
from benchmark_rec import prepare_rows, stats
from benchmark_rec_buckets import verify_acceptance, verify_inputs, run_cycle, TIMINGS
from compare_rec_buckets import make_plan, validate_buckets, SUCCESS, POLICY_SUCCESS
from ocr_det import create_detector, read_image, sha256_file
from ocr_rec import TensorRTRecognizer
from ocr_text_policy import validate_annotations
from profile_rec_shapes import verify_baseline
from trt_bucket_contexts import BucketContexts
from trt_host_profile import HostProfiler

SMOKE_SUCCESS = "smoke_passed_not_benchmarked_not_accuracy_validated"
MODES = ("original-single", "padded-fixed")
METRICS = ("det_preprocess_ms", "det_inference_ms", "det_postprocess_ms", "det_pipeline_ms",
           *("rec_" + k for k in TIMINGS), "rec_cycle_wall_ms", "pipeline_wall_ms")


def run_pipeline(detector, profiler, characters, image, rows, plan, roles, det_shape,
                 mode, phase, cycle, sample):
    sample.update(phase=phase, cycle=cycle, records=[])
    try:
        start = perf_counter()
        boxes, timing, shape = detector.predict(image)  # Always live detection.
        if any(np.asarray(box).shape != (4, 2) or not np.isfinite(box).all() for box in boxes):
            raise ValueError("Invalid/non-finite live detection box")
        actual = [np.asarray(box).tolist() for box in boxes]
        sample.update(boxes=actual, det_input_shape=list(shape))
        if list(shape) != det_shape or len(actual) != len(rows):
            raise ValueError("Live detection input shape/count differs from controlled source")
        changed = [r["index"] for r, box in zip(rows, actual) if not np.array_equal(box, r["box"])]
        sample["changed_box_indices"] = changed
        if changed:
            raise ValueError("Live box geometry/order changed; field roles cannot be reused")
        if len(plan) != len(rows) or any(r["index"] != p["index"] for r, p in zip(rows, plan)):
            raise ValueError("Incomplete or reordered rec plan")
        # Only after exact geometry/order checks can the source roles transfer.
        live_rows = [{**row, "box": box} for row, box in zip(rows, actual)]
        rec = run_cycle(profiler, characters, image, live_rows, plan, roles,
                        mode, phase, cycle, sample["records"])
        end = perf_counter()
        sample["timings"] = {**{"det_" + k: float(v) for k, v in timing.items()},
                             **{"rec_" + k: rec[k] for k in (*TIMINGS, "cycle_wall_ms")},
                             "pipeline_wall_ms": (end-start)*1000}
        if any(not np.isfinite(sample["timings"][k]) or sample["timings"][k] < 0 for k in METRICS):
            # Avoid serializing NaN into an otherwise useful failure report.
            sample.pop("timings")
            raise ValueError("Invalid pipeline timing")
        sample.update(status="consistent", region_count=len(rows),
                      strict_changed_indices=rec["strict_changed_indices"], empty_indices=rec["empty_indices"])
    except BaseException as error:
        sample.update(status="error", error=f"{type(error).__name__}: {error}")
        raise


def measure_pipeline(detector, profiler, characters, image, rows, plan, roles, shape,
                     mode, stage, warmup, repeat, report, snapshot):
    if not rows or mode not in MODES or stage not in ("smoke", "benchmark") or min(warmup, repeat) < 1:
        raise ValueError("Invalid pipeline schedule")
    report.update(validation_cycles=[], warmup_cycles=[], cycles=[])
    for phase, count, key in (("validation", 1, "validation_cycles"),
                              ("warmup", warmup, "warmup_cycles"), ("measured", repeat, "cycles")):
        for cycle in range(count):
            sample = {}
            report[key].append(sample)
            run_pipeline(detector, profiler, characters, image, rows, plan, roles, shape,
                         mode, phase, cycle, sample)
            print(f"{stage} {mode} {phase} {cycle+1}/{count}: "
                  f"{len(rows)} regions, pipeline {sample['timings']['pipeline_wall_ms']:.3f} ms", flush=True)
        snapshot("after_" + phase)  # Never query memory between measured cycles.
    measured = [r for c in report["cycles"] for r in c["records"]]
    all_records = [r for key in ("validation_cycles", "warmup_cycles", "cycles")
                   for c in report[key] for r in c["records"]]
    report["validation"] = {"live_det_calls": 1+warmup+repeat,
                            "validation_rec_calls": len(rows), "warmup_rec_calls": warmup*len(rows),
                            "measured_rec_calls": repeat*len(rows), "all_boxes_equal_to_source": True,
                            "all_outputs_finite": True, "all_policy_equal": True,
                            "strict_changed_indices": sorted({r["index"] for r in all_records if not r["strict_equal"]}),
                            "source_empty_indices": [r["index"] for r in rows if not r["text"]],
                            "note": "Smoke measured phase is a post-warmup invariant check, not a benchmark. Shared empty text is not correctness."}
    report["measured_rec_host_totals"] = {k: sum(r["host"][k] for r in measured)
                                         for k in ("shape_changed", "malloc_count", "free_count")}
    if stage == "smoke":
        report["status"] = SMOKE_SUCCESS
    else:
        report["summary_ms"] = {k: stats(c["timings"][k] for c in report["cycles"]) for k in METRICS}
        report["status"] = POLICY_SUCCESS if any(not r["strict_equal"] for r in all_records) else SUCCESS


def close_pipeline(pool, rec, detector):
    errors = []
    try:
        if pool is not None:
            pool.close()
        if rec is not None:
            rec.close()
    except Exception as error:
        errors.append({"component": "rec_pool_or_owner", "error": f"{type(error).__name__}: {error}"})
    # Detector has a separate engine and stream; still attempt its cleanup.
    try:
        if detector is not None:
            detector.close()
    except Exception as error:
        errors.append({"component": "detector", "error": f"{type(error).__name__}: {error}"})
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("det-engine", "rec-engine", "rec-onnx", "dict", "image", "source-report",
                 "baseline-report", "acceptance-report", "field-annotations"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--branch", choices=["reference", "candidate"], default="reference")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--stage", choices=["smoke", "benchmark"], default="smoke")
    parser.add_argument("--buckets", type=int, nargs="+", default=[64, 128, 160, 256, 512, 960, 1024])
    parser.add_argument("--max-contexts", type=int, default=7)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--run-note", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        validate_buckets(args.buckets)
        if not 1 <= args.max_contexts <= 7:
            raise ValueError("Context cap must be 1..7")
        if args.stage == "smoke" and (args.warmup is not None or args.repeat is not None):
            raise ValueError("Smoke uses one validation, one warmup, one check; omit warmup/repeat")
        warmup = args.warmup if args.warmup is not None else (1 if args.stage == "smoke" else 2)
        repeat = args.repeat if args.repeat is not None else (1 if args.stage == "smoke" else 10)
        if min(warmup, repeat) < 1:
            raise ValueError("Positive warmup/repeat required")
    except ValueError as error:
        parser.error(str(error))
    if args.output.exists():
        parser.error("Report exists; use a new output filename")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    report = {"status": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
              "stage": args.stage, "mode": args.mode, "batch_size": 1, "opencv_threads": 1,
              "warmup_requested": warmup, "repeat_requested": repeat, "buckets": args.buckets,
              "max_contexts": args.max_contexts, "run_note": args.run_note,
              "topology": "Two engine instances, one det context/stream and one rec stream shared by bounded rec contexts; all calls sequential and synchronized, not concurrent kernels.",
              "timing_scope": "Live det preprocess/inference/postprocess + exact box checks + live crop/preprocess/padding + profiled rec call + full CTC/field checks + bookkeeping. Image decode, hashes, model/context load, JSON, prints, memory snapshots excluded. First validation and warmup separated. Not HTTP/layout/NER or cold-start latency; rec inference includes routing, packing, allocation if needed, H2D/enqueue/D2H/sync, not kernel-only.",
              "memory_scope": "Device-wide snapshots outside timers; no samples between measured cycles. Not per-process peak; N/A or failed queries stay unknown. No guaranteed VRAM budget.",
              "field_mapping": "Manual source roles only transferred after ALL live boxes match source geometry and order exactly; controlled regression gate, not general layout classification or accuracy threshold.",
              "limitations": "Single image, no GT accuracy or stable p95/p99; no recognized text stored. Source hashes bind artifacts, not cryptographic proof of engine build precision."}
    detector = rec = pool = None
    def snapshot(label):
        if args.memory:
            report.setdefault("memory_snapshots", []).append(memory_snapshot(label))
    try:
        def read(path):
            return json.loads(Path(path).read_text(encoding="utf-8"))
        source, baseline, accepted, annotation = map(read, (args.source_report, args.baseline_report,
                                                          args.acceptance_report, args.field_annotations))
        provenance = dict(source_branch=args.branch, source_report_sha256=sha256_file(args.source_report),
                          image_sha256=sha256_file(args.image), reference_rec_onnx_sha256=sha256_file(args.rec_onnx),
                          dictionary_sha256=sha256_file(args.dict))
        image = read_image(args.image)
        rows = prepare_rows(source, image, provenance["image_sha256"], provenance["reference_rec_onnx_sha256"],
                            provenance["dictionary_sha256"], args.branch)
        crop_set = [{k: r[k] for k in ("index", "box", "input_shape")} for r in rows]
        provenance["crop_set_sha256"] = hashlib.sha256(json.dumps(crop_set, sort_keys=True).encode()).hexdigest()
        report.update(provenance, rec_engine_sha256=sha256_file(args.rec_engine),
                      det_engine_sha256=sha256_file(args.det_engine),
                      baseline_report_sha256=sha256_file(args.baseline_report),
                      acceptance_report_sha256=sha256_file(args.acceptance_report),
                      field_annotations_sha256=sha256_file(args.field_annotations), field_annotations=annotation)
        verify_baseline(baseline, provenance, report["rec_engine_sha256"])
        det_shape = verify_detector(source[args.branch], "trt", report["det_engine_sha256"])
        roles = validate_annotations(annotation, provenance, rows)
        verify_acceptance(accepted, provenance, report["rec_engine_sha256"], report["field_annotations_sha256"], args.buckets, rows)
        snapshot("before_load")
        tick = perf_counter()
        detector = create_detector("trt", args.det_engine)
        report["det_load_ms"] = (perf_counter()-tick)*1000
        report["detector"] = detector.metadata()
        snapshot("after_detector_load")
        tick = perf_counter()
        rec = TensorRTRecognizer(args.rec_engine, args.dict)
        report["rec_load_ms"] = (perf_counter()-tick)*1000
        report["recognizer"] = rec.metadata()
        plan = make_plan(rows, args.buckets, rec.profile)
        report["plan"] = plan
        report["verified_inputs"] = verify_inputs(image, rows, plan, accepted)
        profiler = HostProfiler(rec.runner)
        report["rec_context_count"] = 1
        if args.mode == "padded-fixed":
            widths = sorted({p["padded_input_shape"][3] for p in plan})
            tick = perf_counter()
            pool = BucketContexts(rec.runner, widths, args.max_contexts)
            pool.initialize()
            report["extra_rec_context_create_ms"] = (perf_counter()-tick)*1000
            report["rec_context_count"] = len(widths)
            profiler = pool
        snapshot("after_all_contexts_load")
        measure_pipeline(detector, profiler, rec.characters, image, rows, plan, roles, det_shape,
                         args.mode, args.stage, warmup, repeat, report, snapshot)
        if pool is not None:
            report["rec_resources"] = pool.resources()
    except KeyboardInterrupt:
        report.update(status="interrupted", error="KeyboardInterrupt")
    except Exception as error:
        report.update(status="invalid_pipeline", error=f"{type(error).__name__}: {error}")
    finally:
        errors = close_pipeline(pool, rec, detector)
        if errors:
            report.update(status="invalid_pipeline", cleanup_errors=errors)
        snapshot("after_cleanup")
        if report["status"] not in (SMOKE_SUCCESS, SUCCESS, POLICY_SUCCESS):
            for key in ("summary_ms", "validation", "measured_rec_host_totals"):
                report.pop(key, None)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Saved: {args.output.resolve()}")
    print(json.dumps({k: report[k] for k in ("status", "stage", "mode", "error", "cleanup_errors",
                     "rec_context_count", "validation", "measured_rec_host_totals", "summary_ms", "memory_snapshots")
                     if k in report}, indent=2))
    if report["status"] not in (SMOKE_SUCCESS, SUCCESS, POLICY_SUCCESS):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
