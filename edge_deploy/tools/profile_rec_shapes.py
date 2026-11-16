"""Two-crop, host-side shape-switch diagnostic; not a production benchmark."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from benchmark_rec import prepare_rows, stats, validate_profile
from ocr_det import read_image, sha256_file
from ocr_rec import TensorRTRecognizer, crop_text_region, preprocess_rec, decode_ctc
from trt_host_profile import HostProfiler, METRICS
from trt_fixed_contexts import FixedContextProfiler, close_context_pair


SUCCESS = "completed_consistent_not_accuracy_validated"


def schedules(order):
    cases = [("repeat_a", ("a", "a")), ("repeat_b", ("b", "b")),
             ("alternate_ab", ("a", "b")), ("alternate_ba", ("b", "a"))]
    if order == "alternating-first":
        return cases[2:] + cases[:2]
    if order != "same-first":
        raise ValueError("Unknown case order")
    return cases


def verify_baseline(baseline, provenance, engine_sha):
    if baseline.get("status") != SUCCESS or baseline.get("rec_backend") != "trt":
        raise ValueError("Expected a successful TRT-rec pipeline baseline")
    for key, value in provenance.items():
        if baseline.get(key) != value:
            raise ValueError(f"Pipeline baseline provenance mismatch: {key}")
    if baseline.get("recognizer", {}).get("engine_sha256") != engine_sha:
        raise ValueError("Engine differs from pipeline baseline")


def select_samples(rows, image, index_a, index_b):
    if index_a == index_b:
        raise ValueError("A/B must select distinct indices")
    by_index = {row["index"]: row for row in rows}
    samples = {}
    for label, index in (("a", index_a), ("b", index_b)):
        if index not in by_index:
            raise ValueError(f"Missing source index {index}")
        row = by_index[index]
        tensor = preprocess_rec(crop_text_region(image, row["box"]))
        samples[label] = {"row": row, "tensor": tensor}
    if samples["a"]["tensor"].shape == samples["b"]["tensor"].shape:
        raise ValueError("A/B must have different original input widths")
    return samples


def measure(profiler, characters, samples, warmup, repeat, order, report):
    if min(warmup, repeat) < 1:
        raise ValueError("warmup/repeat must be positive")
    report["cases"] = []
    for name, sequence in schedules(order):
        case = {"name": name, "sequence": list(sequence), "warmup_records": [], "records": []}
        report["cases"].append(case)
        for phase, count, key in (("warmup", warmup, "warmup_records"),
                                  ("measured", repeat, "records")):
            for cycle in range(count):
                for position, label in enumerate(sequence):
                    sample = samples[label]
                    result = {"phase": phase, "cycle": cycle, "position": position,
                              "sample": label, "index": sample["row"]["index"]}
                    try:
                        raw, details = profiler.run_tensor(sample["tensor"])
                        result.update(details)
                        # Finite/rank/class checks and CTC are OUTSIDE the host call timer.
                        text, score = decode_ctc(raw, characters)
                        result.update(all_finite=True, text_equal=text == sample["row"]["text"],
                                      empty_text=text == "")
                        del raw
                        if not np.isfinite(score) or not result["text_equal"]:
                            raise ValueError("CTC result differs from source or score is non-finite")
                        result["status"] = "consistent"
                    except BaseException as error:
                        if profiler.last_record is not None:
                            result.update(profiler.last_record)
                        result.update(status="error", error=f"{type(error).__name__}: {error}")
                        case[key].append(result)
                        raise
                    case[key].append(result)
        print(f"Completed {name}: {len(case['records'])} measured calls", flush=True)
    # Publish summaries only after ALL cases pass.
    summary = {}
    for case in report["cases"]:
        summary[case["name"]] = {}
        for label in dict.fromkeys(case["sequence"]):
            rows = [r for r in case["records"] if r["sample"] == label]
            summary[case["name"]][label] = {
                "count": len(rows), "shape_changes": sum(r["shape_changed"] for r in rows),
                "malloc_count": sum(r["malloc_count"] for r in rows),
                "free_count": sum(r["free_count"] for r in rows),
                "host_ms": {metric: stats(r[metric] for r in rows)
                            for metric in (*METRICS, "malloc_host_ms", "free_host_ms")}}
    report.update(summary=summary, status=SUCCESS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("engine", "onnx", "image", "source-report", "baseline-report"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--dict", dest="dictionary", required=True)
    parser.add_argument("--branch", choices=["reference", "candidate"], default="reference")
    parser.add_argument("--index-a", type=int, default=50)
    parser.add_argument("--index-b", type=int, default=49)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--order", choices=["same-first", "alternating-first"], default="same-first")
    parser.add_argument("--context-mode", choices=["single", "fixed"], default="single",
                        help="single: original changing-shape context; fixed: two contexts sharing engine/stream")
    parser.add_argument("--memory", action="store_true", help="Device-wide nvidia-smi snapshots outside measurements")
    parser.add_argument("--run-note", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.warmup, args.repeat) < 1:
        parser.error("warmup/repeat must be positive two-call cycles")
    if args.output.exists():
        parser.error("Report exists; use a new output filename")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    report = {
        "status": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
        "warmup_requested": args.warmup, "repeat_requested": args.repeat,
        "order": args.order, "run_note": args.run_note,
        "context_mode": args.context_mode,
        "timing_scope": {
            "prepare_ms": "Existing _prepare: shape setup, exact-size buffer allocation/free, binding, host output allocation.",
            "malloc_host_ms/free_host_ms": "Nested SUBSETS of prepare_ms, not additive stages; include failed API time if any.",
            "pack_ms": "Existing np.ascontiguousarray; preprocessed tensor reused with original layout.",
            "h2d_api_ms/enqueue_api_ms/d2h_api_ms": "Host API return time, NOT pure GPU compute or transfer time. Pageable memory/driver/resource waits may block.",
            "sync_ms": "One final stream synchronization, as in baseline; remaining device work only.",
            "call_wall_ms": "Completed run_tensor sequence plus instrumentation bookkeeping; excludes CTC and source text comparison.",
            "excluded": "Model loading, hashes, crop/preprocess, CTC, detection, layout, NER, HTTP, file IO. No new synchronization/events or allocation policy changes.",
        },
        "limitations": "Two fixed crops, one rec engine, batch=1, original precision and widths. NOT full-page speedup, accuracy, or kernel profiling. Warmups separate; percentiles exploratory. Compare identical sample across cases; case order/device power/other workloads can confound. Engine hash matches baseline, not independent proof of ONNX build provenance.",
    }
    rec = fixed = None
    def snapshot(label):
        if args.memory:
            from benchmark_ocr import memory_snapshot
            report.setdefault("memory_snapshots", []).append(memory_snapshot(label))

    try:
        source = json.loads(Path(args.source_report).read_text(encoding="utf-8"))
        baseline = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
        provenance = dict(source_branch=args.branch,
                          source_report_sha256=sha256_file(args.source_report),
                          image_sha256=sha256_file(args.image),
                          reference_rec_onnx_sha256=sha256_file(args.onnx),
                          dictionary_sha256=sha256_file(args.dictionary))
        image = read_image(args.image)
        rows = prepare_rows(source, image, provenance["image_sha256"],
                            provenance["reference_rec_onnx_sha256"],
                            provenance["dictionary_sha256"], args.branch)
        crop_set = [{k: r[k] for k in ("index", "box", "input_shape")} for r in rows]
        provenance["crop_set_sha256"] = hashlib.sha256(
            json.dumps(crop_set, sort_keys=True).encode()).hexdigest()
        report.update(provenance, baseline_report_sha256=sha256_file(args.baseline_report),
                      engine_sha256=sha256_file(args.engine))
        verify_baseline(baseline, provenance, report["engine_sha256"])
        samples = select_samples(rows, image, args.index_a, args.index_b)
        report["samples"] = {
            label: {"index": sample["row"]["index"], "input_shape": list(sample["tensor"].shape),
                    "tensor_sha256": hashlib.sha256(sample["tensor"].tobytes(order="C")).hexdigest(),
                    "original_c_contiguous": bool(sample["tensor"].flags.c_contiguous)}
            for label, sample in samples.items()}
        snapshot("before_model_load")
        started = perf_counter()
        rec = TensorRTRecognizer(args.engine, args.dictionary)
        report["base_model_load_ms"] = (perf_counter() - started) * 1000
        report["metadata"] = rec.metadata()
        validate_profile([s["row"] for s in samples.values()], rec.profile)
        snapshot("one_context_loaded")
        profiler = HostProfiler(rec.runner)
        if args.context_mode == "fixed":
            fixed = FixedContextProfiler(rec.runner, {k: v["tensor"].shape for k, v in samples.items()})
            profiler = fixed
            report["fixed_context_resources_before_measurement"] = fixed.resources()
            report["timing_scope"]["dispatch_wall_ms"] = "Fixed routing + completed profiled call; recorded per call separately from call_wall_ms. CTC excluded."
            report["context_note"] = "One engine, two independent contexts and I/O buffers, one borrowed stream; sequential calls fully synchronized. Owner is A; secondary is B. First setup remains in warmup; no new engine, padding, precision or batching changes."
            snapshot("two_contexts_loaded")
        measure(profiler, rec.characters, samples,
                args.warmup, args.repeat, args.order, report)
        if fixed is not None:
            fixed.validate_measured(report)
            report["fixed_context_resources_after_measurement"] = fixed.resources()
            report["dispatch_summary_ms"] = {
                case["name"]: {label: stats(r["dispatch_wall_ms"] for r in case["records"] if r["sample"] == label)
                               for label in dict.fromkeys(case["sequence"])} for case in report["cases"]}
        snapshot("after_measurement")
    except KeyboardInterrupt:
        report.update(status="interrupted", error="KeyboardInterrupt")
    except Exception as error:
        report.update(status="invalid_diagnostic", error=f"{type(error).__name__}: {error}")
    finally:
        errors = close_context_pair(fixed, rec)
        if errors:
            report.update(status="invalid_diagnostic", cleanup_error="; ".join(errors))
        snapshot("after_close")
        if report["status"] != SUCCESS:
            report.pop("summary", None)
            report.pop("dispatch_summary_ms", None)
            report.pop("fixed_context_validation", None)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"Saved: {args.output.resolve()}", flush=True)
    print("Status:", report["status"])
    for key in ("error", "cleanup_error"):
        if key in report:
            print(key + ":", report[key])
    for case, groups in report.get("summary", {}).items():
        for label, group in groups.items():
            means = {k: round(v["mean"], 3) for k, v in group["host_ms"].items()}
            print(case, label, "calls/shape_changes/malloc/free:",
                  group["count"], group["shape_changes"], group["malloc_count"], group["free_count"],
                  "host means (ms):", means)
    print("Host diagnostic only; allocation timings are nested in prepare_ms, not extra stages.")
    if "fixed_context_validation" in report:
        print("Fixed context validation:", report["fixed_context_validation"])
        print("Fixed context resources:", json.dumps(report["fixed_context_resources_after_measurement"]))
        print("Dispatch-inclusive means (ms):", {case: {label: round(s["mean"], 3) for label, s in groups.items()}
                                                 for case, groups in report["dispatch_summary_ms"].items()})
    if "memory_snapshots" in report:
        print("Device-wide memory snapshots (not process peak):", json.dumps(report["memory_snapshots"]))
    if report["status"] != SUCCESS:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
