"""Sequential batch=1 ORT/TRT baseline with explicit timing scope."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from ocr_det import create_detector, read_image, sha256_file


def stats(values):
    return {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["ort", "trt"], default="ort")
    parser.add_argument("--images", required=True, help="One image or directory (non-recursive)")
    parser.add_argument("--output", default="reports/ort_cpu.json")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5, help="Warmup cycles across all images")
    parser.add_argument("--repeat", type=int, default=20, help="Measured cycles across all images")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("warmup must be >= 0 and repeat must be >= 1")
    path = Path(args.images)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in extensions) if path.is_dir() else [path]
    if not paths or any(not p.is_file() for p in paths):
        parser.error("No valid image files found")
    cv2.setNumThreads(1)
    detector = create_detector(args.backend, args.model, args.threads)
    dataset = [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in paths]
    first_start = perf_counter()
    first_image = read_image(paths[0])
    _, first_timings, _ = detector.predict(first_image)
    first_request_ms = (perf_counter() - first_start) * 1000
    del first_image
    for cycle in range(args.warmup):
        for p in paths:
            detector.predict(read_image(p))
        print(f"Warmup {cycle + 1}/{args.warmup}", flush=True)
    samples = []
    for cycle in range(args.repeat):
        for p in paths:
            start = perf_counter()
            image = read_image(p)
            decoded = perf_counter()
            boxes, timings, input_shape = detector.predict(image)
            end = perf_counter()
            samples.append({"cycle": cycle, "image": str(p), "input_shape": input_shape,
                            "box_count": len(boxes), **timings,
                            "decode_ms": (decoded - start) * 1000,
                            "request_ms": (end - start) * 1000})
        print(f"Measured cycle {cycle + 1}/{args.repeat}", flush=True)
    metrics = ("decode_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "pipeline_ms", "request_ms")
    summary = {key: stats([s[key] for s in samples]) for key in metrics}
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": detector.metadata(), "dataset": dataset,
        "warmup_cycles": args.warmup, "repeat_cycles": args.repeat,
        "sample_count": len(samples), "first_inference": first_timings,
        "first_request_ms": first_request_ms, "summary_ms": summary,
        "serial_service_images_per_s": 1000 / summary["request_ms"]["mean"],
        "timing_scope": {
            "pipeline_ms": "preprocess + completed backend call + postprocess",
            "request_ms": "file read/decode + pipeline; excludes model load, JSON, visualization, HTTP, queueing",
            "inference_ms": ("ORT call including wrapper/input handling; not isolated kernel time"
                             if args.backend == "ort" else
                             "host contiguous conversion + H2D + TRT enqueue + D2H + stream sync; not GPU-only time; first call also allocates buffers"),
            "first_request_ms": "first request after session load; not a filesystem cold-cache test",
            "serial_service_images_per_s": "reciprocal mean service time; not concurrent or whole-server throughput",
        },
        "per_image_summary_ms": {
            str(p): {k: stats([s[k] for s in samples if s["image"] == str(p)]) for k in metrics}
            for p in paths
        },
        "samples": samples,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {destination.resolve()}")
    if len(samples) < 100:
        print("Small sample count: p95/p99 are exploratory, not stable tail-latency estimates.")


if __name__ == "__main__":
    main()
