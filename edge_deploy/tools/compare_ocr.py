"""Compare two detectors with one shared ORT CPU recognizer. Not an accuracy or speed benchmark."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from ocr_det import create_detector, read_image, sha256_file
from ocr_rec import Recognizer, recognize_boxes


def rect_iou(left, right):
    a, b = np.asarray(left), np.asarray(right)
    lo_a, hi_a, lo_b, hi_b = a.min(0), a.max(0), b.min(0), b.max(0)
    intersection = float(np.prod(np.maximum(0, np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b))))
    union = float(np.prod(hi_a - lo_a) + np.prod(hi_b - lo_b)) - intersection
    return intersection / union if union > 0 else 0.0


def compare_records(reference, candidate, min_iou=0.5):
    errors = sum(r["status"] == "error" for r in reference + candidate)
    if errors:
        return {"status": "invalid_recognition_errors", "error_count": errors}
    if not reference or not candidate:
        return {"status": "insufficient_text_regions", "reference_count": len(reference),
                "candidate_count": len(candidate)}
    matrix = np.array([[rect_iou(a["box"], b["box"]) for b in candidate] for a in reference])
    pairs = []
    # Conservative one-to-one pairing: require unique mutual best IoU. Do not hide ambiguity.
    for i, row in enumerate(matrix):
        j = int(row.argmax())
        value = float(row[j])
        if (value < min_iou or int(matrix[:, j].argmax()) != i
                or np.count_nonzero(np.isclose(row, value, rtol=0, atol=1e-9)) != 1
                or np.count_nonzero(np.isclose(matrix[:, j], value, rtol=0, atol=1e-9)) != 1):
            continue
        pairs.append({"reference_index": i, "candidate_index": j, "rect_iou": value,
                      "text_equal": reference[i]["text"] == candidate[j]["text"]})
    unmatched_a = sorted(set(range(len(reference))) - {p["reference_index"] for p in pairs})
    unmatched_b = sorted(set(range(len(candidate))) - {p["candidate_index"] for p in pairs})
    texts_a, texts_b = [r["text"] for r in reference], [r["text"] for r in candidate]
    return {
        "status": "compared_not_accuracy_validated", "reference_count": len(reference),
        "candidate_count": len(candidate), "reference_empty_count": texts_a.count(""),
        "candidate_empty_count": texts_b.count(""), "matched_count": len(pairs),
        "matched_text_equal_count": sum(p["text_equal"] for p in pairs),
        "changed_pairs": [p for p in pairs if not p["text_equal"]], "pairs": pairs,
        "unmatched_reference_indices": unmatched_a, "unmatched_candidate_indices": unmatched_b,
        "all_regions_matched_text_equal": not unmatched_a and not unmatched_b and all(p["text_equal"] for p in pairs),
        "ordered_texts_equal": texts_a == texts_b,
        "text_multisets_equal": Counter(texts_a) == Counter(texts_b),
        "matching": f"Unique mutual-best axis-aligned crop IoU >= {min_iou}; not an accuracy threshold or split/merge matcher",
        "note": "Empty text is recorded. Text equality, including two empty strings, is not correctness. No ground truth, layout, field assignment or CER measured.",
    }


def run_branch(backend, model, image, recognizer, threads):
    detector = create_detector(backend, model, threads)
    try:
        metadata = detector.metadata()  # Hashing excluded from timings.
        start = perf_counter()
        boxes, det_timing, shape = detector.predict(image)
        records, rec_timing = recognize_boxes(
            recognizer, image, boxes,
            progress=lambda done, total: print(f"Recognized {done}/{total} regions", flush=True))
        wall_ms = (perf_counter() - start) * 1000
        return {"detector": metadata, "input_shape": shape, "records": records,
                "timings": {"det": det_timing, **rec_timing, "det_crop_rec_wall_ms": wall_ms}}
    finally:
        if hasattr(detector, "close"):
            detector.close()
        del detector
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-det", required=True)
    parser.add_argument("--candidate-det", required=True)
    parser.add_argument("--reference-backend", choices=["trt", "ort"], default="trt")
    parser.add_argument("--candidate-backend", choices=["trt", "ort"], default="trt")
    parser.add_argument("--rec-model", required=True)
    parser.add_argument("--dict", required=True, dest="dictionary")
    parser.add_argument("--image", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new report filename")
    if args.threads < 1:
        parser.error("threads must be positive")
    cv2.setNumThreads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "started", "note": "Single-pass diagnostic; shared rec runs reference first, candidate second. No warmup or speedup claim. JSON contains recognized text; review before sharing.",
              "timing_scope": "Branch wall time: det + crop + rec + record creation/progress logging; excludes model load, hashing, image decode, JSON, layout, NER, HTTP. Rec stage sums cover successful calls only; failed branches are invalid."}
    try:
        report["image_sha256"] = sha256_file(args.image)
        image = read_image(args.image)
        report["image_shape"] = list(image.shape)
        rec = Recognizer(args.rec_model, args.dictionary, args.threads)
        report["recognizer"] = rec.metadata()
        for name in ("reference", "candidate"):
            print(f"Running {name}: det + fixed ORT CPU rec ...", flush=True)
            report[name] = run_branch(getattr(args, name + "_backend"), getattr(args, name + "_det"), image, rec, args.threads)
        report["comparison"] = compare_records(report["reference"]["records"], report["candidate"]["records"])
        report["status"] = report["comparison"]["status"]
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"Saved: {args.output.resolve()}", flush=True)
    print(json.dumps({k: v for k, v in report["comparison"].items() if k != "pairs"}, ensure_ascii=False, indent=2))
    for name in ("reference", "candidate"):
        print(name, json.dumps(report[name]["timings"], indent=2))
    if report["status"] != "compared_not_accuracy_validated":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
