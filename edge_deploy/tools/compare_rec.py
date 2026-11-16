"""Fixed recorded crops: compare ORT CPU vs TensorRT rec on identical tensors.

No detection is rerun. No padding/batching/FP16 changes are introduced here.
Single-pass interleaved diagnostic timings are not a warmed-up benchmark.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from compare_trt import tensor_summary
from ocr_det import read_image, sha256_file
from ocr_rec import Recognizer, TensorRTRecognizer, crop_text_region, preprocess_rec, decode_ctc


def compare_rec_outputs(reference, candidate, characters):
    result = {"reference_output": tensor_summary(reference), "candidate_output": tensor_summary(candidate)}
    if reference.shape != candidate.shape:
        return {"status": "invalid_output_shape", **result}
    if not result["reference_output"]["all_finite"] or not result["candidate_output"]["all_finite"]:
        return {"status": "invalid_nonfinite_output", **result}
    ref_text, ref_score = decode_ctc(reference, characters)
    cand_text, cand_score = decode_ctc(candidate, characters)
    error = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    result.update({
        "status": "finite_outputs_compared", "mae": float(error.mean()),
        "max_abs_error": float(error.max()), "elements": int(error.size),
        "allclose_rtol_1e_3_atol_1e_4": bool(np.allclose(reference, candidate, rtol=1e-3, atol=1e-4)),
        "timestep_argmax_disagreement_fraction": float(np.mean(reference.argmax(-1) != candidate.argmax(-1))),
        "reference_text": ref_text, "candidate_text": cand_text,
        "reference_score": ref_score, "candidate_score": cand_score,
        "text_equal": ref_text == cand_text, "score_abs_error": abs(ref_score - cand_score),
    })
    return result


def verify_source(source, image_sha, rec_meta, branch):
    if source.get("status") != "compared_not_accuracy_validated":
        raise ValueError("Expected a completed compare_ocr report without recognition errors")
    if source.get("image_sha256") != image_sha:
        raise ValueError("Image SHA256 differs from recorded detection image")
    old_meta = source.get("recognizer", {})
    for key in ("model_sha256", "dictionary_sha256"):
        if old_meta.get(key) != rec_meta[key]:
            raise ValueError(f"Source recognition {key} mismatch; do not mix baseline versions")
    rows = source[branch]["records"]
    if not rows or any(r.get("status") not in ("ok", "empty_text") for r in rows):
        raise ValueError("Recorded crop set is empty or contains failed records")
    if len({r["index"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source region indices")
    return rows


def summarize(records):
    valid = [r for r in records if r["status"] == "finite_outputs_compared"]
    failed = [r["index"] for r in records if r["status"] != "finite_outputs_compared"]
    total = sum(r["elements"] for r in valid)
    return {
        "status": "compared_not_accuracy_validated" if records and not failed else "invalid_comparison",
        "region_count": len(records), "finite_compared_count": len(valid), "invalid_indices": failed,
        "execution_errors": [{"index": r["index"], "error": r["error"]} for r in records if "error" in r],
        "text_equal_count": sum(r["text_equal"] for r in valid),
        "changed_text_indices": [r["index"] for r in valid if not r["text_equal"]],
        "reference_empty_indices": [r["index"] for r in valid if not r["reference_text"]],
        "candidate_empty_indices": [r["index"] for r in valid if not r["candidate_text"]],
        "allclose_failed_indices": [r["index"] for r in valid if not r["allclose_rtol_1e_3_atol_1e_4"]],
        "element_weighted_mae": sum(r["mae"] * r["elements"] for r in valid) / total if total else None,
        "max_abs_error": max((r["max_abs_error"] for r in valid), default=None),
        "note": "Metrics over finite pairs only. Invalid rows invalidate experiment. Equality is not ground-truth accuracy, including shared empty text.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True, help="Candidate engine; ONNX only for --candidate-backend ort self-test")
    parser.add_argument("--candidate-backend", choices=["trt", "ort"], default="trt")
    parser.add_argument("--dict", dest="dictionary", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--det-report", required=True, help="Existing compare_ocr.py JSON")
    parser.add_argument("--branch", choices=["reference", "candidate"], default="reference")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Report exists; use a new output filename")
    if args.threads < 1:
        parser.error("threads must be positive")
    cv2.setNumThreads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "started", "source_branch": args.branch, "records": [],
              "note": "Same preprocessed crop tensor supplied to both backends. CPU then candidate per crop, no warmup, no speedup claim. JSON includes recognized text.",
              "timing_scope": "Per-call timers exclude crop/preprocess, statistics and CTC. TRT includes H2D, enqueue, D2H, sync and width-change buffer allocation. No det/HTTP/layout/NER or end-to-end benchmark."}
    gpu = None
    try:
        source = json.loads(Path(args.det_report).read_text(encoding="utf-8"))
        report["source_report_sha256"] = sha256_file(args.det_report)
        report["image_sha256"] = sha256_file(args.image)
        image = read_image(args.image)
        cpu = Recognizer(args.onnx, args.dictionary, args.threads)
        report["reference_metadata"] = cpu.metadata()
        rows = verify_source(source, report["image_sha256"], report["reference_metadata"], args.branch)
        report["source_detector"] = source[args.branch].get("detector")
        gpu = (TensorRTRecognizer(args.engine, args.dictionary) if args.candidate_backend == "trt"
               else Recognizer(args.engine, args.dictionary, args.threads))
        report["candidate_metadata"] = gpu.metadata()
        for position, row in enumerate(rows):
            result = {"index": row["index"], "box": row["box"]}
            try:
                crop = crop_text_region(image, row["box"])
                if crop is None:
                    raise ValueError("Empty source crop")
                tensor = np.ascontiguousarray(preprocess_rec(crop))
                result["input_shape"] = list(tensor.shape)
                if result["input_shape"] != row.get("input_shape"):
                    raise ValueError("Preprocessed shape differs from source report")
                start = perf_counter()
                reference = cpu.run_tensor(tensor)
                middle = perf_counter()
                candidate = gpu.run_tensor(tensor)
                end = perf_counter()
                result.update(reference_call_ms=(middle-start)*1000, candidate_call_ms=(end-middle)*1000)
                result.update(compare_rec_outputs(reference, candidate, cpu.characters))
                del reference, candidate
            except Exception as error:
                result.update(status="error", error=f"{type(error).__name__}: {error}")
            report["records"].append(result)
            if (position+1) % 10 == 0 or position+1 == len(rows):
                print(f"Compared {position+1}/{len(rows)} crops", flush=True)
        report["summary"] = summarize(report["records"])
        report["status"] = report["summary"]["status"]
        report["single_pass_call_totals_ms"] = {
            name: sum(r.get(name+"_call_ms", 0) for r in report["records"])
            for name in ("reference", "candidate")}
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if gpu is not None and hasattr(gpu, "close"):
                gpu.close()
        finally:
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            print(f"Saved: {args.output.resolve()}", flush=True)
    print(json.dumps(report["summary"], indent=2))
    print("Single-pass call totals (not benchmark):", report["single_pass_call_totals_ms"])
    if report["status"] != "compared_not_accuracy_validated":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
