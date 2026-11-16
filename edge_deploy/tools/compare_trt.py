"""Compare the same real-image tensor on ORT CPU and a TensorRT engine."""
import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np

from ocr_det import Detector, postprocess, preprocess, read_image, sha256_file


def tensor_summary(array):
    finite = np.isfinite(array)
    values = array[finite]
    return {
        "shape": list(array.shape), "dtype": str(array.dtype), "elements": int(array.size),
        "nan_count": int(np.isnan(array).sum()),
        "positive_inf_count": int(np.isposinf(array).sum()),
        "negative_inf_count": int(np.isneginf(array).sum()),
        "all_finite": bool(finite.all()),
        "finite_min": float(values.min()) if values.size else None,
        "finite_max": float(values.max()) if values.size else None,
        "finite_abs_gt_fp16_max": int(np.count_nonzero(np.abs(values) > 65504)),
    }


def compare_outputs(reference, candidate, image_shape):
    if reference.shape != candidate.shape:
        raise ValueError(f"Output shapes differ: {reference.shape} vs {candidate.shape}")
    summaries = {"ort_output": tensor_summary(reference), "trt_output": tensor_summary(candidate)}
    if not all(s["all_finite"] for s in summaries.values()):
        return {"status": "invalid_nonfinite_output", **summaries,
                "note": "Comparison/postprocessing skipped. Do not benchmark this engine as a valid result. No NaN/Inf replacement was performed."}
    error = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    ref_boxes = [b.tolist() for b in postprocess(reference, image_shape)]
    trt_boxes = [b.tolist() for b in postprocess(candidate, image_shape)]
    return {
        "status": "finite_outputs_compared", **summaries,
        "output_shape": list(reference.shape), "probability_mae": float(error.mean()),
        "probability_max_abs_error": float(error.max()),
        "probability_p99_abs_error": float(np.percentile(error, 99)),
        "allclose_rtol_1e_3_atol_1e_4": bool(np.allclose(reference, candidate, rtol=1e-3, atol=1e-4)),
        "mask_disagreement_fraction_at_0_1": float(np.mean((reference > 0.1) != (candidate > 0.1))),
        "ort_box_count": len(ref_boxes), "trt_box_count": len(trt_boxes),
        "ordered_boxes_exactly_equal": ref_boxes == trt_boxes,
        "ort_boxes": ref_boxes, "trt_boxes": trt_boxes,
        "note": "Numeric tolerance is diagnostic, not an accuracy acceptance gate. Box count alone is insufficient. No ground-truth accuracy or speedup is measured here.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="reports/compare_fp32.json")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    image = read_image(args.image)
    tensor = preprocess(image)
    ort_detector = Detector(args.onnx)
    ort_meta = ort_detector.metadata()
    reference = ort_detector.run_tensor(tensor)
    del ort_detector
    gc.collect()
    from trt_backend import TensorRTDetector
    trt_detector = TensorRTDetector(args.engine)
    try:
        candidate = trt_detector.run_tensor(tensor)
        result = compare_outputs(reference, candidate, image.shape)
        result.update({"ort_metadata": ort_meta, "trt_metadata": trt_detector.metadata(),
                       "image_sha256": sha256_file(args.image), "input_shape": list(tensor.shape)})
    finally:
        trt_detector.close()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"ort_boxes", "trt_boxes", "ort_metadata", "trt_metadata"}}, indent=2))
    print(f"Saved: {path.resolve()}")
    if result["status"] == "invalid_nonfinite_output":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
