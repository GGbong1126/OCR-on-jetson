"""Single-image OCR text detection (no text recognition)."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from ocr_det import create_detector, read_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["ort", "trt"], default="ort")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="reports/smoke")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    detector = create_detector(args.backend, args.model, args.threads)
    start = perf_counter()
    image = read_image(args.image)
    decode_ms = (perf_counter() - start) * 1000
    boxes, timings, input_shape = detector.predict(image)
    timings["decode_ms"] = decode_ms
    result = {
        "metadata": detector.metadata(), "image": str(Path(args.image).resolve()),
        "image_shape": list(image.shape), "input_shape": input_shape,
        "box_count": len(boxes), "boxes": [b.tolist() for b in boxes],
        "first_run_timings": timings,
        "note": "First inference, not a warmed-up benchmark; detection only.",
    }
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "detections.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    canvas = image.copy()
    for i, box in enumerate(boxes):
        # Legacy JSON vertex order can cross; hull is used for drawing only.
        hull = cv2.convexHull(box.astype(np.int32))
        cv2.polylines(canvas, [hull], True, (0, 0, 255), 2)
        anchor = (int(box[:, 0].min()), max(15, int(box[:, 1].min())))
        cv2.putText(canvas, str(i), anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    ok, encoded = cv2.imencode(".jpg", canvas)
    if not ok:
        raise RuntimeError("Failed to encode visualization")
    encoded.tofile(directory / "detections.jpg")
    print(json.dumps({"box_count": len(boxes), "input_shape": input_shape,
                      "timings": timings, "output": str(directory.resolve())}, indent=2))


if __name__ == "__main__":
    main()
