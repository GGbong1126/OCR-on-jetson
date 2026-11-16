"""CPU detection baseline extracted from server.py; preserves its semantics.

This is a reference for the current application, not a claim of parity with
the original Paddle model. Keep BGR, /255 and legacy box ordering unchanged.
"""
import hashlib
import math
import platform
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(image):
    h, w = image.shape[:2]
    ratio = 960 / max(h, w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    new_h = max(((new_h + 31) // 32) * 32, 32)
    new_w = max(((new_w + 31) // 32) * 32, 32)
    img = cv2.resize(image, (new_w, new_h))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(np.transpose(img, (2, 0, 1)), 0)


def postprocess(output, image_shape):
    h, w = image_shape[:2]
    if output.ndim != 4 or output.shape[:2] != (1, 1):
        raise ValueError(f"Expected [1, 1, H, W], got {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("Model produced NaN or Inf")
    prob = output[0, 0]
    pred_h, pred_w = prob.shape
    width_scale, height_scale = w / pred_w, h / pred_h
    mask = (prob > 0.1).astype(np.uint8)
    contours, _ = cv2.findContours(mask * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        mini_box = cv2.minAreaRect(cnt.reshape(-1, 2))
        if min(mini_box[1]) < 3:
            continue
        points_arr = np.array(list(cv2.boxPoints(mini_box)), dtype=np.float32)
        xmin = max(0, int(math.floor(points_arr[:, 0].min())))
        xmax = min(pred_w - 1, max(0, int(math.ceil(points_arr[:, 0].max()))))
        ymin = max(0, int(math.floor(points_arr[:, 1].min())))
        ymax = min(pred_h - 1, max(0, int(math.ceil(points_arr[:, 1].max()))))
        if xmax <= xmin or ymax <= ymin:
            continue
        roi = prob[ymin:ymax + 1, xmin:xmax + 1]
        shifted = points_arr.copy()
        shifted[:, 0] -= xmin
        shifted[:, 1] -= ymin
        fill_mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        cv2.fillPoly(fill_mask, [shifted.astype(np.int32)], 1)
        if cv2.mean(roi, fill_mask)[0] < 0.3:
            continue
        area = cv2.contourArea(points_arr)
        length = cv2.arcLength(points_arr, True)
        if area < 1 or length < 1:
            continue
        try:
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(points_arr.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded_list = offset.Execute(area * 1.2 / length)
            if not expanded_list:
                continue
            expanded = np.array(expanded_list[0], dtype=np.float32)
        except Exception:
            # Preserve the application's handling of degenerate polygons.
            continue
        if len(expanded) == 0:
            continue
        expanded_box = cv2.minAreaRect(expanded.reshape(-1, 2).astype(np.float32))
        final_points = sorted(list(cv2.boxPoints(expanded_box)), key=lambda x: x[0])
        if min(expanded_box[1]) < 5:
            continue
        box = np.array(final_points, dtype=np.float32)
        box[:, 0] = np.clip(np.round(box[:, 0] * width_scale), 0, w)
        box[:, 1] = np.clip(np.round(box[:, 1] * height_scale), 0, h)
        boxes.append(box.astype(np.int32))
    # Preserve server.py's legacy point and reading order for parity.
    if boxes:
        boxes.sort(key=lambda b: int((b[0, 1] + b[2, 1]) / 2))
        avg_h = np.mean([b[2, 1] - b[0, 1] for b in boxes])
        merged, row = [], [boxes[0]]
        for box in boxes[1:]:
            cy0 = int((row[-1][0, 1] + row[-1][2, 1]) / 2)
            cy1 = int((box[0, 1] + box[2, 1]) / 2)
            if abs(cy1 - cy0) < avg_h * 0.8:
                row.append(box)
            else:
                row.sort(key=lambda b: b[0, 0])
                merged.extend(row)
                row = [box]
        row.sort(key=lambda b: b[0, 0])
        merged.extend(row)
        boxes = merged
    return boxes


class Detector:
    def __init__(self, model, threads=2):
        if threads < 1:
            raise ValueError("threads must be positive")
        self.model = Path(model)
        self.threads = threads
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        start = perf_counter()
        self.session = ort.InferenceSession(
            str(self.model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.load_ms = (perf_counter() - start) * 1000
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)":
            raise ValueError("Expected a single float32 image input")
        self.input_name = inputs[0].name
        self.output_name = self.session.get_outputs()[0].name

    def run_tensor(self, tensor):
        return self.session.run([self.output_name], {self.input_name: tensor})[0]

    def predict(self, image):
        start = perf_counter()
        tensor = preprocess(image)
        prepared = perf_counter()
        output = self.run_tensor(tensor)
        inferred = perf_counter()
        boxes = postprocess(output, image.shape)
        end = perf_counter()
        timings = {
            "preprocess_ms": (prepared - start) * 1000,
            "inference_ms": (inferred - prepared) * 1000,
            "postprocess_ms": (end - inferred) * 1000,
            "pipeline_ms": (end - start) * 1000,
        }
        return boxes, timings, list(tensor.shape)

    def metadata(self):
        return {
            "backend": "ort_cpu",
            "model": str(self.model.resolve()), "model_sha256": sha256_file(self.model),
            "python": platform.python_version(), "platform": platform.platform(),
            "onnxruntime": ort.__version__, "numpy": np.__version__,
            "opencv": cv2.__version__, "providers": self.session.get_providers(),
            "intra_op_threads": self.threads, "opencv_threads": cv2.getNumThreads(),
            "session_load_ms": self.load_ms, "batch_size": 1,
            "preprocessing": "BGR / 255; long side 960 (upscale allowed); ceil to 32",
            "postprocessing": "server.py reference: thresh=0.1, box_thresh=0.3, unclip=1.2",
        }


def create_detector(backend, model, threads=2):
    if backend == "ort":
        return Detector(model, threads)
    from trt_backend import TensorRTDetector
    return TensorRTDetector(model)
