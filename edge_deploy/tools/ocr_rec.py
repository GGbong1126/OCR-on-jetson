"""Fixed ORT CPU recognizer preserving server.py crop/preprocess/CTC semantics."""
import json
import math
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort

from ocr_det import sha256_file


def crop_text_region(image, box):
    box = np.asarray(box)
    if box.shape != (4, 2) or not np.isfinite(box).all():
        raise ValueError("Expected finite [4,2] box")
    if not np.equal(box, np.round(box)).all():
        raise ValueError("Expected integer pixel coordinates from detector")
    box = box.astype(np.int64)
    h, w = image.shape[:2]
    x1, y1 = max(0, box[:, 0].min()), max(0, box[:, 1].min())
    x2, y2 = min(w, box[:, 0].max()), min(h, box[:, 1].max())
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def preprocess_rec(cropped):
    if cropped.ndim != 3 or cropped.shape[2] != 3 or cropped.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR image")
    h, w = cropped.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Empty recognition crop")
    content_w = max(1, min(3200, int(math.ceil(48 * (w / float(h))))))
    resized = cv2.resize(cropped, (content_w, 48))
    if content_w < 32:
        padded = np.full((48, 32, 3), 255, dtype=resized.dtype)
        padded[:, :content_w] = resized
        resized = padded
    img = resized.astype(np.float32)
    img = img.transpose((2, 0, 1)) / 255.0
    img = (img - 0.5) / 0.5
    return np.expand_dims(img, 0)


def decode_ctc(output, characters):
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != len(characters):
        raise ValueError(f"Expected [1,T,{len(characters)}], got {output.shape}")
    if output.shape[1] < 1 or not np.isfinite(output).all():
        raise ValueError("Empty or non-finite recognition output")
    probs = output[0]
    indices, scores = np.argmax(probs, axis=1), np.max(probs, axis=1)
    chars, char_scores, prev = [], [], -1
    for idx, score in zip(indices, scores):
        if idx == 0:
            prev = -1
            continue
        if idx == prev:
            continue
        chars.append(characters[idx])
        char_scores.append(score)
        prev = idx
    return ("".join(chars), float(np.mean(char_scores))) if chars else ("", 0.0)


def load_characters(dictionary):
    chars = json.loads(Path(dictionary).read_text(encoding="utf-8"))
    if not isinstance(chars, list) or not chars or any(not isinstance(c, str) or not c for c in chars):
        raise ValueError("Dictionary must be a nonempty JSON list of nonempty strings")
    return ["<blank>"] + chars + [" "]


def validate_rec_shapes(input_shape, output_shape, classes):
    if len(input_shape) != 4 or list(input_shape[1:3]) != [3, 48]:
        raise ValueError(f"Expected [N,3,48,W] recognition input, got {input_shape}")
    if len(output_shape) != 3:
        raise ValueError("Expected [N,T,C] recognition output")
    if isinstance(output_shape[2], int) and output_shape[2] > 0 and output_shape[2] != classes:
        raise ValueError("Recognition output classes do not match dictionary + blank + space")


class Recognizer:
    def __init__(self, model, dictionary, threads=2):
        if threads < 1:
            raise ValueError("threads must be positive")
        self.model, self.dictionary = Path(model), Path(dictionary)
        self.characters = load_characters(self.dictionary)
        self.threads = threads
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        start = perf_counter()
        self.session = ort.InferenceSession(str(self.model), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.load_ms = (perf_counter() - start) * 1000
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected one recognition input/output")
        inp, out = inputs[0], outputs[0]
        if inp.type != "tensor(float)" or out.type != "tensor(float)":
            raise ValueError("Expected FP32 recognition input/output")
        validate_rec_shapes(inp.shape, out.shape, len(self.characters))
        self.input_name, self.output_name = inp.name, out.name

    def run_tensor(self, tensor):
        return self.session.run([self.output_name], {self.input_name: tensor})[0]

    def predict(self, crop):
        start = perf_counter()
        tensor = preprocess_rec(crop)
        prepared = perf_counter()
        raw = self.run_tensor(tensor)
        inferred = perf_counter()
        text, score = decode_ctc(raw, self.characters)
        end = perf_counter()
        return text, score, {
            "preprocess_ms": (prepared - start) * 1000,
            "inference_ms": (inferred - prepared) * 1000,
            "decode_ms": (end - inferred) * 1000,
        }, list(tensor.shape)

    def metadata(self):
        return {
            "backend": "ort_cpu", "model": str(self.model.resolve()),
            "model_sha256": sha256_file(self.model),
            "dictionary": str(self.dictionary.resolve()),
            "dictionary_sha256": sha256_file(self.dictionary), "classes": len(self.characters),
            "onnxruntime": ort.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
            "providers": self.session.get_providers(), "threads": self.threads,
            "session_load_ms": self.load_ms, "batch_size": 1,
            "preprocess": "server BGR; height 48; width ceil(48*w/h), capped 3200; white right padding to min 32; (x/255-0.5)/0.5",
            "decode": "server greedy CTC; blank=0 resets repeat suppression; no softmax added; no confidence filter",
        }


class TensorRTRecognizer(Recognizer):
    """Reuse raw CUDA execution only, never detector preprocessing/postprocessing.

    Buffers use exact sizes, so a changed width may reallocate; not yet a
    high-water-mark buffer pool. run_tensor output is borrowed until next call.
    """
    def __init__(self, model, dictionary):
        from trt_backend import TensorRTDetector
        self.model, self.dictionary = Path(model), Path(dictionary)
        self.characters = load_characters(self.dictionary)
        self.runner = TensorRTDetector(model)
        try:
            runner = self.runner
            validate_rec_shapes(list(runner.engine.get_tensor_shape(runner.input_name)),
                                list(runner.engine.get_tensor_shape(runner.output_name)), len(self.characters))
            self.profile = [list(s) for s in runner.engine.get_tensor_profile_shape(runner.input_name, 0)]
            if any(s[:3] != [1, 3, 48] for s in self.profile):
                raise ValueError("Recognition runner requires batch=1, channels=3, height=48 in profile")
        except Exception:
            self.runner.close()
            raise

    def run_tensor(self, tensor):
        if tensor.ndim != 4 or tensor.shape[:3] != (1, 3, 48):
            raise ValueError("Expected recognition tensor [1,3,48,W]")
        return self.runner.run_tensor(tensor)

    def metadata(self):
        meta = self.runner.metadata()
        meta.update({"task": "recognition", "dictionary": str(self.dictionary.resolve()),
                     "dictionary_sha256": sha256_file(self.dictionary), "classes": len(self.characters),
                     "preprocessing": "same server rec: BGR, H=48, W=32..3200, white right padding if narrow; normalize to [-1,1]; engine profile separately enforced",
                     "postprocessing": "same greedy CTC as ORT rec; no detection postprocess",
                     "timing": "H2D + enqueue + D2H + sync + host packing; width changes may reallocate device/host buffers",
                     "output_ownership": "borrowed; consume/copy before next call"})
        return meta

    def close(self):
        self.runner.close()


def recognize_boxes(recognizer, image, boxes, progress=None):
    records = []
    start = perf_counter()
    sums = {"crop_ms": 0.0, "rec_preprocess_ms": 0.0, "rec_inference_ms": 0.0, "rec_decode_ms": 0.0}
    for index, box in enumerate(boxes):
        record = {"index": index, "box": np.asarray(box).tolist()}
        try:
            t = perf_counter()
            crop = crop_text_region(image, box)
            sums["crop_ms"] += (perf_counter() - t) * 1000
            if crop is None:
                raise ValueError("Empty crop")
            text, score, timing, shape = recognizer.predict(crop)
            for key, value in timing.items():
                sums["rec_" + key] += value
            record.update(status="ok" if text else "empty_text", text=text, score=score,
                          input_shape=shape, timings=timing)
        except Exception as error:
            # Preserve evidence and continue collecting; caller marks entire experiment invalid.
            record.update(status="error", error=f"{type(error).__name__}: {error}")
        records.append(record)
        if progress is not None and ((index + 1) % 10 == 0 or index + 1 == len(boxes)):
            progress(index + 1, len(boxes))
    sums["crop_rec_wall_ms"] = (perf_counter() - start) * 1000
    return records, sums
