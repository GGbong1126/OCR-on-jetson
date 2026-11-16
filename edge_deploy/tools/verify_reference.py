"""Compare extracted preprocessing and final boxes against server.py, without importing the service."""
import argparse
import ast
import math
from pathlib import Path

import cv2
import numpy as np

from ocr_det import Detector, postprocess, preprocess, read_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    tree = ast.parse(Path(args.server).read_text(encoding="utf-8-sig"))
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_ONNXOCREngine")
    method = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "_det_run")
    namespace = {"cv2": cv2, "np": np, "math": math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), args.server, "exec"), namespace)
    cv2.setNumThreads(1)
    detector = Detector(args.model)
    for name, image in [("real", read_image(args.image)), ("blank", np.full((480, 640, 3), 255, dtype=np.uint8))]:
        tensor = preprocess(image)
        raw = detector.session.run([detector.output_name], {detector.input_name: tensor})[0]

        class ReplaySession:
            def get_inputs(self):
                return detector.session.get_inputs()

            def run(self, names, inputs):
                np.testing.assert_array_equal(inputs[detector.input_name], tensor)
                return [raw]

        class Reference:
            det_session = ReplaySession()

        expected = namespace["_det_run"](Reference(), image)
        actual = postprocess(raw, image.shape)
        assert len(expected) == len(actual), (name, len(expected), len(actual))
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)
        print(f"PASS {name}: identical input tensor, box coordinates and order; boxes={len(actual)}; shape={tensor.shape}")
    assert postprocess(np.zeros((1, 1, 32, 32), dtype=np.float32), (100, 100, 3)) == []
    try:
        postprocess(np.full((1, 1, 32, 32), np.nan, dtype=np.float32), (100, 100, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("Non-finite output was not rejected")
    print("PASS empty probability map and invalid output checks")


if __name__ == "__main__":
    main()
