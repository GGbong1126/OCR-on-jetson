"""CPU-only numeric, provenance and mocked TensorRT-adapter checks."""
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_rec import compare_rec_outputs, summarize, verify_source
from ocr_rec import TensorRTRecognizer, validate_rec_shapes
from trt_backend import TensorRTDetector


class RecOutputTests(unittest.TestCase):
    def setUp(self):
        self.chars = ["<blank>", "A", " "]
        self.raw = np.array([[[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]]], dtype=np.float32)

    def test_same_text_with_numeric_difference(self):
        other = self.raw.copy()
        other[0, 0, 1] += 0.01
        r = compare_rec_outputs(self.raw, other, self.chars)
        self.assertTrue(r["text_equal"])
        self.assertFalse(r["allclose_rtol_1e_3_atol_1e_4"])
        self.assertGreater(r["score_abs_error"], 0)
        self.assertEqual(r["timestep_argmax_disagreement_fraction"], 0)

    def test_nonfinite_and_shape_change(self):
        for value in (np.nan, np.inf, -np.inf):
            r = compare_rec_outputs(self.raw, np.full_like(self.raw, value), self.chars)
            self.assertEqual(r["status"], "invalid_nonfinite_output")
            self.assertNotIn("text_equal", r)
        r = compare_rec_outputs(self.raw, self.raw[:, :1], self.chars)
        self.assertEqual(r["status"], "invalid_output_shape")

    def test_changed_empty_text_and_invalid_summary(self):
        other = self.raw.copy()
        other[0, 0] = [0.8, 0.1, 0.1]
        r = {"index": 31, **compare_rec_outputs(self.raw, other, self.chars)}
        s = summarize([r, {"index": 32, "status": "error"}])
        self.assertEqual(s["status"], "invalid_comparison")
        self.assertEqual(s["changed_text_indices"], [31])
        self.assertEqual(s["candidate_empty_indices"], [31])
        self.assertEqual(s["invalid_indices"], [32])
        self.assertEqual(summarize([])["status"], "invalid_comparison")

    def test_source_provenance(self):
        meta = {"model_sha256": "model", "dictionary_sha256": "dict"}
        source = {"status": "compared_not_accuracy_validated", "image_sha256": "image",
                  "recognizer": meta, "reference": {"records": [{"index": 0, "status": "empty_text"}]}}
        self.assertEqual(len(verify_source(source, "image", meta, "reference")), 1)
        with self.assertRaisesRegex(ValueError, "Image SHA256"):
            verify_source(source, "wrong", meta, "reference")
        with self.assertRaisesRegex(ValueError, "dictionary_sha256 mismatch"):
            verify_source(source, "image", {**meta, "dictionary_sha256": "wrong"}, "reference")
        with self.assertRaisesRegex(ValueError, "model_sha256 mismatch"):
            verify_source(source, "image", {**meta, "model_sha256": "wrong"}, "reference")
        source["reference"]["records"] *= 2
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            verify_source(source, "image", meta, "reference")


class RecAdapterTests(unittest.TestCase):
    def test_rec_contract_rejects_det_or_wrong_dictionary(self):
        validate_rec_shapes([-1, 3, 48, -1], [-1, -1, 18385], 18385)
        for inp, out in (([-1, 3, 960, 704], [1, 1, 960, 704]),
                         ([-1, 3, 48, -1], [-1, -1, 10])):
            with self.assertRaises(ValueError):
                validate_rec_shapes(inp, out, 18385)

    def test_out_of_profile_rejected_before_cuda(self):
        runner = TensorRTDetector.__new__(TensorRTDetector)
        runner.closed = False
        runner.current_shape = None
        runner.input_name = "x"
        runner.engine = MagicMock()
        runner.engine.get_tensor_profile_shape.return_value = ([1, 3, 48, 32], [1, 3, 48, 160], [1, 3, 48, 1024])
        rec = TensorRTRecognizer.__new__(TensorRTRecognizer)
        rec.runner = runner
        try:
            with self.assertRaisesRegex(ValueError, "outside engine profile"):
                rec.run_tensor(np.zeros((1, 3, 48, 1025), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "recognition tensor"):
                rec.run_tensor(np.zeros((1, 3, 960, 704), dtype=np.float32))
        finally:
            runner.closed = True

    def test_adapter_uses_raw_runner_and_cleans_up_invalid_engine(self):
        runner = MagicMock()
        runner.input_name, runner.output_name = "x", "y"
        runner.engine.get_tensor_shape.side_effect = lambda name: [-1, 3, 48, -1] if name == "x" else [-1, -1, 3]
        runner.engine.get_tensor_profile_shape.return_value = ([1, 3, 48, 32], [1, 3, 48, 160], [1, 3, 48, 1024])
        with patch("ocr_rec.load_characters", return_value=["<blank>", "A", " "]), patch("trt_backend.TensorRTDetector", return_value=runner):
            rec = TensorRTRecognizer("model.engine", "dict.json")
            tensor = np.zeros((1, 3, 48, 43), dtype=np.float32)
            rec.run_tensor(tensor)
            runner.run_tensor.assert_called_once_with(tensor)
            runner.predict.assert_not_called()
            rec.close()
            runner.close.assert_called_once()
            runner.reset_mock()
            runner.engine.get_tensor_shape.side_effect = lambda name: [-1, 3, 960, 704] if name == "x" else [1, 1, 960, 704]
            with self.assertRaises(ValueError):
                TensorRTRecognizer("det.engine", "dict.json")
            runner.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
