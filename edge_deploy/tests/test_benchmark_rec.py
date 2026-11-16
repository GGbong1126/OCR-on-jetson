"""CPU-only checks for timing separation, validation and report persistence."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_rec import main, measure, prepare_rows, stats, validate_profile


class FakeRecognizer:
    characters = ["<blank>", "A", " "]

    def __init__(self):
        self.widths = []
        self.fail_after = None
        # Deliberately reuse a borrowed output buffer, as the TRT runner does.
        self.raw = np.zeros((1, 1, 3), dtype=np.float32)

    def run_tensor(self, tensor):
        width = tensor.shape[3]
        self.widths.append(width)
        self.raw.fill(0)
        self.raw[0, 0, 1 if width == 32 else 0] = 1
        if self.fail_after is not None and len(self.widths) > self.fail_after:
            self.raw.fill(np.nan)
        return self.raw

    def metadata(self):
        return {"backend": "fake_test_only"}


class RecBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((48, 80, 3), dtype=np.uint8)
        self.rows = [
            {"index": 7, "box": [[0, 0], [32, 0], [32, 48], [0, 48]],
             "input_shape": [1, 3, 48, 32], "status": "ok", "text": "A"},
            {"index": 31, "box": [[32, 0], [80, 0], [80, 48], [32, 48]],
             "input_shape": [1, 3, 48, 48], "status": "empty_text", "text": ""},
        ]
        self.source = {"status": "compared_not_accuracy_validated", "image_sha256": "image",
                       "recognizer": {"model_sha256": "model", "dictionary_sha256": "dict"},
                       "reference": {"records": self.rows}}

    def test_preflight_shapes_text_and_hashes(self):
        self.assertIs(prepare_rows(self.source, self.image, "image", "model", "dict", "reference"), self.rows)
        with self.assertRaisesRegex(ValueError, "Image SHA256"):
            prepare_rows(self.source, self.image, "wrong", "model", "dict", "reference")
        for field, value, message in (("input_shape", [1, 3, 48, 33], "shape mismatch"),
                                      ("text", None, "baseline text"),
                                      ("text", "", "status and text")):
            source = deepcopy(self.source)
            source["reference"]["records"][0][field] = value
            with self.assertRaisesRegex(ValueError, message):
                prepare_rows(source, self.image, "image", "model", "dict", "reference")

    def test_warmup_separate_original_order_and_shared_empty(self):
        rec, report = FakeRecognizer(), {}
        with redirect_stdout(io.StringIO()):
            measure(rec, self.image, self.rows, 1, 2, report)
        self.assertEqual(rec.widths, [32, 48] * 3)
        self.assertEqual(len(report["warmup_records"]), 2)
        self.assertEqual(len(report["records"]), 4)
        self.assertEqual(report["cycle_summary_ms"]["inference_ms"]["count"], 2)
        self.assertEqual(report["per_width_inference_ms"]["32"]["count"], 2)
        self.assertEqual(report["validation"]["source_empty_indices"], [31])
        for cycle in report["cycles"]:
            samples = [r for r in report["records"] if r["cycle"] == cycle["cycle"]]
            self.assertEqual(cycle["inference_ms"], sum(r["inference_ms"] for r in samples))
            self.assertEqual(cycle["empty_indices"], [31])
            self.assertGreaterEqual(cycle["crop_rec_ms"], cycle["inference_ms"])
        self.assertTrue(all("text" not in r for r in report["records"]))

    def test_text_change_during_warmup_invalidates_run(self):
        rows = deepcopy(self.rows)
        rows[0]["text"] = "wrong"
        report = {}
        with self.assertRaisesRegex(ValueError, "text differs"):
            measure(FakeRecognizer(), self.image, rows, 1, 2, report)
        self.assertFalse(report["warmup_records"][0]["text_equal"])
        self.assertEqual(report["records"], [])
        self.assertNotIn("cycle_summary_ms", report)

    def test_nonfinite_after_warmup_preserves_error(self):
        rec, report = FakeRecognizer(), {}
        rec.fail_after = 2
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "non-finite"):
            measure(rec, self.image, self.rows, 1, 2, report)
        self.assertEqual(len(report["warmup_cycles"]), 1)
        self.assertEqual(report["records"][0]["index"], 7)
        self.assertEqual(report["records"][0]["status"], "error")
        self.assertNotIn("cycle_summary_ms", report)

    def test_profile_all_crops_checked(self):
        profile = [[1, 3, 48, 32], [1, 3, 48, 160], [1, 3, 48, 1024]]
        validate_profile(self.rows, profile)
        rows = deepcopy(self.rows)
        rows[1]["input_shape"][3] = 1025
        with self.assertRaisesRegex(ValueError, "Region 31.*outside"):
            validate_profile(rows, profile)

    def test_stats_and_positive_cycles(self):
        self.assertEqual(stats([1, 3])["mean"], 2)
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                stats(values)
        for warmup, repeat in ((0, 1), (1, 0)):
            with self.assertRaises(ValueError):
                measure(FakeRecognizer(), self.image, self.rows, warmup, repeat, {})

    def test_main_saves_partial_failure_and_refuses_overwrite(self):
        # Only temporary fixture files, never actual models or source reports.
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.json"
            source.write_text(json.dumps(self.source), encoding="utf-8")
            output = directory / "result.json"
            argv = ["--backend", "ort", "--model", "model", "--onnx", "model",
                    "--dict", "dict", "--image", "image", "--det-report", str(source),
                    "--warmup", "1", "--repeat", "2", "--output", str(output)]
            rec = FakeRecognizer()
            rec.fail_after = 2
            with patch("benchmark_rec.sha256_file", side_effect=lambda path: str(path)), \
                    patch("benchmark_rec.read_image", return_value=self.image), \
                    patch("benchmark_rec.Recognizer", return_value=rec), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(argv)
                self.assertEqual(error.exception.code, 2)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "invalid_benchmark")
            self.assertEqual(result["records"][0]["index"], 7)
            self.assertNotIn("cycle_summary_ms", result)
            previous = output.read_bytes()
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(argv)
            self.assertEqual(output.read_bytes(), previous)

    def test_main_trt_does_not_create_cpu_session_and_closes_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text(json.dumps(self.source), encoding="utf-8")
            output = Path(directory) / "result.json"
            argv = ["--backend", "trt", "--model", "engine", "--onnx", "model",
                    "--dict", "dict", "--image", "image", "--det-report", str(source),
                    "--warmup", "1", "--repeat", "1", "--output", str(output)]
            rec = FakeRecognizer()
            rec.profile = [[1, 3, 48, 32], [1, 3, 48, 160], [1, 3, 48, 1024]]
            from unittest.mock import Mock
            rec.close = Mock()
            with patch("benchmark_rec.sha256_file", side_effect=lambda path: str(path)), \
                    patch("benchmark_rec.read_image", return_value=self.image), \
                    patch("benchmark_rec.Recognizer") as cpu, \
                    patch("benchmark_rec.TensorRTRecognizer", return_value=rec), redirect_stdout(io.StringIO()):
                main(argv)
            cpu.assert_not_called()
            rec.close.assert_called_once()
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "completed_consistent_not_accuracy_validated")
            self.assertEqual(result["validation"]["measured_checked_count"], 2)


if __name__ == "__main__":
    unittest.main()
