"""CPU-only live-pipeline, resource failure and memory-query tests."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_ocr import (close_models, main, measure_pipeline, memory_snapshot,
                           parse_memory_csv, run_pipeline, verify_detector)
from test_benchmark_rec import FakeRecognizer


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = deepcopy(boxes)
        self.shape = [1, 3, 960, 1600]
        self.calls = 0
        self.closed = False

    def predict(self, image):
        if self.closed:
            raise RuntimeError("Detector was closed early")
        self.calls += 1
        return deepcopy(self.boxes), {"preprocess_ms": 0., "inference_ms": 0.,
                                      "postprocess_ms": 0., "pipeline_ms": 0.}, self.shape

    def metadata(self):
        return {"backend": "tensorrt", "engine_sha256": "det"}

    def close(self):
        self.closed = True


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((48, 80, 3), dtype=np.uint8)
        self.rows = [
            {"index": 7, "box": [[0, 0], [32, 0], [32, 48], [0, 48]],
             "input_shape": [1, 3, 48, 32], "status": "ok", "text": "A"},
            {"index": 31, "box": [[32, 0], [80, 0], [80, 48], [32, 48]],
             "input_shape": [1, 3, 48, 48], "status": "empty_text", "text": ""},
        ]
        self.det = FakeDetector([r["box"] for r in self.rows])
        self.source = {"status": "compared_not_accuracy_validated", "image_sha256": "image",
                       "recognizer": {"model_sha256": "model", "dictionary_sha256": "dict"},
                       "reference": {"records": self.rows, "detector": self.det.metadata(),
                                     "input_shape": self.det.shape}}

    def test_live_detection_every_cycle_and_warmup_exclusion(self):
        report, rec, snapshot = {}, FakeRecognizer(), Mock()
        with redirect_stdout(io.StringIO()):
            measure_pipeline(self.det, rec, self.image, self.rows, self.det.shape, 1, 2, report, snapshot)
        self.assertEqual(self.det.calls, 4)  # validation + warmup + 2 measured
        self.assertEqual(rec.widths, [32, 48]*4)
        self.assertEqual(report["summary_ms"]["pipeline_wall_ms"]["count"], 2)
        self.assertEqual(report["validation"]["measured_rec_calls"], 4)
        self.assertEqual(report["validation"]["source_empty_indices"], [31])
        self.assertEqual(len(report["warmup_cycles"]), 1)
        snapshot.assert_called_once_with("after_validation")
        for sample in report["cycles"]:
            self.assertGreaterEqual(sample["timings"]["pipeline_wall_ms"], sample["timings"]["rec_cycle_wall_ms"])
            self.assertTrue(all("text" not in r for r in sample["records"]))

    def test_live_changed_geometry_or_shape_stops_before_rec(self):
        for kind in ("count", "box", "shape"):
            det = FakeDetector([r["box"] for r in self.rows])
            if kind == "count":
                det.boxes = []
            elif kind == "box":
                det.boxes[0][0][0] += 1
            else:
                det.shape = [1, 3, 32, 32]
            rec, sample = FakeRecognizer(), {}
            with self.assertRaises(ValueError):
                run_pipeline(det, rec, self.image, self.rows, self.det.shape, "validation", 0, sample)
            self.assertEqual(rec.widths, [])
            self.assertEqual(sample["status"], "error")
            self.assertIn("boxes", sample)

    def test_text_failure_preserves_partial_pipeline(self):
        rows, sample = deepcopy(self.rows), {}
        rows[1]["text"] = "different"
        with self.assertRaisesRegex(ValueError, "text differs"):
            run_pipeline(self.det, FakeRecognizer(), self.image, rows, self.det.shape, "measured", 0, sample)
        self.assertEqual(sample["status"], "error")
        self.assertEqual(len(sample["records"]), 2)
        self.assertFalse(sample["records"][1]["text_equal"])
        self.assertNotIn("timings", sample)

    def test_nonfinite_warmup_stops_and_retains_validation(self):
        report, rec = {}, FakeRecognizer()
        rec.fail_after = 2
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "non-finite"):
            measure_pipeline(self.det, rec, self.image, self.rows, self.det.shape, 1, 2, report)
        self.assertEqual(report["validation_run"]["status"], "consistent")
        self.assertEqual(report["warmup_cycles"][0]["records"][0]["status"], "error")
        self.assertNotIn("summary_ms", report)

    def test_detector_identity_and_positive_cycles(self):
        self.assertEqual(verify_detector(self.source["reference"], "trt", "det"), self.det.shape)
        for backend, artifact in (("ort", "det"), ("trt", "wrong")):
            with self.assertRaisesRegex(ValueError, "keep detector fixed"):
                verify_detector(self.source["reference"], backend, artifact)
        with self.assertRaises(ValueError):
            measure_pipeline(self.det, FakeRecognizer(), self.image, self.rows, self.det.shape, 0, 1, {})

    def test_memory_unknowns_and_unavailable_query(self):
        device = parse_memory_csv('0, "GPU, test", 4096, 800, 3296\n')[0]
        self.assertEqual(device["name"], "GPU, test")
        self.assertEqual(device["used_mib"], 800)
        device = parse_memory_csv("0, GPU, N/A, [Not Supported], nan")[0]
        self.assertIsNone(device["total_mib"])
        self.assertIsNone(device["used_mib"])
        self.assertIsNone(device["free_mib"])
        with patch("benchmark_ocr.subprocess.run", side_effect=FileNotFoundError("missing")):
            self.assertFalse(memory_snapshot("test")["available"])
        with patch("benchmark_ocr.subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)):
            self.assertFalse(memory_snapshot("test")["available"])
        with patch("benchmark_ocr.subprocess.run") as run:
            self.assertFalse(memory_snapshot("test", False)["available"])
            run.assert_not_called()

    def test_cleanup_attempts_both_models(self):
        rec, det = Mock(), Mock()
        rec.close.side_effect = RuntimeError("cleanup failed")
        errors = close_models(rec, det)
        det.close.assert_called_once()
        self.assertEqual(errors[0]["model"], "rec")

    def run_mock_main(self, directory, rec_factory):
        source, output = Path(directory)/"source.json", Path(directory)/"output.json"
        source.write_text(json.dumps(self.source), encoding="utf-8")
        args = ["--det-model", "det", "--rec-backend", "trt", "--rec-model", "engine",
                "--rec-onnx", "model", "--dict", "dict", "--image", "image",
                "--source-report", str(source), "--warmup", "1", "--repeat", "1",
                "--output", str(output), "--skip-memory"]
        with patch("benchmark_ocr.sha256_file", side_effect=str), \
                patch("benchmark_ocr.read_image", return_value=self.image), \
                patch("benchmark_ocr.create_detector", return_value=self.det), \
                patch("benchmark_ocr.Recognizer") as cpu, \
                patch("benchmark_ocr.TensorRTRecognizer", side_effect=rec_factory), \
                redirect_stdout(io.StringIO()):
            try:
                main(args)
                code = 0
            except SystemExit as error:
                code = error.code
            cpu.assert_not_called()
        return code, json.loads(output.read_text(encoding="utf-8")), args

    def test_main_both_resident_cleanup_success_and_no_overwrite(self):
        rec = FakeRecognizer()
        rec.close = Mock()
        rec.profile = [[1, 3, 48, 32], [1, 3, 48, 160], [1, 3, 48, 1024]]
        original_run = rec.run_tensor

        def resident_run(tensor):
            self.assertFalse(self.det.closed)
            return original_run(tensor)

        rec.run_tensor = resident_run
        with tempfile.TemporaryDirectory() as directory:
            code, report, args = self.run_mock_main(directory, lambda *unused: rec)
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "completed_consistent_not_accuracy_validated")
            self.assertEqual(len(report["memory_snapshots"]), 6)
            self.assertTrue(self.det.closed)
            rec.close.assert_called_once()
            previous = (Path(directory)/"output.json").read_bytes()
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertEqual((Path(directory)/"output.json").read_bytes(), previous)

    def test_rec_load_failure_closes_detector_and_saves_error(self):
        def fail(*unused):
            raise RuntimeError("simulated out of memory")

        with tempfile.TemporaryDirectory() as directory:
            code, report, _ = self.run_mock_main(directory, fail)
            self.assertEqual(code, 2)
            self.assertTrue(self.det.closed)
            self.assertEqual(report["status"], "invalid_benchmark")
            self.assertIn("out of memory", report["error"])
            self.assertNotIn("summary_ms", report)


if __name__ == "__main__":
    unittest.main()
