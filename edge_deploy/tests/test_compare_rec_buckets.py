"""CPU/fake-runner checks for lossless padding and honest text regression gates."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_rec_buckets import (pad_white, validate_buckets, make_plan, compare_rows,
                                 edit_distance, main, SUCCESS, CHANGED)


class FakeRec:
    characters = ["", "甲", " "]
    profile = [[1, 3, 48, 32], [1, 3, 48, 64], [1, 3, 48, 128]]

    def __init__(self, *args):
        self.widths = []
        self.buffer = np.zeros((1, 16, 3), dtype=np.float32)
        self.change = False
        self.failure = None
        self.closed = False

    def run_tensor(self, tensor):
        self.widths.append(tensor.shape[3])
        if self.failure == "interrupt":
            raise KeyboardInterrupt
        if self.failure == "execution":
            raise RuntimeError("injected execution failure")
        self.buffer.fill(0)
        index = 1 if tensor[0, 0, 0, 0] == -1 else 0
        if self.change and len(self.widths) % 2 == 0:
            index = 2  # A real space; must not strip it or hide blank -> text.
        self.buffer[:, :, index] = 1
        if self.failure == "nan" and len(self.widths) == 2:
            self.buffer.fill(np.nan)
        return self.buffer[:, :tensor.shape[3] // 8, :]

    def metadata(self):
        return {"backend": "fake_test_only"}

    def close(self):
        self.closed = True


class BucketTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((48, 100, 3), 255, dtype=np.uint8)
        self.image[:, :32] = 0
        self.rows = [
            {"index": 7, "box": [[0, 0], [32, 0], [32, 48], [0, 48]],
             "input_shape": [1, 3, 48, 32], "text": "甲", "status": "ok"},
            {"index": 31, "box": [[40, 0], [85, 0], [85, 48], [40, 48]],
             "input_shape": [1, 3, 48, 45], "text": "", "status": "empty_text"},
        ]
        self.plan = make_plan(self.rows, [64, 128], FakeRec.profile)
        self.rec = FakeRec()

    def run_compare(self, report):
        with redirect_stdout(io.StringIO()):
            compare_rows(self.rec, self.image, self.rows, self.plan, report)

    def test_padding_preserves_pixels_noncontiguous_input_and_white(self):
        tensor = np.arange(48 * 45 * 3, dtype=np.float32).reshape(48, 45, 3).transpose(2, 0, 1)[None]
        original = tensor.copy()
        padded = pad_white(tensor, 64)
        np.testing.assert_array_equal(padded[:, :, :, :45], original)
        np.testing.assert_array_equal(tensor, original)
        self.assertTrue((padded[:, :, :, 45:] == 1).all())
        self.assertEqual(padded.dtype, np.float32)
        self.assertIs(pad_white(tensor, 45), tensor)

    def test_padding_rejects_truncation_nonfinite_dtype_and_shape(self):
        tensor = np.zeros((1, 3, 48, 45), dtype=np.float32)
        for data, width in ((tensor, 32), (tensor.astype(np.float16), 64),
                            (tensor[0], 64), (tensor + np.nan, 64)):
            with self.assertRaises(ValueError):
                pad_white(data, width)

    def test_bucket_preflight_rejects_invalid_even_unused_buckets(self):
        for buckets in ([], [32, 32], [64, 32], [31], [64.0], [True]):
            with self.assertRaises(ValueError):
                validate_buckets(buckets)
        for buckets in ([32], [64, 256]):
            with self.assertRaises(ValueError):
                make_plan(self.rows, buckets, FakeRec.profile)
        self.assertEqual(self.plan[1]["padded_input_shape"], [1, 3, 48, 64])

    def test_all_crops_two_calls_full_ctc_borrowed_output_and_empty(self):
        report = {}
        self.run_compare(report)
        self.assertEqual(report["status"], SUCCESS)
        self.assertEqual(self.rec.widths, [32, 64, 45, 64])
        self.assertEqual(report["summary"]["source_empty_indices"], [31])
        self.assertEqual(report["summary"]["padded_empty_indices"], [31])
        first = report["records"][0]
        self.assertNotEqual(first["original"]["output_shape"], first["padded"]["output_shape"])
        self.assertEqual(first["original_to_padded_edit_count"], 0)
        self.assertNotIn("甲", json.dumps(report, ensure_ascii=False))

    def test_changes_collect_all_indices_no_space_stripping_or_empty_filter(self):
        self.rec.change = True
        report = {}
        self.run_compare(report)
        self.assertEqual(report["status"], CHANGED)
        self.assertEqual(report["summary"]["padded_original_changed_indices"], [7, 31])
        self.assertEqual(report["summary"]["padded_empty_indices"], [])
        self.assertEqual(len(report["records"]), 2)
        self.assertEqual(report["records"][1]["padded"]["text_length"], 1)

    def test_reference_drift_is_not_padding_acceptance(self):
        self.rows[0]["text"] = "旧值"
        report = {}
        self.run_compare(report)
        self.assertEqual(report["status"], "completed_with_reference_drift_not_accepted")
        self.assertEqual(report["summary"]["original_source_changed_indices"], [7])
        self.assertEqual(report["summary"]["padded_original_changed_indices"], [])

    def test_failures_stop_and_retain_partial_record(self):
        for failure in ("nan", "interrupt", "execution"):
            self.rec = FakeRec()
            self.rec.failure = failure
            report = {}
            with self.assertRaises((ValueError, RuntimeError, KeyboardInterrupt)):
                self.run_compare(report)
            self.assertEqual(len(report["records"]), 1)
            self.assertEqual(report["records"][0]["status"], "error")
            self.assertNotIn("summary", report)
            if failure == "nan":
                self.assertFalse(report["records"][0]["padded"]["all_finite"])

    def test_unchanged_width_control_and_edit_count(self):
        self.plan = make_plan(self.rows, [32, 45, 64], FakeRec.profile)
        report = {}
        self.run_compare(report)
        self.assertEqual(report["summary"]["unchanged_width_indices"], [7, 31])
        self.assertEqual(report["records"][0]["original_tensor_sha256"], report["records"][0]["padded_tensor_sha256"])
        self.assertEqual(edit_distance("甲乙", "甲 "), 1)
        self.assertEqual(edit_distance("", "  "), 2)

    def test_main_success_changes_failures_cleanup_and_no_overwrite(self):
        source = {"status": "compared_not_accuracy_validated", "image_sha256": "image",
                  "recognizer": {"model_sha256": "onnx", "dictionary_sha256": "dict"},
                  "reference": {"records": self.rows}}
        crop_set = [{k: r[k] for k in ("index", "box", "input_shape")} for r in self.rows]
        baseline = {"status": SUCCESS, "rec_backend": "trt", "source_branch": "reference",
                    "source_report_sha256": "source", "image_sha256": "image",
                    "reference_rec_onnx_sha256": "onnx", "dictionary_sha256": "dict",
                    "crop_set_sha256": hashlib.sha256(json.dumps(crop_set, sort_keys=True).encode()).hexdigest(),
                    "recognizer": {"engine_sha256": "engine"}}
        for mode in ("ok", "change", "nan", "interrupt", "cleanup", "provenance"):
            rec = FakeRec()
            rec.change = mode == "change"
            rec.failure = mode if mode in ("nan", "interrupt") else None
            this_baseline = deepcopy(baseline)
            if mode == "provenance":
                this_baseline["recognizer"]["engine_sha256"] = "wrong"
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "report.json"
                argv = ["--engine", "engine", "--onnx", "onnx", "--dict", "dict", "--image", "image",
                        "--source-report", "source", "--baseline-report", "baseline", "--buckets", "64", "128",
                        "--output", str(output)]
                with patch("compare_rec_buckets.TensorRTRecognizer", return_value=rec) as factory, \
                     patch("compare_rec_buckets.read_image", return_value=self.image), \
                     patch("compare_rec_buckets.sha256_file", side_effect=lambda p: str(p)), \
                     patch.object(Path, "read_text", side_effect=[json.dumps(source), json.dumps(this_baseline)]), \
                     patch.object(rec, "close", side_effect=RuntimeError("cleanup") if mode == "cleanup" else None) as close, \
                     redirect_stdout(io.StringIO()):
                    if mode == "ok":
                        main(argv)
                    else:
                        with self.assertRaises(SystemExit) as error:
                            main(argv)
                        self.assertEqual(error.exception.code, 2)
                data = output.read_bytes()
                result = json.loads(data)
                self.assertEqual("summary" in result, mode in ("ok", "change"))
                self.assertEqual(factory.call_count, int(mode != "provenance"))
                self.assertEqual(close.call_count, int(mode != "provenance"))
                self.assertNotIn("甲", data.decode("utf-8"))
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(argv)
                self.assertEqual(output.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
