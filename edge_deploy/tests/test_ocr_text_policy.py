"""CPU-only tests: field scope, exact symbols, provenance and parallel gates."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import hashlib
import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ocr_text_policy import page_number_equal, validate_annotations
from compare_rec_buckets import compare_rows, main, SUCCESS, CHANGED, POLICY_SUCCESS


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.row = {"index": 55, "box": [[0, 0], [32, 0], [32, 48], [0, 48]],
                    "input_shape": [1, 3, 48, 32], "text": "- 1 -"}
        self.provenance = dict(image_sha256="image", source_report_sha256="source",
                               source_branch="reference", crop_set_sha256="crops")
        self.annotation = dict(self.provenance, schema_version=1, fields=[
            {k: self.row[k] for k in ("index", "box", "input_shape")} | {"role": "page_number"}])

    def test_only_ascii_spaces_tolerated(self):
        for candidate in ("-1-", " -1 - ", "- 1 -"):
            self.assertTrue(page_number_equal("- 1 -", candidate))
        for candidate in ("-2-", "1--", "1", "−1−", "-１-", "-1-abc", "-\t1-", "-\u00a01-", ""):
            self.assertFalse(page_number_equal("- 1 -", candidate), repr(candidate))
        for text in ("", " ", "---", "page 1", "-\t1-", "-\u00a01-"):
            self.assertFalse(page_number_equal(text, text))

    def test_valid_bound_manual_annotation(self):
        self.assertEqual(validate_annotations(self.annotation, self.provenance, [self.row]),
                         {55: "page_number"})

    def test_annotation_rejects_wrong_source_or_geometry(self):
        for key in self.provenance:
            bad = deepcopy(self.annotation)
            bad[key] = "wrong"
            with self.assertRaises(ValueError):
                validate_annotations(bad, self.provenance, [self.row])
        for key, value in (("index", 54), ("index", True), ("role", "any_text"),
                           ("box", []), ("input_shape", [1, 3, 48, 64])):
            bad = deepcopy(self.annotation)
            bad["fields"][0][key] = value
            with self.assertRaises(ValueError):
                validate_annotations(bad, self.provenance, [self.row])

    def test_annotation_rejects_duplicate_empty_and_nonpage_source(self):
        for fields in ([], self.annotation["fields"] * 2):
            bad = dict(self.annotation, fields=fields)
            with self.assertRaises(ValueError):
                validate_annotations(bad, self.provenance, [self.row])
        for text in ("", "姓名 1", "--"):
            with self.assertRaises(ValueError):
                validate_annotations(self.annotation, self.provenance, [dict(self.row, text=text)])

    def compare(self, outputs, roles, source=None):
        rows = [dict(self.row, text=source or self.row["text"])]
        plan = [{"index": 55, "original_input_shape": [1, 3, 48, 32],
                 "padded_input_shape": [1, 3, 48, 64]}]
        def decode(rec, tensor, result):
            text = next(texts)
            result.update(empty_text=text == "", text_length=len(text), all_finite=True)
            return text
        texts = iter(outputs)
        report = {}
        with patch("compare_rec_buckets.decode_call", side_effect=decode), redirect_stdout(io.StringIO()):
            compare_rows(None, np.zeros((48, 32, 3), np.uint8), rows, plan, report, roles)
        return report

    def test_policy_success_keeps_raw_failure_and_edit_distance(self):
        report = self.compare(["- 1 -", "-1-"], {55: "page_number"})
        self.assertEqual(report["status"], POLICY_SUCCESS)
        self.assertEqual(report["strict_status"], CHANGED)
        self.assertEqual(report["summary"]["padded_original_changed_indices"], [55])
        self.assertEqual(report["summary"]["tolerated_page_number_space_change_indices"], [55])
        self.assertEqual(report["summary"]["policy_changed_indices"], [])
        self.assertEqual(report["records"][0]["original_to_padded_edit_count"], 2)

    def test_unannotated_region_stays_strict(self):
        report = self.compare(["- 1 -", "-1-"], {})
        self.assertEqual(report["status"], CHANGED)
        self.assertEqual(report["summary"]["tolerated_page_number_space_change_indices"], [])

    def test_digit_symbol_change_fails_even_with_role(self):
        for text in ("-2-", "1", "--1", "-1-abc"):
            report = self.compare(["- 1 -", text], {55: "page_number"})
            self.assertEqual(report["status"], CHANGED)
            self.assertFalse(report["summary"]["policy_consistency_passed"])

    def test_reference_drift_never_hidden_by_policy(self):
        report = self.compare(["-1-", "-1-"], {55: "page_number"})
        self.assertEqual(report["status"], "completed_with_reference_drift_not_accepted")
        self.assertFalse(report["summary"]["policy_consistency_passed"])

    def test_identical_retains_strict_success(self):
        report = self.compare(["- 1 -", "- 1 -"], {55: "page_number"})
        self.assertEqual(report["status"], SUCCESS)
        self.assertTrue(report["summary"]["strict_consistency_passed"])

    def test_main_annotation_success_invalid_binding_and_cleanup(self):
        crop_set = [{k: self.row[k] for k in ("index", "box", "input_shape")}]
        annotation = deepcopy(self.annotation)
        annotation["crop_set_sha256"] = hashlib.sha256(json.dumps(crop_set, sort_keys=True).encode()).hexdigest()
        for mode in ("ok", "wrong_image", "cleanup"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "result.json"
                this_annotation = deepcopy(annotation)
                if mode == "wrong_image":
                    this_annotation["image_sha256"] = "wrong"
                argv = ["--engine", "engine", "--onnx", "onnx", "--image", "image", "--dict", "dict",
                        "--source-report", "source", "--baseline-report", "baseline", "--buckets", "64",
                        "--field-annotations", "annotation", "--output", str(output)]
                texts = iter(["- 1 -", "-1-"])
                def decode(rec, tensor, result):
                    text = next(texts)
                    result.update(empty_text=False, text_length=len(text), all_finite=True)
                    return text
                with patch("compare_rec_buckets.TensorRTRecognizer") as factory, \
                     patch("compare_rec_buckets.prepare_rows", return_value=[self.row]), \
                     patch("compare_rec_buckets.verify_baseline"), \
                     patch("compare_rec_buckets.read_image", return_value=np.zeros((48, 32, 3), np.uint8)), \
                     patch("compare_rec_buckets.sha256_file", side_effect=str), \
                     patch.object(Path, "read_text", side_effect=["{}", "{}", json.dumps(this_annotation)]), \
                     patch("compare_rec_buckets.decode_call", side_effect=decode), redirect_stdout(io.StringIO()):
                    rec = factory.return_value
                    rec.profile = [[1, 3, 48, 32], [1, 3, 48, 64], [1, 3, 48, 128]]
                    rec.metadata.return_value = {"backend": "fake_test_only"}
                    if mode == "cleanup":
                        rec.close.side_effect = RuntimeError("cleanup failed")
                    if mode == "ok":
                        main(argv)
                    else:
                        with self.assertRaises(SystemExit) as error:
                            main(argv)
                        self.assertEqual(error.exception.code, 2)
                    self.assertEqual(factory.call_count, int(mode != "wrong_image"))
                report = json.loads(output.read_bytes())
                if mode == "ok":
                    self.assertEqual(report["status"], POLICY_SUCCESS)
                    self.assertEqual(report["strict_status"], CHANGED)
                    self.assertEqual(report["field_annotations"], this_annotation)
                else:
                    self.assertEqual(report["status"], "invalid_comparison")
                    self.assertNotIn("summary", report)
                    self.assertNotIn("strict_status", report)


if __name__ == "__main__":
    unittest.main()
