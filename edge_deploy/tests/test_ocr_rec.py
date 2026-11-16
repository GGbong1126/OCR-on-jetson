"""CPU checks, including AST replay of server methods without importing the service."""
import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ocr_rec import crop_text_region, preprocess_rec, decode_ctc, recognize_boxes
from compare_ocr import compare_records


class RecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        server = Path(__file__).resolve().parents[2] / "server.py"
        tree = ast.parse(server.read_text(encoding="utf-8-sig"))
        engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_ONNXOCREngine")
        methods = [n for n in engine.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("_crop_text_region", "_rec_run")]
        namespace = {"cv2": cv2, "np": np, "math": math}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(server), "exec"), namespace)
        cls.original_crop = staticmethod(namespace["_crop_text_region"])
        cls.original_rec = staticmethod(namespace["_rec_run"])

    def test_preprocess_and_decode_match_server(self):
        characters = ["<blank>", "A", "B", " "]
        raw = np.eye(4, dtype=np.float32)[[1, 1, 0, 1, 2, 2, 3]][None]
        rng = np.random.default_rng(123)
        for shape in ((100, 1, 3), (48, 32, 3), (25, 123, 3), (2, 200, 3)):
            with self.subTest(shape=shape):
                crop = rng.integers(0, 256, shape, dtype=np.uint8)
                tensor = preprocess_rec(crop)

                def replay(names, inputs):
                    np.testing.assert_array_equal(inputs["x"], tensor)
                    return [raw]

                ref = SimpleNamespace(_char_dict=characters, rec_session=SimpleNamespace(
                    get_inputs=lambda: [SimpleNamespace(name="x")], run=replay))
                self.assertEqual(self.original_rec(ref, crop), decode_ctc(raw, characters))
                self.assertEqual(decode_ctc(raw, characters)[0], "AAB ")
        narrow = preprocess_rec(np.zeros((100, 1, 3), dtype=np.uint8))
        self.assertEqual(narrow.shape, (1, 3, 48, 32))
        self.assertTrue((narrow[:, :, :, 1:] == 1).all())

    def test_crop_matches_server_with_clipping(self):
        image = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
        points = np.array([[-2, -3], [15, -3], [15, 22], [-2, 22]], dtype=np.int32)
        np.testing.assert_array_equal(crop_text_region(image, points), self.original_crop(None, image, points))
        self.assertIsNone(crop_text_region(image, np.zeros((4, 2), dtype=np.int32)))
        with self.assertRaises(ValueError):
            crop_text_region(image, np.full((4, 2), 0.5))

    def test_ctc_empty_and_invalid(self):
        chars = ["<blank>", "A"]
        self.assertEqual(decode_ctc(np.array([[[1., 0.]]], dtype=np.float32), chars), ("", 0.0))
        for raw in (np.zeros((1, 3, 4)), np.full((1, 2, 2), np.nan),
                    np.full((1, 2, 2), np.inf), np.zeros((1, 0, 2))):
            with self.assertRaises(ValueError):
                decode_ctc(raw, chars)

    def test_failed_crop_not_silently_dropped(self):
        rec = SimpleNamespace(predict=lambda crop: ("", 0., {"preprocess_ms": 0., "inference_ms": 0., "decode_ms": 0.}, [1, 3, 48, 32]))
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        records, _ = recognize_boxes(rec, image, [np.zeros((4, 2)), np.array([[1, 1], [4, 1], [4, 4], [1, 4]])])
        self.assertEqual([r["status"] for r in records], ["error", "empty_text"])
        self.assertEqual(compare_records(records, records)["status"], "invalid_recognition_errors")


def record(x, text="A"):
    return {"status": "ok" if text else "empty_text", "box": [[x, 0], [x+10, 0], [x+10, 10], [x, 10]], "text": text}


class TextComparisonTests(unittest.TestCase):
    def test_reordered_regions_match_geometrically(self):
        result = compare_records([record(0, "A"), record(20, "B")], [record(20, "B"), record(0, "A")])
        self.assertTrue(result["all_regions_matched_text_equal"])
        self.assertFalse(result["ordered_texts_equal"])
        self.assertTrue(result["text_multisets_equal"])

    def test_changed_text_and_missing_region(self):
        result = compare_records([record(0), record(20)], [record(0, "B")])
        self.assertEqual(len(result["changed_pairs"]), 1)
        self.assertEqual(result["unmatched_reference_indices"], [1])
        self.assertFalse(result["all_regions_matched_text_equal"])

    def test_ambiguous_matching_and_no_boxes(self):
        result = compare_records([record(0)], [record(0), record(0)])
        self.assertEqual(result["matched_count"], 0)
        self.assertEqual(compare_records([], [record(0)])["status"], "insufficient_text_regions")

    def test_empty_text_not_hidden(self):
        result = compare_records([record(0, "")], [record(0, "")])
        self.assertEqual(result["reference_empty_count"], 1)
        self.assertEqual(result["candidate_empty_count"], 1)
        self.assertEqual(result["status"], "compared_not_accuracy_validated")


if __name__ == "__main__":
    unittest.main()
