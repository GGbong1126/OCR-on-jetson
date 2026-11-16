"""CPU-only checks of comparison and input profile rejection."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_trt import compare_outputs
from trt_backend import TensorRTDetector


class ComparisonTests(unittest.TestCase):
    def test_identical_maps_and_threshold_crossing(self):
        a = np.zeros((1, 1, 32, 32), dtype=np.float32)
        result = compare_outputs(a, a.copy(), (32, 32, 3))
        self.assertTrue(result["ordered_boxes_exactly_equal"])
        self.assertEqual(result["probability_max_abs_error"], 0)
        b = a.copy()
        b[0, 0, 0, 0] = 0.2
        result = compare_outputs(a, b, (32, 32, 3))
        self.assertAlmostEqual(result["mask_disagreement_fraction_at_0_1"], 1 / 1024)
        self.assertFalse(result["allclose_rtol_1e_3_atol_1e_4"])

    def test_reject_bad_output(self):
        a = np.zeros((1, 1, 32, 32), dtype=np.float32)
        with self.assertRaises(ValueError):
            compare_outputs(a, a[:, :, :16], (32, 32, 3))
        result = compare_outputs(a, np.full_like(a, np.nan), (32, 32, 3))
        self.assertEqual(result["status"], "invalid_nonfinite_output")
        self.assertTrue(result["ort_output"]["all_finite"])
        self.assertEqual(result["trt_output"]["nan_count"], 1024)
        self.assertIsNone(result["trt_output"]["finite_min"])
        self.assertNotIn("probability_mae", result)
        b = a.copy()
        b[0, 0, 0, 0] = np.inf
        b[0, 0, 0, 1] = -np.inf
        result = compare_outputs(a, b, (32, 32, 3))
        self.assertEqual(result["trt_output"]["positive_inf_count"], 1)
        self.assertEqual(result["trt_output"]["negative_inf_count"], 1)

    def test_reject_shape_before_cuda_allocation(self):
        detector = TensorRTDetector.__new__(TensorRTDetector)
        detector.current_shape = None
        detector.input_name = "x"
        shape = (1, 3, 960, 704)
        detector.engine = SimpleNamespace(get_tensor_profile_shape=lambda *args: (shape, shape, shape))
        with self.assertRaisesRegex(ValueError, "outside engine profile"):
            detector._prepare((1, 3, 736, 960))


if __name__ == "__main__":
    unittest.main()
