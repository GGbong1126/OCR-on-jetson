"""CPU-only tests; no TensorRT SDK or GPU build is exercised here."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_mixed import add_fixed_profile, apply_constraints, select_prefix


def entry(name, inputs=(), kind="CONVOLUTION", floating=True, shape=False):
    return {"name": name, "type": kind, "inputs": list(inputs),
            "outputs": [{"name": name, "floating": floating, "shape_tensor": shape}]}


class MixedBuildTests(unittest.TestCase):
    def test_prefix_includes_boundary_and_upstream_out_of_order(self):
        layers = [entry("unrelated"), entry("boundary", ["upstream"]),
                  entry("tail", ["boundary"]), entry("upstream", ["input"])]
        self.assertEqual(select_prefix(layers, "boundary"), {0, 1, 3})

    def test_missing_or_ambiguous_boundary_fails(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            select_prefix([entry("x")], "missing")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_prefix([entry("x"), entry("x")], "x")

    def test_constraints_do_not_change_shape_constants_or_tail(self):
        layers = [entry("conv"), entry("concat", kind="CONCATENATION"),
                  entry("shape", shape=True), entry("weights", kind="CONSTANT"),
                  entry("tail")]
        calls = []
        native = [SimpleNamespace(set_output_type=lambda j, dtype, i=i: calls.append((i, j, dtype)))
                  for i in range(len(layers))]
        network = SimpleNamespace(get_layer=lambda i: native[i])
        apply_constraints(network, layers, {0, 1, 2, 3}, SimpleNamespace(float32="fp32"))
        self.assertEqual(native[0].precision, "fp32")
        self.assertFalse(hasattr(native[1], "precision"))
        self.assertFalse(hasattr(native[4], "precision"))
        self.assertEqual(calls, [(0, 0, "fp32"), (1, 0, "fp32")])

    def test_unreviewed_floating_layer_fails(self):
        network = SimpleNamespace(get_layer=lambda i: None)
        with self.assertRaisesRegex(ValueError, "Unreviewed"):
            apply_constraints(network, [entry("plugin", kind="PLUGIN_V3")], {0},
                              SimpleNamespace(float32="fp32"))


class FixedProfileTests(unittest.TestCase):
    def setUp(self):
        self.shape = (1, 3, 960, 704)
        self.profile = MagicMock()
        self.profile.set_shape.return_value = None  # TRT 10.7 Python contract.
        self.profile.__bool__.return_value = True
        self.profile.get_shape.return_value = [self.shape] * 3
        self.builder = SimpleNamespace(create_optimization_profile=lambda: self.profile)
        self.config = MagicMock()
        self.config.add_optimization_profile.return_value = 0

    def run_profile(self):
        return add_fixed_profile(self.builder, self.config, "x", self.shape)

    def test_none_return_and_zero_index_are_success(self):
        result = self.run_profile()
        self.assertEqual(result, {"index": 0, "input_name": "x",
                                  "min_opt_max": [list(self.shape)] * 3})
        self.profile.set_shape.assert_called_once_with("x", self.shape, self.shape, self.shape)
        self.config.add_optimization_profile.assert_called_once_with(self.profile)

    def test_setter_exception_is_not_suppressed(self):
        self.profile.set_shape.side_effect = RuntimeError("invalid dimensions")
        with self.assertRaisesRegex(RuntimeError, "invalid dimensions"):
            self.run_profile()
        self.config.add_optimization_profile.assert_not_called()

    def test_invalid_profile_is_rejected(self):
        self.profile.__bool__.return_value = False
        with self.assertRaisesRegex(ValueError, "Invalid input profile"):
            self.run_profile()
        self.config.add_optimization_profile.assert_not_called()

    def test_missing_or_wrong_readback_is_rejected(self):
        for actual in ([], [(1, 3, 32, 32)] * 3):
            with self.subTest(actual=actual):
                self.profile.get_shape.return_value = actual
                with self.assertRaisesRegex(ValueError, "readback mismatch"):
                    self.run_profile()
        self.config.add_optimization_profile.assert_not_called()

    def test_negative_index_is_rejected(self):
        self.config.add_optimization_profile.return_value = -1
        with self.assertRaisesRegex(ValueError, "Could not add input profile"):
            self.run_profile()


if __name__ == "__main__":
    unittest.main()
