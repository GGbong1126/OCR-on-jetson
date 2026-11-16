from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import profile_rec_runtime as m


class FakeContext:
    def __init__(self):
        self.calls, self.ok = [], True

    def set_binding_shape(self, index, shape):
        self.calls.append((index, shape))
        return self.ok

    def execute_v2(self, bindings):
        return True


class Tests(unittest.TestCase):
    def test_reference_never_skips(self):
        obj = FakeContext()
        proxy = m.ContextProxy(obj)
        proxy.row = m.new_row(0)
        for _ in range(2):
            proxy.set_binding_shape(0, [1, 3, 48, 160])
        self.assertEqual(len(obj.calls), 2)
        self.assertEqual(proxy.row['shape_changes'], 1)

    def test_only_successful_identical_shape_skipped(self):
        obj = FakeContext()
        proxy = m.ContextProxy(obj, True)
        for width in (160, 160, 32, 160):
            self.assertTrue(proxy.set_binding_shape(0, [1, 3, 48, width]))
        self.assertEqual(len(obj.calls), 3)
        obj.ok = False
        for _ in range(2):
            self.assertFalse(proxy.set_binding_shape(0, [1, 3, 48, 90]))
        self.assertEqual(len(obj.calls), 5)

    def test_shapes_scoped_by_binding(self):
        obj = FakeContext()
        proxy = m.ContextProxy(obj, True)
        proxy.set_binding_shape(0, (1, 2))
        proxy.set_binding_shape(1, (1, 2))
        self.assertEqual(len(obj.calls), 2)

    def test_pattern_original_order_and_extremes(self):
        feeds = [{'x': np.zeros((1, 3, 48, w), np.float32)} for w in (160, 45, 900, 80)]
        result = m.patterns(feeds)
        self.assertEqual(result['original_sequence'], [0, 1, 2, 3])
        self.assertEqual(result['alternate_ab'], [1, 2])
        self.assertEqual(result['repeat_a'], [1, 1])
        with self.assertRaises(ValueError):
            m.patterns(feeds[:1])

    def test_cache_simulation_is_bounded(self):
        shapes = [(1, 2), (1, 3), (1, 2), (1, 4), (1, 3)]
        row = m.describe_sequence(shapes, 2)
        self.assertEqual(row['cold_sequence_hits'], 1)
        self.assertEqual(row['adjacent_changes'], 4)
        self.assertEqual(row['unique_shapes'], 3)


if __name__ == '__main__':
    unittest.main()
