from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import profile_model_routes as m


def case(w):
    return {'feeds': {'x': np.zeros((1, 3, 48, w), np.float32)}, 'outputs_sha256': []}


class Tests(unittest.TestCase):
    def test_choose_distinct_shapes(self):
        values = [case(w) for w in (256, 32, 128, 32, 1536)]
        self.assertEqual(m.choose_cases(values), [1, 0, 4])

    def test_single_shape_samples_and_small_counts(self):
        self.assertEqual(m.choose_cases([case(32)]), [0])
        self.assertEqual(m.choose_cases([case(32) for _ in range(9)]), [0, 4, 8])
        self.assertEqual(m.choose_cases([case(32), case(160)]), [0, 1])
        with self.assertRaises(ValueError):
            m.choose_cases([])

    def test_signature_includes_names_dtype_all_inputs(self):
        a = {'x': np.zeros((1, 2), np.int64), 'mask': np.ones((1, 2), np.int64)}
        b = dict(reversed(list(a.items())))
        self.assertEqual(m.signature(a), m.signature(b))
        b['x'] = b['x'].astype(np.int32)
        self.assertNotEqual(m.signature(a), m.signature(b))

    def test_layer_aggregation(self):
        row = m.layer_summary([('conv+relu', 2), ('head', 6), ('conv+relu', 2), ('head', 6)], 2)
        self.assertEqual(row['summed_layer_ms'], 8)
        self.assertEqual(row['layers'][0]['name'], 'head')
        self.assertEqual(row['layers'][0]['fraction_of_layer_sum'], .75)

    def test_bad_callbacks(self):
        for calls in ([], [('x', -1)], [('x', float('nan'))], [('x', float('inf'))], [('x', 0)]):
            with self.assertRaises(ValueError):
                m.layer_summary(calls, 1)

    def test_exact_output_guard_and_private_metadata(self):
        value = np.ones((1, 2), np.float32)
        c = case(32)
        c['outputs_sha256'] = [m.digest_tensor(value)]
        m.verify([value], c)
        with self.assertRaises(RuntimeError):
            m.verify([value + 1], c)
        self.assertEqual(set(m.describe(c)['inputs']['x']), {'shape', 'dtype', 'sha256'})


if __name__ == '__main__':
    unittest.main()
