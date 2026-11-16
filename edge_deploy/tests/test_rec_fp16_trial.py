from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import optimize_rec_fp16 as mod


def ctc(ids, classes=5):
    value = np.zeros((1, len(ids), classes), np.float32)
    for i, k in enumerate(ids):
        value[0, i, k] = 1
    return value


class Tests(unittest.TestCase):
    def test_blank_and_repeat_semantics(self):
        self.assertEqual(mod.tokens(ctc([0, 1, 1, 0, 1, 2, 2, 0])), [1, 1, 2])

    def test_text_consistent_not_numeric_equal(self):
        a = ctc([1, 0, 2])
        b = a * 0.5
        row = mod.compare(a, b)
        self.assertTrue(row['token_sequence_equal'])
        self.assertFalse(row['allclose_diagnostic'])

    def test_text_drift_rejected(self):
        self.assertFalse(mod.compare(ctc([1, 0, 2]), ctc([1, 0, 3]))['token_sequence_equal'])

    def test_invalid_outputs_rejected(self):
        for b in (np.full((1, 3, 5), np.nan), np.full((1, 3, 5), np.inf), ctc([1, 2])):
            with self.assertRaises(ValueError):
                mod.compare(ctc([1, 0, 2]), b)

    def test_empty_text_is_recorded(self):
        row = mod.compare(ctc([0, 0]), ctc([0, 0]))
        self.assertTrue(row['reference_empty'])
        self.assertTrue(row['candidate_empty'])

    def test_invalid_timings_rejected(self):
        for data in ([], [float('nan')], [-1]):
            with self.assertRaises(ValueError):
                mod.summary(data)


if __name__ == '__main__':
    unittest.main()
