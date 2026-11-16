from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import diagnose_rec_mixed as d


class Tests(unittest.TestCase):
    def test_only_complete_rejections(self):
        for status, count in [('running', 26), ('differences_not_accepted', 25),
                              ('all_26_consistent_not_gt', 26)]:
            with self.assertRaises(ValueError):
                d.failure_cases({'status': status, 'rows': [{'passed': False}] * count})
        with self.assertRaises(ValueError):
            d.failure_cases({'status': 'differences_not_accepted', 'rows': [{'passed': True}] * 26})

    def test_returns_only_failed_cases(self):
        rows = [{'passed': True}] * 25 + [{'passed': False, 'marker': 'failure'}]
        self.assertEqual(d.failure_cases({'status': 'differences_not_accepted', 'rows': rows}), rows[-1:])

    def test_edit_summary_has_counts_not_text(self):
        result = d.private_edit_summary('甲1-2', '甲12')
        self.assertEqual(result[0]['reference_categories']['punctuation_or_symbol'], 1)
        self.assertNotIn('reference_text', result[0])
        self.assertNotIn('candidate_text', result[0])


if __name__ == '__main__':
    unittest.main()
