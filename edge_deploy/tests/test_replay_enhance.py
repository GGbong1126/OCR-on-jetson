from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import replay_enhance_fp16 as t


class Tests(unittest.TestCase):
    def test_completed_failure_required(self):
        with self.assertRaises(ValueError):
            t.select_case({'status': 'running'})

    def test_first_numeric_failure_selected_not_other_changes(self):
        rows = [{'image_type': 1, 'passed': False, 'full_text_diagnostics': {'number_symbol_sequence_equal': True}},
                {'image_type': 1, 'passed': False, 'full_text_diagnostics': {'number_symbol_sequence_equal': False}, 'index': 2},
                {'image_type': 1, 'passed': False, 'full_text_diagnostics': {'number_symbol_sequence_equal': False}, 'index': 3}]
        self.assertEqual(t.select_case({'status': 'differences_not_accepted', 'rows': rows})['index'], 2)


if __name__ == '__main__':
    unittest.main()
