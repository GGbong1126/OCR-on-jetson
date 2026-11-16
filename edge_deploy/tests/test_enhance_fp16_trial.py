from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import optimize_enhance_fp16 as t


class Tests(unittest.TestCase):
    def test_identical_outputs(self):
        x = np.zeros(t.build.OUTPUT_SHAPE, np.float32)
        r = t.compare_arrays(x, x.copy())
        self.assertEqual(r['mae'], 0)
        self.assertTrue(r['allclose_diagnostic'])
        self.assertEqual(r['raw_tile_global_mask_disagreement'], 0)

    def test_small_error_can_cross_business_threshold(self):
        x = np.full(t.build.OUTPUT_SHAPE, .55, np.float32)
        y = x.copy()
        y[0, 0, 0, 0] = .5499
        r = t.compare_arrays(x, y)
        self.assertTrue(r['allclose_diagnostic'])
        self.assertEqual(r['raw_tile_global_mask_disagreement'], 1)

    def test_invalid_output_rejected(self):
        x = np.zeros(t.build.OUTPUT_SHAPE, np.float32)
        for out in (x.astype(np.float16), x[:, :, :-1], np.full_like(x, np.nan), np.full_like(x, np.inf)):
            with self.assertRaises(ValueError):
                t.compare_arrays(x, out)

    def test_usage_requires_same_real_inputs(self):
        t.verify_usage(1, ['a', 'b'], ['a', 'b'])
        t.verify_usage(3, [], [])
        for values in ((1, [], []), (1, ['a'], ['b']), (1, ['a', 'b'], ['b', 'a']),
                       (3, ['a'], []), (3, [], ['a']), (2, [], [])):
            with self.assertRaises(ValueError):
                t.verify_usage(*values)

    def test_quality_gate_is_complete_strict_and_unique(self):
        rows = [{'image_type': 1, 'image_sha256': str(i), 'passed': True, 'same_original_inputs': True}
                for i in range(13)]
        rows.append({'image_type': 3, 'image_sha256': '0', 'passed': True, 'same_original_inputs': True})
        self.assertTrue(t.quality_ok(rows))
        self.assertFalse(t.quality_ok(rows[:-1]))
        for change in ({'passed': False}, {'same_original_inputs': False}, {'image_sha256': '1'}):
            copy = [dict(r) for r in rows]
            copy[0].update(change)
            self.assertFalse(t.quality_ok(copy))

    def test_private_number_symbol_diagnostic(self):
        r = t.text_diagnostics('甲1-2', '乙1 - 2')
        self.assertTrue(r['number_symbol_sequence_equal'])
        self.assertTrue(r['edit_categories'])
        self.assertFalse(t.text_diagnostics('甲1-2', '甲12')['number_symbol_sequence_equal'])
        self.assertFalse(t.text_diagnostics('甲1-2', '甲1-3')['number_symbol_sequence_equal'])
        self.assertNotIn('reference_text', r)
        self.assertNotIn('candidate_text', r)

    def test_geometry_is_diagnostic_but_text_and_fields_are_strict(self):
        strict = {k: True for k in ('FullText', 'Text', 'MarkdownBodyLines', 'NER', 'raw_line_texts')}
        strict['special'] = False
        fields = {'api_special_text_equal': {k: True for k in ('title', 'category_no', 'page_no')}}
        self.assertTrue(t.business_passed(strict, fields))
        for key in ('FullText', 'Text', 'MarkdownBodyLines', 'NER', 'raw_line_texts'):
            self.assertFalse(t.business_passed(dict(strict, **{key: False}), fields))
        fields['api_special_text_equal']['page_no'] = False
        self.assertFalse(t.business_passed(strict, fields))


if __name__ == '__main__':
    unittest.main()
