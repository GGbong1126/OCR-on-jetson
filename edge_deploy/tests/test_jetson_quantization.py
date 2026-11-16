"""CPU-only PTQ split/preprocess/gate tests, not a GPU validation."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import quantize_yolo_ptq as q


class QuantizationTests(unittest.TestCase):
    def rows(self):
        return [{'sha256': s, 'business_pixels_sha256': 'pixels' + s, 'file': s + '.JPG'} for s in ('c', 'a', 'd', 'b')]

    def test_deterministic_disjoint_split(self):
        cal, holdout = q.split_rows(self.rows(), 2)
        self.assertEqual([r['sha256'] for r in cal], ['a', 'b'])
        self.assertEqual([r['sha256'] for r in holdout], ['c', 'd'])
        self.assertEqual((cal, holdout), q.split_rows(list(reversed(self.rows())), 2))

    def test_empty_sets_rejected(self):
        for count in (0, 4, 5):
            with self.assertRaises(ValueError): q.split_rows(self.rows(), count)

    def test_duplicate_source_rejected(self):
        rows = self.rows(); rows[1]['sha256'] = rows[0]['sha256']
        with self.assertRaises(ValueError): q.split_rows(rows, 2)

    def test_pixel_duplicate_rejected(self):
        rows = self.rows(); rows[1]['business_pixels_sha256'] = rows[0]['business_pixels_sha256']
        with self.assertRaises(ValueError): q.split_rows(rows, 2)

    def test_report_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'r.json'; q.save(p, {'x': 1})
            with self.assertRaises(FileExistsError): q.save(p, {'x': 2})
            self.assertIn('1', p.read_text())

    def test_calibration_square_preserves_rgb_channels(self):
        rgb = np.full((400, 200, 3), (255, 0, 128), dtype=np.uint8)
        cal = q.tensor_from_rgb(rgb, True)
        runtime = q.tensor_from_rgb(rgb, False)
        self.assertEqual(cal.shape, (1, 3, 640, 640))
        self.assertEqual(runtime.shape, (1, 3, 640, 320))
        np.testing.assert_allclose(cal[0, :, 320, 320], [1, 0, 128 / 255], atol=1e-6)
        np.testing.assert_allclose(cal[0, :, 320, 0], [114 / 255] * 3, atol=1e-6)
        self.assertEqual(cal.dtype, np.float32)
        self.assertTrue(cal.flags.c_contiguous)

    def test_gate_does_not_strip_spaces_or_fields(self):
        ref = {k: 'a 1' for k in ('FullText', 'Text', 'MarkdownBodyLines', 'NER', 'special')}
        ref.update(RawLines=[{'text': 'a 1'}], Box=[{'score': .9}])
        out = dict(ref, Box=[{'score': .899}])
        self.assertTrue(q.strict_compare(ref, out)['passed'])
        out['FullText'] = 'a1'
        self.assertFalse(q.strict_compare(ref, out)['passed'])
        out = dict(ref, special='changed')
        self.assertFalse(q.strict_compare(ref, out)['passed'])

    def test_field_diagnostics_distinguishes_number_from_content(self):
        ref = {'special': {'page_no': [{'text': '1', 'layout_number': '4'}]},
               'Text': [{'Class': 'page_no', 'Number': ['4'], 'Content': ['1']}],
               'Box': [{'Class': 'page_no'}]}
        out = {'special': {'page_no': [{'text': '1', 'layout_number': '3'}]},
               'Text': [{'Class': 'page_no', 'Number': ['3'], 'Content': ['1']}],
               'Box': [{'Class': 'page_no'}]}
        # Test helper only; the board experiment passes the real server helper.
        field = lambda value, key: ''.join(r['text'] for r in value.get(key, []))
        row = q.field_diagnostics(ref, out, field)
        self.assertTrue(row['api_special_text_equal']['page_no'])
        self.assertTrue(row['text_class_content_equal_without_number'])
        out['special']['page_no'][0]['text'] = '2'
        self.assertFalse(q.field_diagnostics(ref, out, field)['api_special_text_equal']['page_no'])


if __name__ == '__main__': unittest.main()
