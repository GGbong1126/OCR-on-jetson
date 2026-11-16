"""HTTP fixtures only; no model, private document or licensed request."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'business'))
import acceptance_business as accept
import compare_business as compare


def ocr():
    return {'isSuccess': True, 'status': '200', 'msg': {'pages': [
        {'width': 100, 'height': 100, 'markdown': 'private-header\n\n\nprivate-body',
         'blocks': [{'text': 'private-body', 'x1': 1, 'y1': 2, 'x2': 20, 'y2': 30},
                    {'text': 'private-more', 'x1': 1, 'y1': 40, 'x2': 20, 'y2': 60}]}]}}


def fields():
    return {'isSuccess': True, 'status': '200', 'msg': {
        'title': 'private-header', 'class': '', 'pagenumber': '1', 'NER': {'姓名': 'private-name'}}}


class BusinessCompareTests(unittest.TestCase):
    def test_same_port_aliases_rejected(self):
        for other in ('http://localhost:5051', 'http://[::1]:5051', 'http://127.0.0.1:5051/'):
            with self.assertRaises(ValueError): compare.endpoints('http://127.0.0.1:5051', other)

    def test_identical(self):
        result = compare.compare_ocr(ocr(), ocr())
        self.assertEqual(result['matched_text_equal_count'], 2)
        self.assertTrue(result['markdown_exactly_equal'])
        self.assertNotIn('private', json.dumps(result))

    def test_reordered_blocks_match_without_order_claim(self):
        b = ocr()
        b['msg']['pages'][0]['blocks'].reverse()
        result = compare.compare_ocr(ocr(), b)
        self.assertFalse(result['ordered_texts_equal'])
        self.assertTrue(result['text_multisets_equal'])
        self.assertEqual(result['matched_text_equal_count'], 2)

    def test_text_and_geometry_changes_are_separate(self):
        b = ocr()
        b['msg']['pages'][0]['blocks'][0]['text'] = 'private-different'
        b['msg']['pages'][0]['blocks'][1]['x2'] += 1
        result = compare.compare_ocr(ocr(), b)
        self.assertFalse(result['pairs'][0]['text_equal'])
        self.assertTrue(result['pairs'][0]['coordinates_equal'])
        self.assertTrue(result['pairs'][1]['text_equal'])
        self.assertFalse(result['pairs'][1]['coordinates_equal'])

    def test_ambiguous_geometry_is_unmatched(self):
        b = ocr()
        b['msg']['pages'][0]['blocks'].append(deepcopy(b['msg']['pages'][0]['blocks'][0]))
        result = compare.compare_ocr(ocr(), b)
        self.assertEqual(result['unmatched_reference_indices'], [0])
        self.assertEqual(result['unmatched_candidate_indices'], [0, 2])

    def test_dimensions_mismatch_disables_geometric_matching(self):
        b = ocr()
        b['msg']['pages'][0]['width'] += 1
        result = compare.compare_ocr(ocr(), b)
        self.assertFalse(result['dimensions_equal'])
        self.assertEqual(result['matched_count'], 0)

    def test_missing_empty_and_spaces_not_normalized(self):
        a, b = fields(), fields()
        b['msg']['NER']['性别'] = ''
        b['msg']['pagenumber'] = ' 1'
        result = compare.compare_fields(a, b)
        self.assertFalse(result['all_requested_fields_equal'])
        self.assertFalse(result['fields']['NER.性别']['presence_equal'])
        self.assertFalse(result['fields']['top.pagenumber']['value_equal'])
        self.assertNotIn('private', json.dumps(result))

    def prepared(self):
        return [({'index': 1, 'image_file': 'a.jpg', 'image_sha256': 'mock', 'image_bytes': 1,
                  'status': 'not_attempted'}, b'a')]

    def test_run_same_bytes_and_own_ocr_extract_no_private_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref, cand = ocr(), ocr()
            with patch.object(compare, 'probe'), patch.object(accept, 'call', side_effect=[
                    (200, ref), (200, fields()), (200, cand), (200, fields())]) as call, redirect_stdout(io.StringIO()) as out:
                report = compare.run('http://127.0.0.1:5052', 'http://127.0.0.1:5051',
                                     self.prepared(), '1', Path(tmp)/'run', True)
            self.assertEqual(report['completed_count'], 1)
            self.assertEqual(report['extract_attempts'], 2)
            self.assertEqual(call.call_args_list[0].args[2], call.call_args_list[2].args[2])
            self.assertIs(call.call_args_list[1].args[2]['ocrResult'], ref['msg'])
            self.assertIs(call.call_args_list[3].args[2]['ocrResult'], cand['msg'])
            for text in (out.getvalue(), (Path(tmp)/'run/summary.json').read_text(encoding='utf-8')):
                self.assertNotIn('private-body', text)
                self.assertNotIn('private-name', text)
                self.assertNotIn('private-header', text)

    def test_timeout_stops_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(compare, 'probe'), patch.object(accept, 'call',
                side_effect=[(200, ocr()), TimeoutError('private-failure')]) as call, redirect_stdout(io.StringIO()):
            report = compare.run('http://127.0.0.1:5052', 'http://127.0.0.1:5051', self.prepared(), '1', Path(tmp)/'run', True)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(report['extract_attempts'], 1)
        self.assertEqual(report['records'][0]['failed_stage'], 'reference_extract_in_flight')
        self.assertNotIn('private-failure', json.dumps(report))

    def test_existing_output_sends_no_calls(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(compare, 'probe') as probe:
            with self.assertRaises(FileExistsError):
                compare.run('http://127.0.0.1:5052', 'http://127.0.0.1:5051', self.prepared(), '1', Path(tmp))
            probe.assert_not_called()

    def test_probe_uses_only_invalid_empty_requests(self):
        with patch.object(compare, 'call', return_value=(200, {'isSuccess': False, 'status': '400'})) as call:
            compare.probe('http://127.0.0.1:5052')
        self.assertEqual([c.args[1:] for c in call.call_args_list], [('/OCR', {}), ('/Extract', {})])


if __name__ == '__main__':
    unittest.main()
