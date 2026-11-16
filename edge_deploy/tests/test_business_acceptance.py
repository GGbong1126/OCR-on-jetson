"""No live HTTP, OCR models or licensed count changes."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'business'))
import acceptance_business as accept


class BusinessAcceptanceTests(unittest.TestCase):
    def health(self):
        return (200, dict(status='ready', scope='full_business', ocr_backend='onnxruntime_cpu',
                         **{key: True for key in ('cell', 'layout', 'yolo', 'ner', 'image_enhancement')}))

    def ocr(self):
        return (200, {'isSuccess': True, 'status': '200', 'msg': {'pages': [{'width': 30, 'height': 40,
            'markdown': 'private-title\n\n\nprivate-body', 'blocks': [
                {'text': 'private-body', 'x1': 1, 'y1': 2, 'x2': 20, 'y2': 30}]}]}})

    def fields(self):
        return (200, {'isSuccess': True, 'status': '200', 'msg': {
            'title': 'private-title', 'class': '', 'pagenumber': '', 'NER': {'姓名': 'private-name'}}})

    def prepared(self, count=1):
        return [({'index': i, 'image_file': f'{i}.jpg', 'image_sha256': 'mock', 'image_bytes': 1,
                  'status': 'not_attempted'}, bytes([i])) for i in range(1, count+1)]

    def test_field_stats_do_not_store_values_or_extra_keys(self):
        fields = self.fields()[1]['msg']
        fields['unexpected-private-key'] = 'private-value'
        stats = accept.field_statistics(fields)
        self.assertTrue(stats['top.title']['nonempty'])
        self.assertFalse(stats['top.class']['nonempty'])
        self.assertFalse(stats['NER.性别']['present'])
        self.assertNotIn('private', json.dumps(stats))

    def test_prepare_hash_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.jpg'; a.write_bytes(b'abc')
            prepared=accept.prepare_images([a])
            self.assertEqual(prepared[0][0]['image_sha256'], hashlib.sha256(b'abc').hexdigest())
            with self.assertRaises(ValueError): accept.prepare_images([a, a])
            with self.assertRaises(ValueError): accept.prepare_images([])
            with self.assertRaises(ValueError): accept.prepare_images([a]*6)

    def test_success_saves_no_private_text_and_forwards_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'; ocr=self.ocr()
            with patch.object(accept, 'call', side_effect=[self.health(), ocr, self.fields(), self.health()]) as call, redirect_stdout(io.StringIO()) as stdout:
                report=accept.run('http://127.0.0.1:5051', self.prepared(), '1', output, True)
            self.assertEqual(report['completed_count'], 1)
            self.assertEqual(report['extract_attempts'], 1)
            self.assertEqual(report['status'], 'completed_pending_manual_review')
            self.assertIs(call.call_args_list[2].args[2]['ocrResult'], ocr[1]['msg'])
            saved=(output/'summary.json').read_text(encoding='utf-8')
            review=(output/'manual_review.md').read_text(encoding='utf-8')
            for value in (saved, review, stdout.getvalue()): self.assertNotIn('private', value)
            self.assertNotIn('accuracy_passed', saved)

    def test_ocr_business_failure_stops_without_extract_or_retry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(accept, 'call', side_effect=[self.health(),
                (200, {'isSuccess': False, 'status': '408', 'msg': 'private-error'})]) as call, redirect_stdout(io.StringIO()):
            report=accept.run('http://127.0.0.1:5051', self.prepared(3), '1', Path(tmp)/'run', True)
        self.assertEqual(report['status'], 'stopped_on_error_no_retry')
        self.assertEqual(report['ocr_attempts'], 1)
        self.assertEqual(report['extract_attempts'], 0)
        self.assertEqual(report['records'][0]['ocr']['application_status'], '408')
        self.assertEqual(report['records'][1]['status'], 'not_attempted')
        self.assertEqual(call.call_count, 2)
        self.assertNotIn('private-error', str(report))

    def test_uncertain_extract_attempt_is_recorded_and_never_retried(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(accept, 'call', side_effect=[self.health(),
                self.ocr(), TimeoutError('private-error')]) as call, redirect_stdout(io.StringIO()):
            report=accept.run('http://127.0.0.1:5051', self.prepared(2), '1', Path(tmp)/'run', True)
        self.assertEqual(report['extract_attempts'], 1)
        self.assertEqual(report['records'][0]['failed_stage'], 'extract_in_flight')
        self.assertEqual(report['records'][1]['status'], 'not_attempted')
        self.assertEqual(call.call_count, 3)
        self.assertNotIn('private-error', str(report))

    def test_existing_output_rejected_before_any_http(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(accept, 'call') as call:
            with self.assertRaises(FileExistsError):
                accept.run('http://127.0.0.1:5051', self.prepared(), '1', Path(tmp), True)
            call.assert_not_called()

    def test_without_extract_sends_only_ocr_and_health(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(accept, 'call', side_effect=[self.health(),
                self.ocr(), self.health()]) as call, redirect_stdout(io.StringIO()):
            report=accept.run('http://127.0.0.1:5051', self.prepared(), '3', Path(tmp)/'run')
        self.assertEqual(report['extract_attempts'], 0)
        self.assertEqual(call.call_count, 3)

    def test_unhealthy_start_creates_failure_report_but_no_business_calls(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(accept, 'call', return_value=(200, {'status': 'ready'})) as call:
            report=accept.run('http://127.0.0.1:5051', self.prepared(), '1', Path(tmp)/'run', True)
        self.assertEqual(report['status'], 'stopped_on_error_no_retry')
        self.assertEqual(report['ocr_attempts'], 0)
        self.assertEqual(call.call_count, 1)


if __name__ == '__main__':
    unittest.main()
