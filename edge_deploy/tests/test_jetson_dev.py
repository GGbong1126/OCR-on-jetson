import ast
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from flask import Flask, request, Response

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'jetson'))
import serve_jetson_dev as d
import health_jetson_dev as h
from test_rec_service import body


def original_extract():
    # Compile only the real handler, not the heavy server or its initialization.
    root = Path(__file__).resolve().parents[2]
    # On the board tests are flat in the isolated stage.
    source = root/'server.py'
    if not source.exists(): source = d.RELEASE/'server.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'api_extract')
    node.decorator_list = []
    ns = dict(time=time, request=request, log=lambda *_: None, deal_counter=Mock(),
        pipeline=SimpleNamespace(_ner_pipeline=object(), _run_ner=lambda _: {}),
        _normalize_ocr_payload=lambda x: x,
        _read_markdown_header=lambda _: ({'title':'sample','class':'','pagenumber':'12'}, 'sample body'),
        _field_aliases=lambda *args: [args[0]], _semantic_kind=lambda *_: 'title',
        _convert_top_value=lambda v, _: v,
        _error_response=lambda code,msg: dict(isSuccess=False,status=code,msg=dict(error=msg)),
        _json_response=lambda value: value)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<isolated-original-extract>', 'exec'), ns)
    return ns['api_extract'], ns


class Tests(unittest.TestCase):
    def test_original_extract_business_same_without_quota(self):
        fn, ns = original_extract()
        clone = d.dev_extract(fn)
        app = Flask(__name__)
        payload = dict(ocrResult=dict(pages=[dict(markdown='sample')]),
                       topFields=[dict(name='title',type='String',description='title')], nerFields=[])
        with app.test_request_context(json=payload): expected = fn()
        ns['deal_counter'].assert_called_once_with(1)
        ns['deal_counter'].reset_mock()
        with app.test_request_context(json=payload): actual = clone()
        self.assertEqual(expected, actual)
        self.assertTrue(actual['isSuccess'])
        ns['deal_counter'].assert_not_called()
        self.assertIs(clone.__code__, fn.__code__)
        self.assertIs(fn.__globals__['deal_counter'], ns['deal_counter'])

    def test_original_validation_unchanged(self):
        fn, ns = original_extract(); clone = d.dev_extract(fn); app = Flask(__name__)
        for payload in (None, [], {}, dict(ocrResult={},topFields=[],nerFields=[])):
            with app.test_request_context(json=payload): original = fn()
            with app.test_request_context(json=payload): candidate = clone()
            self.assertEqual(candidate, original)
            self.assertEqual(candidate['status'], '400')
        ns['deal_counter'].assert_not_called()

    def test_policy_is_no_storage_not_unlimited_credits(self):
        with patch('builtins.open', side_effect=AssertionError('storage accessed')):
            self.assertIsNone(d.unmetered_usage(1))
            for x in (0, -1, True, 2, '1'):
                with self.assertRaises(ValueError): d.unmetered_usage(x)

    def test_explicit_start_environment(self):
        with patch.dict(d.os.environ, {}, clear=True):
            with self.assertRaises(ValueError): d.preflight()

    def make_app(self):
        fn, ns = original_extract()
        server = SimpleNamespace(api_extract=fn, api_ocr=lambda: dict(status='200', isSuccess=True),
                                 logger=SimpleNamespace(disabled=False), pipeline=None,
                                 validate_license=Mock(side_effect=AssertionError('license accessed')),
                                 initialize_counter_file=Mock(side_effect=AssertionError('quota initialized')))
        stats = body()['models_execution']
        models = SimpleNamespace(stats=lambda: copy.deepcopy(stats), warmup_execution=copy.deepcopy(stats))
        adapter = SimpleNamespace(healthy=Mock(return_value=True))
        app = d.assemble(server, ns['pipeline'], models, dict(cpu_modules={}), adapter)
        return app, server, adapter

    def test_health_and_dev_marker(self):
        app, server, adapter = self.make_app()
        with app.test_client() as client:
            response = client.get('/healthz')
            self.assertTrue(h.ready(response.get_json()))
            self.assertEqual(response.headers['X-AutoOCR-Mode'], d.MODE)
            self.assertEqual(client.post('/OCR',json={}).status_code,200)
        server.validate_license.assert_not_called()
        server.initialize_counter_file.assert_not_called()

    def test_nonlocal_and_forwarded_header(self):
        app, _, _ = self.make_app()
        with app.test_client() as client:
            self.assertEqual(client.get('/healthz',environ_base={'REMOTE_ADDR':'192.0.2.1'},
                headers={'X-Forwarded-For':'127.0.0.1'}).status_code,403)
            self.assertEqual(client.get('/healthz').status_code,200)

    def test_latched_error_and_partial_success_rejected(self):
        app, _, adapter = self.make_app()
        with app.test_client() as client:
            adapter.healthy.return_value = False
            self.assertEqual(client.post('/OCR',json={}).status_code,503)
            adapter.healthy.side_effect = [True, False]
            self.assertEqual(client.post('/OCR',json={}).status_code,503)

    def test_health_no_licensed_service_misidentified(self):
        b = body()
        self.assertFalse(h.ready(b))
        b.update(service_mode=d.MODE,authorization=dict(d.AUTHORIZATION))
        self.assertTrue(h.ready(b))
        b['authorization']['usage_limit']='enabled'
        self.assertFalse(h.ready(b))

    def test_launch_contract(self):
        directory=Path(d.__file__).parent
        start=(directory/'start_jetson_dev.sh').read_text()
        unit=(directory/'autoocr-jetson-dev.service').read_text()
        self.assertIn('AUTOOCR_ENABLE_DEV_NO_AUTH',start)
        self.assertIn('--bind 127.0.0.1:5051 --workers 1 --worker-class sync --threads 1',start)
        for token in ('--check-license','--activate','--preload','0.0.0.0','initialize_counter'):
            self.assertNotIn(token,start)
        self.assertIn('health_jetson_dev.py --timeout 300',unit)
        self.assertIn('Environment=AUTOOCR_ENABLE_DEV_NO_AUTH=1',unit)


if __name__ == '__main__': unittest.main()
