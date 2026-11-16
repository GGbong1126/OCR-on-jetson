import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import health_rec_wide as h


def native(calls=1, profile=None):
    return dict(backend='tensorrt82_cuda', precision='fp32_no_tf32', calls=calls,
                errors=0, closed=False, profiles=profile or {})


def body(mode='on'):
    rows = {k: native() for k in h.ROLES}
    rows['rec']['profiles'] = copy.deepcopy(h.ORIGINAL)
    for k in ('det', 'rec'):
        rows[k]['cpu_fallback'] = dict(calls=0, errors=0)
    if mode == 'on':
        rows['rec'].update(backend='tensorrt82_fp32_wide_overlay', calls=4,
            original_execution=native(1, copy.deepcopy(h.ORIGINAL)),
            wide_execution=native(3, copy.deepcopy(h.WIDE)), route_errors=0, wide_profiles=copy.deepcopy(h.WIDE))
    return dict(status='ready', scope='full_business', deployment='jetson-r2', cpu_modules={},
        ocr_backend='tensorrt82_fp32_wide_with_ort_fallback' if mode == 'on' else 'tensorrt82_fp32_with_ort_fallback',
        layout=True, cell=True, yolo=True, ner=True, image_enhancement=True,
        out_of_profile_policy='onnxruntime_cpu_original_tensor_no_resize',
        models_execution=copy.deepcopy(rows), warmup_execution=copy.deepcopy(rows))


class Tests(unittest.TestCase):
    def test_modes_and_original_checker_regression(self):
        self.assertTrue(h.ready(body()))
        self.assertTrue(h.ready(body('off'), 'off'))
        self.assertFalse(h.ready(body(), 'off'))
        self.assertFalse(h.ready(body('off'), 'on'))
        self.assertFalse(h.ready(body(), 'auto'))
        import health_jetson
        self.assertFalse(health_jetson.ready(body()))
        self.assertTrue(health_jetson.ready(body('off')))

    def test_malformed_never_ready(self):
        for value in (None, [], True, 'ready', {}):
            self.assertFalse(h.ready(value))
        for key in ('models_execution', 'warmup_execution'):
            for value in (None, [], 'bad', {}):
                x = body(); x[key] = value
                self.assertFalse(h.ready(x))

    def test_all_roles_closed_errors_warmups(self):
        for location in ('models_execution', 'warmup_execution'):
            for role in h.ROLES:
                for key, value in [('closed', True), ('closed', None), ('errors', 1),
                                   ('errors', False), ('calls', 0), ('calls', True), ('precision', 'fp16')]:
                    x = body(); x[location][role][key] = value
                    with self.subTest(location=location, role=role, key=key, value=value):
                        self.assertFalse(h.ready(x))

    def test_nested_faults_and_profiles(self):
        for location in ('models_execution', 'warmup_execution'):
            for sub in ('original_execution', 'wide_execution'):
                for key, value in [('closed', True), ('errors', 1), ('calls', 0),
                                   ('profiles', h.ORIGINAL if sub == 'wide_execution' else h.WIDE)]:
                    x = body(); x[location]['rec'][sub][key] = value
                    self.assertFalse(h.ready(x))
            for key, value in [('route_errors', 1), ('wide_profiles', {}), ('calls', 999), ('profiles', {})]:
                x = body(); x[location]['rec'][key] = value
                self.assertFalse(h.ready(x))

    def test_cpu_fallback_success_allowed_error_rejected(self):
        for mode in ('on', 'off'):
            x = body(mode)
            x['models_execution']['rec']['cpu_fallback']['calls'] = 1
            self.assertTrue(h.ready(x, mode))
            x['models_execution']['rec']['cpu_fallback']['errors'] = 1
            self.assertFalse(h.ready(x, mode))

    def test_wrong_scope_and_missing_models(self):
        for key, val in [('status', 'error'), ('scope', 'ocr_only'), ('deployment', 'jetson-r1'),
                         ('layout', 1), ('ner', False), ('cpu_modules', {'cell': 'cpu'}),
                         ('out_of_profile_policy', 'resize')]:
            x = body(); x[key] = val
            self.assertFalse(h.ready(x))
        x = body(); del x['models_execution']['ner']
        self.assertFalse(h.ready(x))

    def test_counters_must_not_reset(self):
        x = body('off'); x['warmup_execution']['ner']['calls'] = 5
        self.assertFalse(h.ready(x, 'off'))
        x = body(); x['warmup_execution']['rec']['cpu_fallback']['calls'] = 2
        self.assertFalse(h.ready(x))
        x = body(); x['warmup_execution']['rec']['wide_execution']['calls'] = 4
        x['warmup_execution']['rec']['calls'] = 5
        self.assertFalse(h.ready(x))

    def test_http_poll_no_network(self):
        response = Mock(status=200)
        response.read.return_value = json.dumps(body()).encode()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock(); opener.open.return_value = response
        h.wait_ready(1, 'on', opener)
        self.assertEqual(opener.open.call_args.args[0], h.URL)
        response.read.assert_called_once_with(65537)
        self.assertIsNone(h.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://example.invalid'))

    def test_poll_timeout_and_validation(self):
        for timeout, mode in ((0, 'on'), (True, 'on'), (1, 'auto')):
            with self.assertRaises(ValueError): h.wait_ready(timeout, mode, Mock())
        opener = Mock(); opener.open.side_effect = OSError('no listener')
        with patch.object(h.time, 'monotonic', side_effect=[0, 0, 0, 2, 2]), patch.object(h.time, 'sleep'):
            with self.assertRaises(RuntimeError): h.wait_ready(1, 'on', opener)

    def test_poll_rejects_bad_status_oversize_and_body(self):
        for status, raw in ((503, json.dumps(body()).encode()), (200, b'x'*65537),
                            (200, b'not JSON'), (200, b'[]')):
            response = Mock(status=status)
            response.read.return_value = raw
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            opener = Mock(); opener.open.return_value = response
            with patch.object(h.time, 'monotonic', side_effect=[0, 0, 0, 2, 2]), patch.object(h.time, 'sleep'):
                with self.assertRaises(RuntimeError): h.wait_ready(1, 'on', opener)

    def test_launcher_and_unit_contract(self):
        directory = Path(h.__file__).parent
        script = (directory/'start_rec_wide.sh').read_text()
        unit = (directory/'autoocr-jetson-wide.service').read_text()
        self.assertLess(script.index('runtime/license.dat'), script.index('serve_jetson.py'))
        self.assertLess(script.index('--check-license'), script.index('exec "$root/envs'))
        self.assertIn('--config /dev/null', script)
        self.assertIn('--bind 127.0.0.1:5051 --workers 1 --worker-class sync --threads 1', script)
        self.assertIn('serve_rec_wide:create_app()', script)
        self.assertNotIn('--preload', script)
        self.assertNotIn('--activate', script)
        self.assertNotIn('systemctl', script)
        self.assertIn('RestartPreventExitStatus=78', unit)
        self.assertIn('health_rec_wide.py --mode on --timeout 300', unit)
        self.assertNotIn('Conflicts=', unit)

    def test_delivery_reuse_never_overwrites(self):
        import validate_rec_service as v
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); stage = root/'stage'; stage.mkdir()
            package = root/'package'
            for name in v.DELIVERY: (stage/name).write_bytes(name.encode())
            self.assertFalse(v.stage_delivery(stage, package))
            self.assertTrue(v.stage_delivery(stage, package))
            (package/v.DELIVERY[0]).write_bytes(b'changed')
            with self.assertRaises(ValueError): v.stage_delivery(stage, package)
            self.assertEqual((package/v.DELIVERY[0]).read_bytes(), b'changed')
            (package/v.DELIVERY[0]).write_bytes(v.DELIVERY[0].encode())
            (package/'extra').write_bytes(b'x')
            with self.assertRaises(ValueError): v.stage_delivery(stage, package)

    def test_privilege_limited_to_static_unit_verify(self):
        import validate_rec_service as v
        checks = v.syntax_commands()
        self.assertEqual(checks[0], ['/bin/bash', '-n', str(v.PACKAGE/'start_rec_wide.sh')])
        self.assertEqual(checks[1], ['sudo', '-n', 'systemd-analyze', '--generators=no',
                                   'verify', str(v.PACKAGE/'autoocr-jetson-wide.service')])


if __name__ == '__main__':
    unittest.main()
