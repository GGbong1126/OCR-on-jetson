"""Service configuration/startup checks; no live sockets, models or system changes."""
import configparser
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

BUSINESS = Path(__file__).resolve().parents[1] / 'business'
sys.path.insert(0, str(BUSINESS))
import service_health as health


class BusinessServiceTests(unittest.TestCase):
    def ready(self):
        return dict(status='ready', scope='full_business', ocr_backend='onnxruntime_cpu',
                    **{name: True for name in health.MODELS})

    def test_all_markers_required(self):
        self.assertTrue(health.is_ready(200, self.ready()))
        self.assertFalse(health.is_ready(503, self.ready()))
        self.assertFalse(health.is_ready(200, []))
        for key in self.ready():
            body = self.ready()
            body.pop(key)
            self.assertFalse(health.is_ready(200, body))
        body = self.ready(); body['ner'] = 1
        self.assertFalse(health.is_ready(200, body))

    def test_port_check_does_not_listen(self):
        with patch.object(health.socket, 'socket') as factory:
            sock = factory.return_value.__enter__.return_value
            health.check_port()
            sock.bind.assert_called_once_with(('127.0.0.1', 5051))
            sock.listen.assert_not_called()
            sock.bind.side_effect = OSError('occupied')
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                health.check_port()

    def test_ready_wait_uses_health_only_and_no_proxy(self):
        with patch.object(health, 'build_opener') as build, patch.object(health, 'ProxyHandler') as proxy:
            response = build.return_value.open.return_value.__enter__.return_value
            response.status = 200
            response.read.return_value = json.dumps(self.ready()).encode()
            health.wait_ready(5)
            proxy.assert_called_once_with({})
            self.assertEqual(build.return_value.open.call_args.args, (health.URL,))
            self.assertLessEqual(build.return_value.open.call_args.kwargs['timeout'], 3)

    def test_deadline_rejects_invalid_or_unreachable_health(self):
        for data in (b'[]', b'invalid json', b'x'*4097, OSError('offline')):
            with patch.object(health, 'build_opener') as build, patch.object(health.time, 'monotonic',
                    side_effect=[0, 0, 0, 2, 2]):
                if isinstance(data, Exception):
                    build.return_value.open.side_effect = data
                else:
                    response = build.return_value.open.return_value.__enter__.return_value
                    response.status = 200
                    response.read.return_value = data
                with self.assertRaisesRegex(RuntimeError, 'deadline'):
                    health.wait_ready(1)

    def test_nonpositive_timeout_rejected(self):
        with self.assertRaises(ValueError):
            health.wait_ready(0)

    def test_unit_keeps_identity_paths_and_bounded_restart(self):
        text = (BUSINESS/'autoocr-business.service').read_text(encoding='utf-8')
        unit = configparser.ConfigParser(interpolation=None, strict=False)
        unit.read_string(text)
        service = unit['Service']
        self.assertEqual(service['User'], 'q1220')
        self.assertIn('Environment=HOME=/home/q1220', text)
        self.assertIn('Environment=AUTOOCR_RUNTIME_DIR=/home/q1220/projects/autoocr-business/runtime', text)
        self.assertEqual(service['Type'], 'exec')
        self.assertIn('--check-port', service['ExecStartPre'])
        self.assertIn('--wait-ready --timeout 180', service['ExecStartPost'])
        self.assertEqual(service['Restart'], 'on-failure')
        self.assertEqual(unit['Unit']['StartLimitBurst'], '3')
        self.assertEqual(service['KillMode'], 'mixed')
        self.assertGreater(int(service['TimeoutStopSec']), 60)
        self.assertNotIn('ExecReload', service)
        self.assertNotIn('--daemon', text)


if __name__ == '__main__':
    unittest.main()
