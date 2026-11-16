import ast
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import rec_wide_adapter as a
import serve_rec_wide as s


class Fake:
    def __init__(self, profile):
        self.profiles = {'x': profile}
        self.calls = self.errors = 0
        self.closed = False
        self.role = 'rec'
        self.received = []
        self.inputs = [SimpleNamespace(name='x')]
        self.outputs = [SimpleNamespace(name='out')]
        self.dtypes = {0: np.dtype('float32'), 1: np.dtype('float32')}

    def in_profile(self, feeds):
        if set(feeds) != {'x'}:
            raise ValueError('names')
        shape = feeds['x'].shape
        lo, _, hi = self.profiles['x']
        return len(shape) == 4 and all(x <= y <= z for x, y, z in zip(lo, shape, hi))

    def run(self, names, feeds):
        self.calls += 1
        self.received.append(feeds)
        return [feeds['x']]

    def stats(self):
        return dict(calls=self.calls, errors=self.errors, closed=self.closed, profiles=self.profiles)

    def close(self):
        self.closed = True


def outer(route):
    source = Path(a.__file__).with_name('jetson_models.py')
    if not source.exists():
        source = a.RELEASE / 'edge_deploy/jetson/jetson_models.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    classes = [v for v in tree.body if isinstance(v, ast.ClassDef) and v.name in ('CheckedCPU', 'OCRSession')]
    cpu = Fake([[1, 3, 48, 1], [1, 3, 48, 160], [1, 3, 48, 9999]])
    scope = dict(np=np, cpu_session=lambda path: cpu)
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), 'exec'), scope)
    return scope['OCRSession'](route, Path('original.onnx')), cpu


class RouteTests(unittest.TestCase):
    def route(self):
        return a.WideRoute(Fake(a.ORIGINAL_PROFILE), Fake(a.WIDE_PROFILE))

    def test_original_tensor_and_union_cpu_policy(self):
        route = self.route()
        rec, cpu = outer(route)
        for width in (31, 32, 1536, 1537, 1696, 2304, 2305):
            x = np.zeros((1, 3, 48, width), np.float32)
            self.assertIs(rec.run(None, {'x': x})[0], x)
        self.assertEqual((route.base.calls, route.wide.calls, cpu.calls), (2, 3, 2))
        self.assertEqual(rec.stats()['calls'], 5)

    def test_fault_blocks_every_route_even_outside_union(self):
        route = self.route()
        rec, cpu = outer(route)
        route.wide.run = Mock(side_effect=RuntimeError('synthetic fault'))
        for width in (1700, 160, 2305):
            with self.assertRaises(RuntimeError):
                rec.run(None, {'x': np.zeros((1, 3, 48, width), np.float32)})
        self.assertEqual((route.base.calls, cpu.calls), (0, 0))
        self.assertGreater(route.errors, 0)

    def test_bad_names_latched(self):
        route = self.route()
        with self.assertRaises(ValueError):
            route.in_profile({'bad': np.zeros((1, 3, 48, 32))})
        self.assertEqual(route.errors, 1)

    def test_close_is_idempotent_and_outside_cpu_rejected(self):
        route = self.route()
        rec, cpu = outer(route)
        route.close()
        route.close()
        self.assertTrue(route.base.closed and route.wide.closed)
        with self.assertRaises(RuntimeError):
            rec.run(None, {'x': np.zeros((1, 3, 48, 2305), np.float32)})
        self.assertEqual(cpu.calls, 0)

    def test_close_both_on_failure(self):
        route = self.route()
        route.wide.close = Mock(side_effect=RuntimeError('close'))
        with self.assertRaises(RuntimeError):
            route.close()
        self.assertTrue(route.base.closed)

    def test_models_closed_not_ready(self):
        m = SimpleNamespace(healthy=True, stats=lambda: {'rec': dict(closed=True)})
        self.assertFalse(a.healthy(m))

    def test_install_warms_before_publish(self):
        base = Fake(a.ORIGINAL_PROFILE)
        rec, _ = outer(base)
        m = SimpleNamespace(models={'rec': rec}, stats=lambda: {'rec': rec.stats()})
        wide = Fake(a.WIDE_PROFILE)
        with patch.object(a, 'validate_config', return_value=({}, Path('engine'))), \
                patch.object(a, 'digest', return_value=a.SOURCE_SHA), \
                patch.dict(sys.modules, {'trt82_backend': SimpleNamespace(Session=lambda *v: wide)}):
            a.install(m, 'config')
        self.assertEqual(wide.calls, 3)
        self.assertIs(rec.gpu.base, base)
        self.assertEqual([list(v['x'].shape) for v in wide.received], a.WIDE_PROFILE)

    def test_install_failure_keeps_base_and_closes_new(self):
        base = Fake(a.ORIGINAL_PROFILE)
        rec, _ = outer(base)
        m = SimpleNamespace(models={'rec': rec}, stats=lambda: {})
        wide = Fake(a.WIDE_PROFILE)
        wide.run = Mock(side_effect=RuntimeError('warmup'))
        with patch.object(a, 'validate_config', return_value=({}, Path('engine'))), \
                patch.object(a, 'digest', return_value=a.SOURCE_SHA), \
                patch.dict(sys.modules, {'trt82_backend': SimpleNamespace(Session=lambda *v: wide)}):
            with self.assertRaises(RuntimeError):
                a.install(m, 'config')
        self.assertIs(rec.gpu, base)
        self.assertTrue(wide.closed)
        self.assertFalse(base.closed)


class ConfigTests(unittest.TestCase):
    def test_immutable_required_contract(self):
        with patch.object(a, 'digest', return_value='mock'):
            c = a.configuration(Path('.'))
        self.assertEqual(c['profile'], a.WIDE_PROFILE)
        self.assertEqual(c['engine_sha256'], a.ENGINE_SHA)
        self.assertNotIn('enabled', c)

    def test_changed_config_and_engine_rejected(self):
        # Mock file reads only; do not write fake models or configuration files.
        with patch.object(a, 'configuration', return_value={'engine': 'rec_wide_fp32.engine', 'adapters': {'rec_wide_adapter.py': 'adapter'}}), \
                patch.object(Path, 'read_text', return_value='{}'):
            with self.assertRaises(ValueError):
                a.validate_config(Path('config.json'))
        cfg = {'engine': 'rec_wide_fp32.engine', 'adapters': {'rec_wide_adapter.py': 'adapter'}}
        with patch.object(a, 'configuration', return_value=cfg), \
                patch.object(Path, 'read_text', return_value=json.dumps(cfg)), \
                patch.object(a, 'digest', side_effect=['adapter', 'wrong-engine']), \
                patch.object(Path, 'is_symlink', return_value=False):
            with self.assertRaises(ValueError):
                a.validate_config(Path('config.json'))

    def test_unset_is_off_but_empty_not_off(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(s.selected_config())
        with patch.dict(os.environ, {s.CONFIG_ENV: ''}):
            with self.assertRaises(ValueError):
                s.selected_config()


class FactoryTests(unittest.TestCase):
    def test_disabled_delegates_original_factory(self):
        base = SimpleNamespace(create_app=Mock(return_value='original'))
        with patch.object(s, 'selected_config', return_value=None), patch.object(s, 'original_factory', return_value=base):
            self.assertEqual(s.create_app(), 'original')
        base.create_app.assert_called_once_with()

    def test_missing_license_never_loads_models_or_counters(self):
        server = SimpleNamespace(validate_license=Mock(return_value=False), initialize_counter_file=Mock())
        base = SimpleNamespace(deployment=lambda: (server, a.RELEASE, {}))
        with patch.object(s, 'selected_config', return_value=Path('config')), \
                patch.object(s, 'original_factory', return_value=base), patch.object(s, 'load_models') as load:
            with self.assertRaises(RuntimeError):
                s.create_app()
        load.assert_not_called()
        server.initialize_counter_file.assert_not_called()

    def test_offline_never_validates_license_or_counters(self):
        server = SimpleNamespace(validate_license=Mock(), initialize_counter_file=Mock())
        base = SimpleNamespace(deployment=lambda: (server, a.RELEASE, {}))
        with patch.object(s, 'selected_config', return_value=None), \
                patch.object(s, 'original_factory', return_value=base), \
                patch.object(s, 'load_models', return_value=('pipeline', 'models')):
            self.assertEqual(s.load_business()[1:3], ('pipeline', 'models'))
        server.validate_license.assert_not_called()
        server.initialize_counter_file.assert_not_called()

    def test_integrated_guards_and_counter_order(self):
        events = []
        class App:
            config = {}
            extensions = {}
            def before_request(self, fn): self.before = fn; return fn
            def after_request(self, fn): self.after = fn; return fn
            def get(self, path):
                def register(fn): self.health = fn; return fn
                return register
            def make_response(self, value): return value
        app = App()
        server = SimpleNamespace(validate_license=lambda: events.append('license') or True,
                                 initialize_counter_file=lambda: events.append('counter'), app_online=app)
        models = SimpleNamespace(healthy=True, stats=lambda: {'rec': {'closed': False}}, warmup_execution={})
        base = SimpleNamespace(deployment=lambda: (server, a.RELEASE, {'cpu_modules': {}}))
        def load(*args): events.append('models'); return 'pipeline', models
        with patch.object(s, 'selected_config', return_value=Path('config')), \
                patch.object(s, 'original_factory', return_value=base), patch.object(s, 'load_models', side_effect=load):
            self.assertIs(s.create_app(), app)
        self.assertEqual(events, ['license', 'models', 'counter'])
        self.assertIsNone(app.before())
        self.assertEqual(app.health()[1], 200)
        models.healthy = False
        self.assertEqual(app.before()[1], 503)
        self.assertEqual(app.after('partial')[1], 503)
        self.assertEqual(app.health()[1], 503)


if __name__ == '__main__':
    unittest.main()
