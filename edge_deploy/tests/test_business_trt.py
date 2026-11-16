"""CPU-only/mocked adapters: do not load real TRT, server, licenses or sockets."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'edge_deploy/business'))
import trt_sessions as bridge
import service_health
import release_business as release


class AdapterTests(unittest.TestCase):
    def session(self, role='rec'):
        cpu = Mock()
        cpu.get_inputs.return_value = [SimpleNamespace(name='x')]
        cpu.get_providers.return_value = ['CPUExecutionProvider']
        runner = Mock(input_name='x')
        runner.engine.get_tensor_profile_shape.return_value = [[1, 3, 48, 32], [1, 3, 48, 160], [1, 3, 48, 1536]]
        runner.run_tensor.return_value = np.ones((1, 4, 5), dtype=np.float32)
        cpu.run.return_value = [np.ones((1, 4, 5), dtype=np.float32)]
        return bridge.TRTSession(cpu, runner, role, 5)

    def test_inside_profile_gpu_owned_output_and_counters(self):
        s = self.session()
        x = np.zeros((1, 3, 48, 138), np.float32)
        result = s.run(None, {'x': x})[0]
        self.assertIs(s.runner.run_tensor.call_args.args[0], x)
        self.assertFalse(np.shares_memory(result, s.runner.run_tensor.return_value))
        s.cpu.run.assert_not_called()
        self.assertEqual(s.stats()['trt_calls'], 1)

    def test_out_of_profile_is_explicit_cpu_without_transform(self):
        for width in (31, 1537, 3200):
            s = self.session()
            x = np.ones((1, 3, 48, width), np.float32)
            s.run(None, {'x': x})
            self.assertIs(s.cpu.run.call_args.args[1]['x'], x)
            s.runner.run_tensor.assert_not_called()
            self.assertEqual(s.stats()['ort_fallback_out_of_profile_calls'], 1)

    def test_profile_bounds_inclusive(self):
        s = self.session()
        self.assertTrue(s.in_profile((1, 3, 48, 32)))
        self.assertTrue(s.in_profile((1, 3, 48, 1536)))
        self.assertFalse(s.in_profile((1, 3, 49, 160)))

    def test_gpu_error_latched_no_silent_cpu_retry(self):
        s = self.session()
        s.runner.run_tensor.side_effect = RuntimeError('synthetic CUDA failure')
        with self.assertRaises(RuntimeError):
            s.run(None, {'x': np.zeros((1, 3, 48, 160), np.float32)})
        self.assertTrue(s.failed)
        s.cpu.run.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'unhealthy'):
            s.run(None, {'x': np.zeros((1, 3, 48, 2000), np.float32)})

    def test_nonfinite_and_wrong_outputs_fail(self):
        for output in (np.full((1, 4, 5), np.nan, np.float32), np.ones((1, 4, 6), np.float32),
                       np.ones((1, 4, 5), np.float16)):
            s = self.session(); s.runner.run_tensor.return_value = output
            with self.assertRaises(RuntimeError):
                s.run(None, {'x': np.zeros((1, 3, 48, 160), np.float32)})
            self.assertTrue(s.failed)

    def test_bad_feed_fails(self):
        s = self.session()
        with self.assertRaises(ValueError):
            s.run(None, {'x': np.full((1, 3, 48, 160), np.inf, np.float32)})
        s.runner.run_tensor.assert_not_called()

    def test_warmup_is_not_counted_as_business_inference(self):
        s = self.session(); s.warmup()
        self.assertEqual(s.warmup_trt_calls, 1)
        self.assertEqual(s.trt_calls, 0)
        self.assertIsNone(s.last_input_shape)

    def test_original_swallowed_rec_exception_cannot_return_partial_success(self):
        s = self.session()
        original = Mock()
        def partial(image):
            s.failed = True
            return ['partial']
        original.predict.side_effect = partial
        wrapper = bridge.CheckedOCREngine(original, {'rec': s})
        with self.assertRaisesRegex(RuntimeError, 'partial'):
            wrapper.predict(object())
        self.assertEqual(wrapper.successful_predict_calls, 0)

    def test_original_results_preserved_and_close_idempotent(self):
        s = self.session(); original = Mock()
        wrapper = bridge.CheckedOCREngine(original, {'rec': s})
        image = object()
        self.assertIs(wrapper.predict(image), original.predict.return_value)
        original.predict.assert_called_once_with(image)
        wrapper.close(); wrapper.close()
        s.runner.close.assert_called_once()
        self.assertFalse(wrapper.healthy)

    def test_health_does_not_accept_wrong_backend(self):
        body = dict(status='ready', scope='full_business', ocr_backend=bridge.BACKEND,
                    **{key: True for key in service_health.MODELS})
        self.assertFalse(service_health.is_ready(200, body, bridge.BACKEND))
        body['ocr_execution'] = {role: dict(healthy=True, warmup_trt_calls=1, errors=0) for role in ('det', 'rec')}
        self.assertFalse(service_health.is_ready(200, body))
        self.assertTrue(service_health.is_ready(200, body, bridge.BACKEND))
        self.assertFalse(service_health.is_ready(503, body, bridge.BACKEND))

    def test_detector_shape_guard(self):
        s = self.session('det')
        s.runner.run_tensor.return_value = np.zeros((1, 1, 48, 160), np.float32)
        s.run(None, {'x': np.zeros((1, 3, 48, 160), np.float32)})
        s.runner.run_tensor.return_value = np.zeros((1, 1, 48, 159), np.float32)
        with self.assertRaises(RuntimeError):
            s.run(None, {'x': np.zeros((1, 3, 48, 160), np.float32)})


class TRTPackagingTests(unittest.TestCase):
    def test_gpu_release_copies_only_approved_assets_and_checks_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; engines = root / 'engines'; engines.mkdir()
            for name in (*release.SOURCE_FILES, 'pth/imgpre/test.pth', *release.TRT_FILES):
                target = engines / Path(name).name if name in release.TRT_ENGINES else source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(('fixture:' + name).encode())
            spec = json.loads((ROOT / release.TRT_CONFIG).read_text(encoding='utf-8'))
            for row in spec['models'].values():
                row['onnx_sha256'] = release.fingerprint(source / row['onnx'])['sha256']
                row['engine_sha256'] = release.fingerprint(engines / Path(row['engine']).name)['sha256']
            (source / release.TRT_CONFIG).write_text(json.dumps(spec), encoding='utf-8')
            env = root / 'env.json'
            env.write_text(json.dumps(dict(schema_version=1, platform='Linux', machine='x86_64',
                                          python='3.10.12', packages={'tensorrt': '10.7.0'})))
            (engines / 'failed_fp16.engine').write_bytes(b'do not ship')
            output = root / 'output'
            release.build(source, output, 'fixture-trt', env, engines)
            self.assertEqual(json.loads((output / release.MANIFEST).read_text())['kind'], 'trt_business_source_models')
            self.assertEqual(set(p.name for p in (output / 'pth/tensorrt').iterdir()),
                             {Path(p).name for p in release.TRT_ENGINES})
            (engines / Path(release.TRT_ENGINES[0]).name).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash'):
                release.build(source, root / 'bad', 'bad', env, engines)
            self.assertFalse((root / 'bad').exists())


if __name__ == '__main__':
    unittest.main()
