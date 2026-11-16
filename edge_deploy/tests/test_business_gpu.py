"""GPU deployment guards tested without a GPU, license or real model."""
from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'business'))
from gpu_models import CheckedCall, GPUModels, finite_numpy
import service_health as health
import release_business as release
import serve_business as serve


class GPUChecks(unittest.TestCase):
    def test_success_retains_output_and_counts(self):
        out = [np.zeros(3)]
        target = Mock(); target.run.return_value = out
        wrap = CheckedCall(target, 'layout', target.run, finite_numpy)
        self.assertIs(wrap.run(None, {'image': 1}), out)
        self.assertEqual(wrap.calls, 1)
        self.assertEqual(wrap.stats()['backend'], 'onnxruntime_cuda')

    def test_failure_latches_and_never_retries(self):
        for output in (np.array([np.nan]), np.array([np.inf])):
            target = Mock(return_value=[output])
            wrap = CheckedCall(target, 'cell', target, finite_numpy)
            with self.assertRaises(RuntimeError): wrap(None, {})
            with self.assertRaisesRegex(RuntimeError, 'unhealthy'): wrap(None, {})
            self.assertEqual(target.call_count, 1)
            self.assertEqual(wrap.errors, 1)
            self.assertEqual(wrap.calls, 0)

    def test_execution_exception_latches(self):
        wrap = CheckedCall(None, 'ner', Mock(side_effect=RuntimeError('test')), lambda x: None)
        with self.assertRaises(RuntimeError): wrap('synthetic')
        self.assertEqual(wrap.errors, 1)

    def test_all_five_roles_required(self):
        group = GPUModels.__new__(GPUModels)
        group.models = {}
        self.assertFalse(group.healthy)
        group.models = {k: SimpleNamespace(errors=0) for k in health.MODELS}
        self.assertTrue(group.healthy)
        group.models['yolo'].errors = 1
        self.assertFalse(group.healthy)

    def body(self):
        b = dict(status='ready', scope='full_business', ocr_backend='onnxruntime_cpu',
                 other_models_backend='cuda', **dict.fromkeys(health.MODELS, True))
        b['models_execution'] = {name: dict(device='cuda:0', errors=0, warmup_calls=1,
                     warmup_node_providers={'CUDAExecutionProvider': 1}) for name in health.MODELS}
        return b

    def test_readiness_requires_execution_not_availability(self):
        b = self.body()
        self.assertTrue(health.is_ready(200, b, other_backend='cuda'))
        for name in health.MODELS:
            b = self.body(); b['models_execution'][name]['warmup_calls'] = 0
            self.assertFalse(health.is_ready(200, b, other_backend='cuda'))
        b = self.body(); b['models_execution']['cell']['warmup_node_providers'] = {}
        self.assertFalse(health.is_ready(200, b, other_backend='cuda'))

    def test_cpu_fallback_session_explicit(self):
        group = GPUModels.__new__(GPUModels)
        group.ort = Mock(); path = str(Path('ocr_rec.onnx').resolve())
        group.cpu_paths = {path}; group.roles = {}
        group.session_factory(path)
        self.assertEqual(group.ort.InferenceSession.call_args.kwargs['providers'], ['CPUExecutionProvider'])

    def test_gpu_environment_excludes_cpu_packages(self):
        env = {'packages': {'torch': '2.8.0+cu126', 'torchvision': '0.23.0+cu126', 'onnxruntime-gpu': '1.19.2'}}
        release.check_gpu_environment(env)
        env['packages']['onnxruntime'] = '1.19.2'
        with self.assertRaises(ValueError): release.check_gpu_environment(env)
        del env['packages']['onnxruntime']
        env['packages']['torch'] = '2.8.0+cpu'
        with self.assertRaises(ValueError): release.check_gpu_environment(env)

    def test_factory_gpu_settings_and_swallowed_error_http_guard(self):
        from flask import Flask
        app = Flask('gpu_guard_test')
        gpu = Mock(healthy=True); gpu.stats.return_value = {}
        trt = Mock(healthy=True); trt.stats.return_value = {}
        @app.get('/synthetic-optional-stage-error')
        def simulate_swallowed_error():
            gpu.healthy = False
            return {'status': '200', 'partial': True}
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'fixture'; model.write_bytes(b'test')
            cfg = SimpleNamespace(**{k: str(model) for k in (
                'ONNX_LAYOUT_MODEL', 'ONNX_CELL_MODEL', 'ONNX_OCR_DET_MODEL', 'ONNX_OCR_REC_MODEL',
                'LAYOUT_MODEL_DIR', 'TABLE_CELL_MODEL_DIR', 'OCR_MODEL_DIR', 'NER_MODEL_DIR', 'IMGPRE_MODEL_DIR')},
                YOLO_CONF=0.3)
            pipeline = Mock(_use_onnx_layout=True, _use_onnx_cell=True, _use_onnx_ocr=True)
            server = SimpleNamespace(cfg=cfg, validate_license=Mock(return_value=True),
                     OCRPipeline=Mock(return_value=pipeline), cv2=Mock(), torch=Mock(),
                     initialize_counter_file=Mock(), app_online=app)
            server.OCRPipeline.FORCE_PADDLE = False
            with patch.dict(os.environ, {'AUTOOCR_OCR_BACKEND': 'trt_fp32',
                          'AUTOOCR_OTHER_BACKEND': 'cuda', 'AUTOOCR_SOURCE_ROOT': tmp}), \
                 patch.object(serve, 'load_server', return_value=server), \
                 patch('gpu_models.GPUModels', return_value=gpu), \
                 patch('trt_sessions.attach', return_value=trt):
                serve.create_app()
            server.OCRPipeline.assert_called_once_with(use_gpu=True,
                 onnx_session_factory=gpu.session_factory, imgpre_fp16=False, imgpre_batch=1)
            client = app.test_client()
            self.assertEqual(client.get('/healthz').json['other_models_backend'], 'cuda')
            response = client.get('/synthetic-optional-stage-error')
            self.assertEqual(response.status_code, 503)
            self.assertNotIn('partial', response.json)
            self.assertEqual(client.get('/healthz').status_code, 503)


if __name__ == '__main__':
    unittest.main()
