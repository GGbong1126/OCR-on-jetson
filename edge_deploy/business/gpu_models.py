"""Explicit CUDA placement for non-OCR models, FP32, bounded memory, fail closed.

No license access. CPU shape/control ORT nodes are allowed and reported; a CUDA
provider that silently fails to load is not. OCR fallback sessions stay CPU.
"""
from collections import Counter
import json
from pathlib import Path
import tempfile

import numpy as np


class CheckedCall:
    def __init__(self, target, role, invoke, validate):
        self.target, self.role = target, role
        self.invoke, self.validate = invoke, validate
        self.calls = self.warmup_calls = self.errors = 0

    def __getattr__(self, name):
        return getattr(self.target, name)

    def __call__(self, *args, **kwargs):
        if self.errors:
            raise RuntimeError(self.role + ' GPU is unhealthy')
        try:
            result = self.invoke(*args, **kwargs)
            self.validate(result)
            self.calls += 1
            return result
        except Exception:
            self.errors += 1
            raise

    def run(self, *args, **kwargs):
        return self(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self(*args, **kwargs)

    def stats(self):
        return {'backend': 'onnxruntime_cuda' if self.role in ('layout', 'cell') else 'pytorch_cuda',
                'device': 'cuda:0', 'precision': 'fp32_no_tf32', 'calls': self.calls,
                'warmup_calls': self.warmup_calls, 'errors': self.errors}


def finite_numpy(outputs):
    if not all(np.isfinite(x).all() for x in outputs):
        raise RuntimeError('Non-finite GPU output')


class GPUModels:
    def __init__(self, cfg):
        # Import torch first to load matching CUDA12/cuDNN9 shared libraries.
        import torch
        import onnxruntime as ort
        if not torch.cuda.is_available() or 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('GPU PyTorch and ORT CUDA provider are both required')
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.torch, self.ort = torch, ort
        self.roles = {str(Path(cfg.ONNX_LAYOUT_MODEL).resolve()): 'layout',
                      str(Path(cfg.ONNX_CELL_MODEL).resolve()): 'cell'}
        self.cpu_paths = {str(Path(cfg.ONNX_OCR_DET_MODEL).resolve()),
                          str(Path(cfg.ONNX_OCR_REC_MODEL).resolve())}
        self.models, self.node_counts = {}, {}

    def session_factory(self, path):
        path = str(Path(path).resolve())
        options = self.ort.SessionOptions()
        options.intra_op_num_threads = 2
        if path in self.cpu_paths:
            return self.ort.InferenceSession(path, sess_options=options, providers=['CPUExecutionProvider'])
        role = self.roles[path]  # Reject unclassified model paths.
        options.enable_profiling = True
        # Only synthetic warmup is profiled; the private trace is removed after aggregation.
        with tempfile.TemporaryDirectory(prefix='autoocr-ort-') as tmp:
            options.profile_file_prefix = str(Path(tmp) / role)
            session = self.ort.InferenceSession(path, sess_options=options, providers=[
                ('CUDAExecutionProvider', {'device_id': 0, 'use_tf32': 0,
                  'gpu_mem_limit': 768 * 1024**2, 'arena_extend_strategy': 'kSameAsRequested',
                  'cudnn_conv_algo_search': 'HEURISTIC', 'cudnn_conv_use_max_workspace': 0}),
                'CPUExecutionProvider'])
            if session.get_providers()[0] != 'CUDAExecutionProvider':
                raise RuntimeError(role + ' did not initialize CUDA')
            session.disable_fallback()
            checked = CheckedCall(session, role, session.run, finite_numpy)
            size = 800 if role == 'layout' else 640
            feeds = {}
            for item in session.get_inputs():
                if 'image' in item.name:
                    feeds[item.name] = np.zeros((1, 3, size, size), np.float32)
                elif 'im_shape' in item.name:
                    feeds[item.name] = np.array([[size, size]], np.float32)
                elif 'scale_factor' in item.name:
                    feeds[item.name] = np.ones((1, 2), np.float32)
                else:
                    raise RuntimeError('Unexpected ONNX input: ' + item.name)
            checked.run(None, feeds)
            trace = json.loads(Path(session.end_profiling()).read_text())
            counts = Counter(row.get('args', {}).get('provider') for row in trace
                             if row.get('cat') == 'Node' and row.get('args', {}).get('provider'))
            if not counts['CUDAExecutionProvider']:
                raise RuntimeError(role + ' has no profiled CUDA kernels')
            self.node_counts[role] = dict(counts)
        checked.warmup_calls, checked.calls = checked.calls, 0
        self.models[role] = checked
        return checked

    def check_torch_model(self, model):
        params = list(model.parameters())
        if not params or any(p.device.type != 'cuda' or p.device.index != 0
                             or (p.is_floating_point() and p.dtype != self.torch.float32) for p in params):
            raise RuntimeError('Expected model parameters on CUDA:0 in FP32')

    def attach(self, pipeline):
        torch = self.torch
        pipeline._yolo_model.to('cuda:0')
        self.check_torch_model(pipeline._yolo_model.model)
        self.check_torch_model(pipeline._ner_pipeline.model)
        self.check_torch_model(pipeline._imgpre_model)

        def tensor_ok(tensor):
            if tensor.device.type != 'cuda' or not bool(torch.isfinite(tensor).all()):
                raise RuntimeError('Expected finite CUDA output')

        def yolo_ok(result):
            for item in result:
                tensor_ok(item.boxes.data)

        def ner_ok(result):
            # HF pipeline returns CPU-side records after GPU model execution.
            finite_numpy([np.asarray([float(row['score']) for row in result])])

        yolo = CheckedCall(pipeline._yolo_model, 'yolo', pipeline._yolo_model.predict, yolo_ok)
        ner = CheckedCall(pipeline._ner_pipeline, 'ner', pipeline._ner_pipeline, ner_ok)
        enhance = CheckedCall(pipeline._imgpre_model, 'image_enhancement', pipeline._imgpre_model, tensor_ok)
        self.models.update(yolo=yolo, ner=ner, image_enhancement=enhance)
        pipeline._yolo_model, pipeline._ner_pipeline, pipeline._imgpre_model = yolo, ner, enhance
        # Real forward passes without user data, HTTP, activation, or licensed extraction.
        yolo.predict(np.zeros((640, 640, 3), np.uint8), device='cuda:0', half=False, verbose=False)
        ner('测试')
        with torch.inference_mode():
            enhance(torch.zeros((1, 3, 512, 512), device='cuda', dtype=torch.float32))
        for role in ('yolo', 'ner', 'image_enhancement'):
            row = self.models[role]
            row.warmup_calls, row.calls = row.calls, 0
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        original = pipeline.process

        def checked_process(*args, **kwargs):
            if not self.healthy:
                raise RuntimeError('GPU models unhealthy')
            result = original(*args, **kwargs)
            if not self.healthy:
                raise RuntimeError('GPU error: refusing partial business result')
            return result
        pipeline.process = checked_process

    @property
    def healthy(self):
        return len(self.models) == 5 and all(not row.errors for row in self.models.values())

    def stats(self):
        return {role: dict(row.stats(), **({'warmup_node_providers': self.node_counts[role]}
                     if role in self.node_counts else {})) for role, row in self.models.items()}
