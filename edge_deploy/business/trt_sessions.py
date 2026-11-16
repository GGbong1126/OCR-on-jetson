"""Native TensorRT session adapters for the UNCHANGED server OCR algorithm.

Only out-of-profile tensors go to the existing ORT CPU session. No padding,
truncation, precision change, CUDA-error retry or silent partial OCR on failure.
"""
import atexit
import json
from pathlib import Path
import threading

import numpy as np

from release_business import fingerprint, plain

BACKEND = 'tensorrt_fp32_with_ort_fallback'


class TRTSession:
    def __init__(self, cpu, runner, role, classes):
        self.cpu, self.runner, self.role, self.classes = cpu, runner, role, classes
        self.profile = [list(s) for s in runner.engine.get_tensor_profile_shape(runner.input_name, 0)]
        self.input_name = cpu.get_inputs()[0].name
        if self.input_name != runner.input_name or cpu.get_providers() != ['CPUExecutionProvider']:
            raise ValueError('Expected matching input name and an explicit CPU-only fallback')
        self.trt_calls = self.ort_fallback_calls = self.errors = self.warmup_trt_calls = 0
        self.failed = False
        self.last_input_shape = None

    def get_inputs(self):
        return self.cpu.get_inputs()

    def in_profile(self, shape):
        low, _, high = self.profile
        return len(shape) == len(low) and all(lo <= d <= hi for d, lo, hi in zip(shape, low, high))

    def validate_output(self, output, shape):
        if not isinstance(output, np.ndarray) or output.dtype != np.float32 or not np.isfinite(output).all():
            raise RuntimeError('OCR backend returned invalid/nonfinite output')
        if self.role == 'det':
            valid = output.shape == (1, 1, shape[2], shape[3])
        else:
            valid = output.ndim == 3 and output.shape[0] == 1 and output.shape[1] > 0 and output.shape[2] == self.classes
        if not valid:
            raise RuntimeError('OCR backend returned incompatible output shape')

    def run(self, outputs, feeds):
        if self.failed:
            raise RuntimeError('OCR backend is unhealthy; restart or roll back the service')
        try:
            if outputs is not None or set(feeds) != {self.input_name}:
                raise ValueError('Expected one named tensor and all outputs')
            tensor = feeds[self.input_name]
            if (not isinstance(tensor, np.ndarray) or tensor.dtype != np.float32
                    or tensor.ndim != 4 or tensor.shape[:2] != (1, 3) or not np.isfinite(tensor).all()):
                raise ValueError('Expected finite float32 [1,3,H,W]')
            self.last_input_shape = list(tensor.shape)
            if self.in_profile(tensor.shape):
                # Device selection is thread-local. The service is serial; no shared-context concurrency.
                self.runner.cuda.call('cudaSetDevice', 0)
                output = self.runner.run_tensor(tensor)
                self.validate_output(output, tensor.shape)
                self.trt_calls += 1
                # Preserve ORT's owned-result behavior, not the experiment runner's borrowed buffer.
                return [output.copy()]
            output = self.cpu.run(None, feeds)[0]
            self.validate_output(output, tensor.shape)
            self.ort_fallback_calls += 1
            return [output]
        except Exception:
            self.errors += 1
            self.failed = True
            raise

    def warmup(self):
        self.run(None, {self.input_name: np.zeros(self.profile[1], dtype=np.float32)})
        self.warmup_trt_calls = self.trt_calls
        self.trt_calls = 0
        self.last_input_shape = None

    def stats(self):
        return {'profile': self.profile, 'trt_calls': self.trt_calls,
                'ort_fallback_out_of_profile_calls': self.ort_fallback_calls,
                'warmup_trt_calls': self.warmup_trt_calls, 'errors': self.errors,
                'healthy': not self.failed, 'last_input_shape': self.last_input_shape}


class CheckedOCREngine:
    def __init__(self, original, sessions):
        self.original, self.sessions = original, sessions
        self.lock = threading.Lock()
        self.closed = False
        self.successful_predict_calls = 0

    def predict(self, image):
        # Protect the entire native predict, including decoding of returned tensors.
        with self.lock:
            if self.closed or not self.healthy:
                raise RuntimeError('OCR backend is unavailable')
            result = self.original.predict(image)
            # Native predict catches individual rec exceptions; never return partial
            # success when an adapter/CUDA error was swallowed inside that loop.
            if not self.healthy:
                raise RuntimeError('OCR backend failed; partial results are rejected')
            self.successful_predict_calls += 1
            return result

    @property
    def healthy(self):
        return not self.closed and all(not s.failed for s in self.sessions.values())

    def stats(self):
        return {'successful_predict_calls': self.successful_predict_calls,
                **{name: session.stats() for name, session in self.sessions.items()}}

    def close(self):
        with self.lock:
            if self.closed:
                return
            for session in self.sessions.values():
                session.runner.close()
            self.closed = True


def attach(original, source, runner_factory=None):
    """Called after the native loader, before publishing readiness or accepting requests."""
    source = plain(source)
    spec = json.loads((source / 'edge_deploy/business/trt_models.json').read_text(encoding='utf-8'))
    if spec.get('schema_version') != 1 or spec.get('precision') != 'fp32_no_tf32' or set(spec['models']) != {'det', 'rec'}:
        raise ValueError('Unsupported TensorRT asset manifest')
    if runner_factory is None:
        from edge_deploy.tools.trt_backend import TensorRTDetector
        runner_factory = TensorRTDetector
    runners, sessions = [], {}
    try:
        for role, row in spec['models'].items():
            for key in ('onnx', 'engine'):
                path = plain(source / row[key])
                if not path.is_relative_to(source) or fingerprint(path)['sha256'] != row[key + '_sha256']:
                    raise ValueError('TensorRT asset hash/path mismatch: ' + role)
            runner = runner_factory(source / row['engine'])
            runners.append(runner)
            if runner.trt.__version__ != spec['tensorrt_version']:
                raise RuntimeError('TensorRT runtime version differs from the approved engine build')
            session = TRTSession(getattr(original, role + '_session'), runner, role, len(original._char_dict))
            if session.profile != row['profile']:
                raise ValueError('Actual engine profile differs from the approved manifest')
            session.warmup()  # Real CUDA execution + finite/shape check; no OCR/Extract/counts.
            sessions[role] = session
        wrapper = CheckedOCREngine(original, sessions)
        # Change only raw-tensor execution. Original preprocessing, crops, dictionary,
        # DB postprocessing, CTC, thresholds, ordering and result format stay untouched.
        original.det_session, original.rec_session = sessions['det'], sessions['rec']
        atexit.register(wrapper.close)
        return wrapper
    except Exception:
        for runner in reversed(runners):
            runner.close()
        raise
