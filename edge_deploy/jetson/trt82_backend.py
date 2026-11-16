"""Synchronous, single-profile TRT8.2 session with owned NumPy outputs.

No pycuda/torch CUDA dependency. Preserve original tensors; do not pad or resize.
Only an explicit out-of-profile policy may fall back, never an execution error.
"""
import ctypes as ct
import hashlib
from pathlib import Path
import platform
import threading
from types import SimpleNamespace

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Cuda:
    def __init__(self):
        self.lib = ct.CDLL('libcudart.so.10.2')
        signatures = {'cudaSetDevice': [ct.c_int],
                      'cudaMalloc': [ct.POINTER(ct.c_void_p), ct.c_size_t],
                      'cudaFree': [ct.c_void_p], 'cudaDeviceSynchronize': [],
                      'cudaRuntimeGetVersion': [ct.POINTER(ct.c_int)],
                      'cudaMemcpy': [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int]}
        for name, types in signatures.items():
            getattr(self.lib, name).argtypes = types
            getattr(self.lib, name).restype = ct.c_int
        self.lib.cudaGetErrorString.argtypes = [ct.c_int]
        self.lib.cudaGetErrorString.restype = ct.c_char_p
        version = ct.c_int()
        self.call('cudaRuntimeGetVersion', ct.byref(version))
        if version.value != 10020:
            raise RuntimeError('Expected CUDA10.2 runtime')
        self.call('cudaSetDevice', 0)

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            raise RuntimeError('{}: {} {}'.format(name, code, self.lib.cudaGetErrorString(code)))


class Session:
    def __init__(self, engine_path, expected_sha256, role, plugins=()):
        import tensorrt as trt
        if platform.machine() != 'aarch64' or not trt.__version__.startswith('8.2.1'):
            raise RuntimeError('Expected Jetson TRT8.2.1')
        lib = ct.CDLL('libnvinfer.so.8')
        lib.getInferLibVersion.restype = ct.c_int
        if lib.getInferLibVersion() != 8201 or sha256(engine_path) != expected_sha256:
            raise RuntimeError('Runtime or engine hash differs from deployment configuration')
        self.trt, self.role, self.cuda = trt, role, Cuda()
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        self.plugin_handles = []
        for path, expected in plugins:
            if sha256(path) != expected:
                raise RuntimeError('Plugin hash differs from deployment configuration')
            self.plugin_handles.append(ct.CDLL(str(Path(path).resolve()), mode=ct.RTLD_GLOBAL))
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None or self.engine.num_optimization_profiles != 1:
            raise RuntimeError('Expected one-profile deserialized engine')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Cannot create execution context')
        types = {trt.float32: np.float32, trt.float16: np.float16, trt.int32: np.int32,
                 trt.int8: np.int8, trt.bool: np.bool_}
        self.inputs, self.outputs, self.dtypes, self.profiles = [], [], {}, {}
        for i in range(self.engine.num_bindings):
            if (self.engine.get_binding_format(i) != trt.TensorFormat.LINEAR
                    or self.engine.get_location(i) != trt.TensorLocation.DEVICE
                    or self.engine.is_shape_binding(i)):
                raise RuntimeError('Only LINEAR DEVICE execution bindings are supported')
            name = self.engine.get_binding_name(i)
            self.dtypes[i] = np.dtype(types[self.engine.get_binding_dtype(i)])
            item = SimpleNamespace(name=name, shape=list(self.engine.get_binding_shape(i)), index=i)
            if self.engine.binding_is_input(i):
                self.inputs.append(item)
                self.profiles[name] = [list(s) for s in self.engine.get_profile_shape(0, i)]
            else:
                self.outputs.append(item)
        self.buffers = {}
        self.lock = threading.Lock()
        self.calls = self.errors = 0
        self.closed = False

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def in_profile(self, feeds):
        if set(feeds) != set(self.profiles):
            raise ValueError('Input names differ from the engine')
        for name, value in feeds.items():
            lo, _, hi = self.profiles[name]
            if value.ndim != len(lo) or any(d < a or d > b for d, a, b in zip(value.shape, lo, hi)):
                return False
        return True

    def _buffer(self, index, count):
        previous = self.buffers.get(index)
        if previous is not None and previous[1] >= count:
            return previous[0]
        if previous is not None:
            self.cuda.call('cudaFree', previous[0])
            del self.buffers[index]
        ptr = ct.c_void_p()
        self.cuda.call('cudaMalloc', ct.byref(ptr), count)
        self.buffers[index] = (ptr, count)
        return ptr

    def run(self, output_names, feeds):
        with self.lock:
            if self.closed or self.errors:
                raise RuntimeError(self.role + ' GPU session unhealthy')
            try:
                if output_names not in (None, []):
                    raise ValueError('Only all outputs are supported')
                if not self.in_profile(feeds):
                    raise ValueError('Input outside the engine profile')
                self.cuda.call('cudaSetDevice', 0)
                arrays = {}
                for item in self.inputs:
                    value = np.asarray(feeds[item.name])
                    dtype = self.dtypes[item.index]
                    if value.dtype == np.int64 and dtype == np.int32:
                        if value.size and (value.min() < -(2**31) or value.max() >= 2**31):
                            raise ValueError('INT64 input cannot be represented as INT32')
                        value = value.astype(np.int32)
                    if value.dtype != dtype or not np.isfinite(value).all():
                        raise ValueError('Input dtype or finite check failed')
                    arrays[item.index] = np.ascontiguousarray(value)
                    if not self.context.set_binding_shape(item.index, value.shape):
                        raise RuntimeError('Cannot set binding shape')
                for item in self.outputs:
                    shape = tuple(self.context.get_binding_shape(item.index))
                    if any(d <= 0 for d in shape):
                        raise RuntimeError('Unsupported unresolved/empty output shape')
                    arrays[item.index] = np.empty(shape, dtype=self.dtypes[item.index])
                pointers = [self._buffer(i, arrays[i].nbytes).value for i in range(self.engine.num_bindings)]
                for item in self.inputs:
                    a = arrays[item.index]
                    self.cuda.call('cudaMemcpy', ct.c_void_p(pointers[item.index]), ct.c_void_p(a.ctypes.data), a.nbytes, 1)
                if not self.context.execute_v2(pointers):
                    raise RuntimeError('TensorRT execution failed')
                self.cuda.call('cudaDeviceSynchronize')
                outputs = []
                for item in self.outputs:
                    a = arrays[item.index]
                    self.cuda.call('cudaMemcpy', ct.c_void_p(a.ctypes.data), ct.c_void_p(pointers[item.index]), a.nbytes, 2)
                    if not np.isfinite(a).all():
                        raise RuntimeError('Nonfinite GPU output')
                    outputs.append(a)
                self.calls += 1
                return outputs
            except Exception:
                self.errors += 1
                raise

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.context = None
            for ptr, _ in self.buffers.values():
                self.cuda.call('cudaFree', ptr)
            self.buffers.clear()
            self.engine = self.runtime = None
            self.plugin_handles.clear()
            self.closed = True

    def stats(self):
        return {'backend': 'tensorrt82_cuda', 'precision': 'fp32_no_tf32',
                'calls': self.calls, 'errors': self.errors, 'closed': self.closed,
                'profiles': self.profiles}
