"""TensorRT 10 single-image backend for WSL/Linux CUDA 12.

Uses the installed CUDA Runtime through ctypes; no PyTorch/PyCUDA needed.
One context, one stream, cached buffers; sequential use only. run_tensor's
output is borrowed and is overwritten by the next call. Copy if retaining it.
"""
import ctypes as ct
import os
import platform
import warnings
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

if __package__:
    from .ocr_det import Detector, sha256_file
else:
    from ocr_det import Detector, sha256_file


class CudaRuntime:
    def __init__(self):
        library = os.environ.get("CUDA_RUNTIME_LIB", "libcudart.so.12")
        try:
            self.lib = ct.CDLL(library)
        except OSError as exc:
            raise RuntimeError("Cannot load libcudart.so.12. Add /usr/local/cuda-12.6/lib64 to LD_LIBRARY_PATH") from exc
        signatures = {
            "cudaSetDevice": [ct.c_int],
            "cudaMalloc": [ct.POINTER(ct.c_void_p), ct.c_size_t],
            "cudaFree": [ct.c_void_p],
            "cudaStreamCreate": [ct.POINTER(ct.c_void_p)],
            "cudaStreamDestroy": [ct.c_void_p],
            "cudaStreamSynchronize": [ct.c_void_p],
            "cudaMemcpyAsync": [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_void_p],
            "cudaRuntimeGetVersion": [ct.POINTER(ct.c_int)],
        }
        for name, argtypes in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = argtypes, ct.c_int
        self.lib.cudaGetErrorString.argtypes = [ct.c_int]
        self.lib.cudaGetErrorString.restype = ct.c_char_p

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            msg = self.lib.cudaGetErrorString(code)
            raise RuntimeError(f"{name}: CUDA error {code}: {msg.decode() if msg else 'unknown'}")


class TensorRTDetector(Detector):
    def __init__(self, model):
        import tensorrt as trt
        self.trt = trt
        self.model = Path(model)
        self.cuda = CudaRuntime()
        self.stream = ct.c_void_p()
        self.buffers = {}
        self.current_shape = None
        self.context = self.engine = self.runtime = None
        self.closed = False
        start = perf_counter()
        try:
            self.cuda.call("cudaSetDevice", 0)
            self.logger = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(self.logger, "")
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(self.model.read_bytes())
            if self.engine is None:
                raise RuntimeError("Engine deserialization failed; check GPU/TRT version and build log")
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError("This baseline supports exactly one image input and one output")
            self.input_name, self.output_name = inputs[0], outputs[0]
            for name in names:
                if self.engine.get_tensor_dtype(name) != trt.float32:
                    raise ValueError(f"Expected FP32 I/O for {name}; internal FP16 compute is allowed")
                if self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                    raise ValueError("Host tensors are not supported by this baseline")
                if self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                    raise ValueError("Only linear tensor layout is supported")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Cannot allocate execution context")
            self.cuda.call("cudaStreamCreate", ct.byref(self.stream))
            version = ct.c_int()
            self.cuda.call("cudaRuntimeGetVersion", ct.byref(version))
            self.cuda_version = version.value
            self.load_ms = (perf_counter() - start) * 1000
        except Exception:
            self.close()
            raise

    def _prepare(self, shape):
        if shape == self.current_shape:
            return
        low, _, high = self.engine.get_tensor_profile_shape(self.input_name, 0)
        if len(shape) != len(low) or any(s < lo or s > hi for s, lo, hi in zip(shape, low, high)):
            raise ValueError(f"Input {shape} outside engine profile [{tuple(low)}, {tuple(high)}]; rebuild the profile or use the matching image")
        if not self.context.set_input_shape(self.input_name, shape):
            raise ValueError(f"Cannot set input shape {shape}")
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        if any(d <= 0 for d in output_shape):
            raise ValueError(f"Unresolved output shape: {output_shape}")
        self.current_shape = None
        for name, dims in [(self.input_name, shape), (self.output_name, output_shape)]:
            nbytes = math_product(dims) * np.dtype(np.float32).itemsize
            old = self.buffers.get(name)
            if old and old[1] != nbytes:
                self.cuda.call("cudaFree", old[0])
                del self.buffers[name]
            if name not in self.buffers:
                pointer = ct.c_void_p()
                self.cuda.call("cudaMalloc", ct.byref(pointer), nbytes)
                self.buffers[name] = (pointer, nbytes)
            if not self.context.set_tensor_address(name, self.buffers[name][0].value):
                raise RuntimeError(f"Cannot bind {name}")
        self.host_output = np.empty(output_shape, dtype=np.float32)
        self.current_shape = shape

    def run_tensor(self, tensor):
        if self.closed:
            raise RuntimeError("Detector is closed")
        if tensor.dtype != np.float32 or tensor.ndim != 4 or tensor.shape[:2] != (1, 3):
            raise ValueError("Expected float32 [1,3,H,W]")
        self._prepare(tuple(tensor.shape))
        host_input = np.ascontiguousarray(tensor)
        try:
            self.cuda.call("cudaMemcpyAsync", self.buffers[self.input_name][0],
                           ct.c_void_p(host_input.ctypes.data), host_input.nbytes, 1, self.stream)
            if not self.context.execute_async_v3(stream_handle=self.stream.value):
                raise RuntimeError("TensorRT execution failed")
            self.cuda.call("cudaMemcpyAsync", ct.c_void_p(self.host_output.ctypes.data),
                           self.buffers[self.output_name][0], self.host_output.nbytes, 2, self.stream)
        finally:
            # Host buffers remain alive until all copies and kernels complete.
            self.cuda.call("cudaStreamSynchronize", self.stream)
        return self.host_output

    def metadata(self):
        return {
            "backend": "tensorrt", "engine": str(self.model.resolve()),
            "engine_sha256": sha256_file(self.model), "tensorrt": self.trt.__version__,
            "cuda_runtime_version": self.cuda_version, "device_index": 0,
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "opencv": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(), "batch_size": 1,
            "session_load_ms": self.load_ms,
            "profile": [list(s) for s in self.engine.get_tensor_profile_shape(self.input_name, 0)],
            "io_dtype": "float32; internal precision determined by engine build",
            "timing": "H2D + enqueue + D2H + stream synchronization + host input packing; buffer allocation on first shape use",
            "preprocessing": "BGR / 255; long side 960 (upscale allowed); ceil to 32",
            "postprocessing": "same server.py reference as ORT baseline",
        }

    def close(self):
        if self.closed:
            return
        if self.stream.value:
            self.cuda.call("cudaStreamSynchronize", self.stream)
        self.context = None
        for name in list(self.buffers):
            self.cuda.call("cudaFree", self.buffers[name][0])
            del self.buffers[name]
        if self.stream.value:
            self.cuda.call("cudaStreamDestroy", self.stream)
            self.stream = ct.c_void_p()
        self.engine = self.runtime = None
        self.closed = True

    def __del__(self):
        if hasattr(self, "closed"):
            try:
                self.close()
            except Exception as exc:
                warnings.warn(f"CUDA cleanup failed: {exc}")


def math_product(dims):
    import math
    return math.prod(dims)
