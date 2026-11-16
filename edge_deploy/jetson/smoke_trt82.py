"""Small real TensorRT/CUDA computation; no model, license or clock changes.

For the pinned JetPack 4.6.1 stack only. This is not an OCR accuracy/performance
test. JSON is emitted to stdout; the caller may preserve it as a report.
"""
import ctypes as ct
import json
import platform

import numpy as np


def main():
    import tensorrt as trt
    if platform.machine() != 'aarch64' or trt.__version__ != '8.2.1.9':
        raise RuntimeError('Expected aarch64 and verified TensorRT Python 8.2.1.9, got ' + trt.__version__)
    cuda = ct.CDLL('libcudart.so.10.2')
    signatures = {
        'cudaSetDevice': [ct.c_int],
        'cudaGetDeviceCount': [ct.POINTER(ct.c_int)],
        'cudaRuntimeGetVersion': [ct.POINTER(ct.c_int)],
        'cudaDriverGetVersion': [ct.POINTER(ct.c_int)],
        'cudaMalloc': [ct.POINTER(ct.c_void_p), ct.c_size_t],
        'cudaFree': [ct.c_void_p],
        'cudaMemcpy': [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int],
        'cudaDeviceSynchronize': [],
    }
    for name, args in signatures.items():
        getattr(cuda, name).argtypes = args
        getattr(cuda, name).restype = ct.c_int
    cuda.cudaGetErrorString.argtypes = [ct.c_int]
    cuda.cudaGetErrorString.restype = ct.c_char_p

    def call(name, *args):
        code = getattr(cuda, name)(*args)
        if code:
            raise RuntimeError('{}: {} {}'.format(name, code, cuda.cudaGetErrorString(code)))

    count, runtime, driver = ct.c_int(), ct.c_int(), ct.c_int()
    call('cudaGetDeviceCount', ct.byref(count))
    call('cudaRuntimeGetVersion', ct.byref(runtime))
    call('cudaDriverGetVersion', ct.byref(driver))
    if count.value < 1 or runtime.value != 10020:
        raise RuntimeError('CUDA10.2 GPU runtime unavailable')
    call('cudaSetDevice', 0)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    shape = (1, 3, 32, 32)
    x = network.add_input('x', trt.float32, shape)
    y = network.add_elementwise(x, x, trt.ElementWiseOperation.PROD).get_output(0)
    y.name = 'square'
    network.mark_output(y)
    config = builder.create_builder_config()
    config.max_workspace_size = 64 * 1024**2
    config.clear_flag(trt.BuilderFlag.TF32)
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError('Synthetic TensorRT engine build failed')
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError('Execution context allocation failed')
    data = (np.arange(np.prod(shape), dtype=np.float32) % 31).reshape(shape)
    output = np.empty_like(data)
    pointers = [ct.c_void_p() for _ in range(2)]
    try:
        for p in pointers:
            call('cudaMalloc', ct.byref(p), data.nbytes)
        call('cudaMemcpy', pointers[0], ct.c_void_p(data.ctypes.data), data.nbytes, 1)
        bindings = [0] * engine.num_bindings
        bindings[engine.get_binding_index('x')] = pointers[0].value
        bindings[engine.get_binding_index('square')] = pointers[1].value
        if not context.execute_v2(bindings):
            raise RuntimeError('TensorRT GPU execution failed')
        call('cudaDeviceSynchronize')
        call('cudaMemcpy', ct.c_void_p(output.ctypes.data), pointers[1], output.nbytes, 2)
        if not np.isfinite(output).all() or not np.array_equal(output, data * data):
            raise RuntimeError('Synthetic GPU result differs from CPU expectation')
        print(json.dumps({
            'status': 'synthetic_gpu_compute_passed_not_ocr_validated',
            'machine': platform.machine(), 'python': platform.python_version(),
            'numpy': np.__version__, 'tensorrt': trt.__version__,
            'cuda_runtime': runtime.value, 'cuda_driver': driver.value,
            'device_count': count.value, 'input_shape': list(shape),
            'operation': 'elementwise square FP32, no TF32',
            'elements_checked': int(data.size), 'exactly_equal': True,
            'note': 'No OCR model, business service, accuracy or benchmark tested.'
        }, indent=2))
    finally:
        context = None
        for p in pointers:
            if p.value:
                call('cudaFree', p)


if __name__ == '__main__':
    main()
