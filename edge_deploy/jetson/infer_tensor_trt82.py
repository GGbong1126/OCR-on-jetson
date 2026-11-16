"""Check one real preprocessed tensor against CPU using a Jetson TRT8.2 engine.

NumPy + ctypes only; no new packages, model transformation, image preprocessing,
license access or service changes. Synchronous FP32 I/O, one optimization profile.
The output array can be checked with the unchanged host postprocessor.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    import tensorrt as trt
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--case', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Output exists; choose a new report directory')
    if platform.machine() != 'aarch64' or trt.__version__ != '8.2.1.9':
        raise RuntimeError('Expected verified aarch64 TensorRT8.2.1.9')
    meta = json.loads((args.case / 'case.json').read_text())
    for name in ('input', 'reference'):
        if sha256(args.case / (name + '.npy')) != meta[name + '_npy_sha256']:
            raise ValueError('Case array hash mismatch: ' + name)
    tensor = np.load(str(args.case / 'input.npy'), allow_pickle=False)
    reference = np.load(str(args.case / 'reference.npy'), allow_pickle=False)
    if (tensor.dtype != np.float32 or tensor.ndim != 4 or tensor.shape[:2] != (1, 3)
            or not np.isfinite(tensor).all() or not np.isfinite(reference).all()):
        raise ValueError('Expected finite FP32 image tensor and finite CPU reference')
    tensor = np.ascontiguousarray(tensor)
    cuda = ct.CDLL('libcudart.so.10.2')
    signatures = {'cudaSetDevice': [ct.c_int],
                  'cudaMalloc': [ct.POINTER(ct.c_void_p), ct.c_size_t],
                  'cudaFree': [ct.c_void_p], 'cudaDeviceSynchronize': [],
                  'cudaMemcpy': [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int]}
    for name, types in signatures.items():
        getattr(cuda, name).argtypes = types
        getattr(cuda, name).restype = ct.c_int
    cuda.cudaGetErrorString.argtypes = [ct.c_int]
    cuda.cudaGetErrorString.restype = ct.c_char_p

    def call(name, *values):
        code = getattr(cuda, name)(*values)
        if code:
            raise RuntimeError('{}: {} {}'.format(name, code, cuda.cudaGetErrorString(code)))

    call('cudaSetDevice', 0)
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None or engine.num_bindings != 2 or engine.num_optimization_profiles != 1:
        raise ValueError('Expected a loaded single-profile two-binding engine')
    ins = [i for i in range(2) if engine.binding_is_input(i)]
    if len(ins) != 1:
        raise ValueError('Expected one input and one output')
    ii, oi = ins[0], 1 - ins[0]
    for i in range(2):
        if (engine.get_binding_dtype(i) != trt.float32
                or engine.get_binding_format(i) != trt.TensorFormat.LINEAR
                or engine.get_location(i) != trt.TensorLocation.DEVICE):
            raise ValueError('Expected DEVICE, LINEAR, FP32 bindings')
    low, opt, high = engine.get_profile_shape(0, ii)
    if len(low) != tensor.ndim or any(d < lo or d > hi for d, lo, hi in zip(tensor.shape, low, high)):
        raise ValueError('Input outside engine profile')
    context = engine.create_execution_context()
    if context is None or not context.set_binding_shape(ii, tensor.shape):
        raise RuntimeError('Cannot create context or set input shape')
    shape = tuple(context.get_binding_shape(oi))
    if any(d <= 0 for d in shape) or shape != reference.shape:
        raise ValueError('Output shape differs from CPU reference')
    output = np.empty(shape, dtype=np.float32)
    pointers = [ct.c_void_p(), ct.c_void_p()]
    times = []
    try:
        call('cudaMalloc', ct.byref(pointers[ii]), tensor.nbytes)
        call('cudaMalloc', ct.byref(pointers[oi]), output.nbytes)
        for _ in range(3):
            start = perf_counter()
            call('cudaMemcpy', pointers[ii], ct.c_void_p(tensor.ctypes.data), tensor.nbytes, 1)
            if not context.execute_v2([p.value for p in pointers]):
                raise RuntimeError('TensorRT execution failed')
            call('cudaDeviceSynchronize')
            call('cudaMemcpy', ct.c_void_p(output.ctypes.data), pointers[oi], output.nbytes, 2)
            times.append((perf_counter() - start) * 1000)
            if not np.isfinite(output).all():
                raise ValueError('Nonfinite model output; experiment invalid')
        error = np.abs(output.astype(np.float64) - reference.astype(np.float64))
        result = {'status': 'finite_tensor_compared_not_accuracy_validated',
                  'engine_sha256': sha256(args.engine), 'case': meta,
                  'tensorrt': trt.__version__, 'output_shape': list(shape),
                  'probability_mae': float(error.mean()), 'max_abs_error': float(error.max()),
                  'allclose_rtol_1e_3_atol_1e_4': bool(np.allclose(output, reference, rtol=1e-3, atol=1e-4)),
                  'mask_disagreement_fraction_at_0_1': float(np.mean((output > .1) != (reference > .1))),
                  'call_ms': times,
                  'note': '3 single-tensor calls incl H2D/execute/sync/D2H; not benchmark, OCR text or GT accuracy.'}
        args.output.mkdir(parents=True)
        np.save(str(args.output / 'output.npy'), output, allow_pickle=False)
        with (args.output / 'comparison.json').open('x') as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
    finally:
        context = None
        for p in pointers:
            if p.value:
                call('cudaFree', p)


if __name__ == '__main__':
    main()
