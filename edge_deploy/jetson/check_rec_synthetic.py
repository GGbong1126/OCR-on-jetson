"""Synthetic width-boundary ORT/TRT check; no private document input."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import onnxruntime as ort
from trt82_backend import Session, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--onnx', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Report exists')
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 2
    cpu = ort.InferenceSession(str(args.onnx), sess_options=opt, providers=['CPUExecutionProvider'])
    gpu = Session(args.engine, sha256(args.engine), 'rec')
    report = {'status': 'running', 'model_sha256': sha256(args.onnx),
              'engine_sha256': sha256(args.engine), 'comparisons': []}
    try:
        rng = np.random.default_rng(0)
        for width in (32, 160, 959, 1536):
            value = rng.uniform(-1, 1, (1, 3, 48, width)).astype(np.float32)
            start = perf_counter()
            a = cpu.run(None, {cpu.get_inputs()[0].name: value})[0]
            b = gpu.run(None, {'x': value})[0]
            if a.shape != b.shape or not np.isfinite(a).all():
                raise ValueError('Invalid reference')
            diff = np.abs(a.astype(np.float64)-b.astype(np.float64))
            row = {'width': width, 'shape': list(a.shape), 'mae': float(diff.mean()),
                   'max_abs_error': float(diff.max()),
                   'allclose': bool(np.allclose(a,b,rtol=1e-3,atol=1e-4)),
                   'argmax_equal': bool(np.array_equal(a.argmax(-1), b.argmax(-1))),
                   'combined_diagnostic_wall_s': perf_counter()-start}
            report['comparisons'].append(row)
            print(json.dumps(row), flush=True)
        report['status'] = 'synthetic_compared_not_text_validated'
    except Exception as exc:
        report.update(status='error', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        gpu.close()
        with args.output.open('x') as f:
            json.dump(report,f,indent=2)


if __name__ == '__main__':
    main()
