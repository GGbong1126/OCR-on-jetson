"""Compare native TRT8.2 against saved host tensors, without business activation."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from trt82_backend import Session, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--case', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Report exists')
    engine_hash = sha256(args.engine)
    session = Session(args.engine, engine_hash, 'diagnostic')
    report = {'engine_sha256': engine_hash, 'status': 'running', 'calls': []}
    try:
        feeds = {}
        for item in session.inputs:
            path = args.case / (item.name + '.npy')
            if not path.is_file() and len(session.inputs) == 1:
                path = args.case / 'input.npy'
            feeds[item.name] = np.load(path, allow_pickle=False)
        reference = np.load(args.case / 'reference.npy', allow_pickle=False)
        report['input_shapes'] = {k: list(v.shape) for k, v in feeds.items()}
        report['array_sha256'] = {p.name: sha256(p) for p in args.case.glob('*.npy')}
        for _ in range(2):
            start = perf_counter()
            output = session.run(None, feeds)[0]
            elapsed = (perf_counter() - start) * 1000
            if reference.shape != output.shape or not np.isfinite(reference).all():
                raise ValueError('Reference shape/finite mismatch')
            diff = np.abs(output.astype(np.float64) - reference.astype(np.float64))
            report['calls'].append({'wall_ms': elapsed, 'output_shape': list(output.shape),
                'mae': float(diff.mean()), 'max_abs_error': float(diff.max()),
                'allclose': bool(np.allclose(reference, output, rtol=1e-3, atol=1e-4)),
                'argmax_last_axis_equal': bool(np.array_equal(reference.argmax(-1), output.argmax(-1)))})
        report['status'] = 'finite_compared_not_task_validated'
        report['note'] = 'Two diagnostic calls, no benchmark or ground-truth accuracy claim.'
        report['execution'] = session.stats()
    except Exception as exc:
        report.update(status='error', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        session.close()
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
