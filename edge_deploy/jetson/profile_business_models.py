"""Offline per-model call accounting, preserving all seven release backends.

No HTTP, activation or counters; no tensors or recognized document text saved.
Times include existing session checks/copies/sync and explicit CPU fallback.
Instrumented single passes are bottleneck diagnostics, not service benchmarks.
"""
import argparse
from collections import Counter
import contextlib
import json
import logging
import os
from pathlib import Path
import sys
import time

import quantize_yolo_ptq as q


def summarize(rows):
    result = {}
    for role in sorted({r['role'] for r in rows}):
        group = [r for r in rows if r['role'] == role]
        times = [r['wall_ms'] for r in group]
        shapes = Counter(json.dumps(r['shapes'], sort_keys=True) for r in group)
        result[role] = {'calls': len(group), 'total_ms': sum(times),
                        'mean_ms': sum(times) / len(times),
                        'unique_shapes': len(shapes), 'shape_counts': dict(shapes),
                        'cpu_fallback_calls': sum(r['cpu_fallback'] for r in group)}
    return result


def attach_timers(models, calls):
    originals = []
    for role, session in models.models.items():
        original = session.run
        originals.append((session, original))

        def run(names, feeds, role=role, session=session, original=original):
            shapes = {k: list(v.shape) for k, v in feeds.items()}
            fallback = hasattr(session, 'gpu') and not session.gpu.in_profile(feeds)
            start = time.perf_counter()
            result = original(names, feeds)
            elapsed = (time.perf_counter() - start) * 1000
            calls.append({'role': role, 'shapes': shapes, 'cpu_fallback': fallback,
                          'wall_ms': elapsed})
            return result

        session.run = run
    return originals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--count', type=int, default=3)
    a = p.parse_args()
    stage = a.stage.resolve()
    if not stage.is_relative_to(q.BASE / 'staging') or not stage.is_dir():
        p.error('Use an existing project staging directory')
    target = stage / 'module_profile.json'
    if target.exists() or not 1 <= a.count <= 8:
        p.error('Report exists or invalid calibration subset count')
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE),
                      AUTOOCR_RUNTIME_DIR=str(stage / 'runtime'), AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL)
    sys.path.insert(0, str(q.RELEASE / 'edge_deploy/jetson'))
    from serve_jetson import deployment, load_models
    plan = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
    images = plan['calibration'][:a.count]
    report = {'status': 'running', 'script_sha256': q.digest(__file__),
              'deployment_sha256': plan['deployment_sha256'],
              'source_plan_sha256': q.digest(q.BASE / 'staging/quant_yolo_e92_r1/plan.json'),
              'subset': 'first N hash-ordered calibration images, no holdout',
              'before': q.memory(), 'rows': [], 'note': __doc__}
    calls, originals, models = [], [], None
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, root, cfg = deployment()
                pipeline, models = load_models(server, root, cfg)
            originals = attach_timers(models, calls)
            report['engine_sha256'] = {k: v['sha256'] for k, v in cfg['engines'].items()}
            report['after_warmup'] = q.memory()
            for index, image in enumerate(images):
                for kind in (3, 1):
                    calls.clear()
                    start = time.perf_counter()
                    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                        out = pipeline.process(str(Path(plan['images']) / image['file']),
                                               image_type=kind, run_ner=True)
                    elapsed = (time.perf_counter() - start) * 1000
                    if not out['success'] or not models.healthy:
                        raise RuntimeError('Pipeline or optional backend error')
                    row = {'image_sha256': image['sha256'], 'image_type': kind,
                           'pipeline_wall_ms': elapsed, 'modules': summarize(calls),
                           'calls': list(calls)}
                    report['rows'].append(row)
                    print(json.dumps({'image': index + 1, 'image_type': kind,
                                      'pipeline_ms': round(elapsed, 1),
                                      'modules_ms': {k: round(v['total_ms'], 1)
                                                     for k, v in row['modules'].items()}}), flush=True)
        report.update(status='profiled_not_benchmark_or_accuracy', execution=models.stats(),
                      after=q.memory())
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        for session, original in originals:
            session.run = original
        if models is not None:
            for session in models.models.values():
                getattr(session, 'gpu', session).close()
        q.save(target, report)


if __name__ == '__main__':
    main()
