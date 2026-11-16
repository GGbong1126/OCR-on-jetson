"""E102: same-FP32-engine runtime diagnostics, not a service benchmark.

Capture real recognition inputs in RAM only. Compare repeated and alternating
widths, preserving each tensor. Instrument existing Session.run with proxies,
then measure uninstrumented calls separately. Never changes a release.
"""
import argparse
import contextlib
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import optimize_rec_fp16 as common
import quantize_yolo_ptq as q


def tensor_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class ContextProxy:
    """Only skip an already successful, identical binding shape if requested.

    No external profile/shape mutation is allowed while installed. Owns neither
    the context nor its memory. Fresh outputs remain owned by Session.run.
    """
    def __init__(self, target, skip=False):
        self.target, self.skip = target, skip
        self.shapes = {}
        self.row = None

    def __getattr__(self, name):
        return getattr(self.target, name)

    def set_binding_shape(self, index, shape):
        shape = tuple(shape)
        changed = self.shapes.get(index) != shape
        if self.row is None:
            if self.skip and not changed:
                return True
            ok = self.target.set_binding_shape(index, shape)
            if ok:
                self.shapes[index] = shape
            return ok
        start = time.perf_counter()
        skipped = self.skip and not changed
        ok = True if skipped else self.target.set_binding_shape(index, shape)
        elapsed = (time.perf_counter() - start) * 1000
        if ok:
            self.shapes[index] = shape
        if self.row is not None:
            self.row['shape_ms'] += elapsed
            self.row['shape_changes'] += int(changed)
            self.row['shape_api_calls'] += int(not skipped)
        return ok

    def execute_v2(self, bindings):
        if self.row is None:
            return self.target.execute_v2(bindings)
        start = time.perf_counter()
        ok = self.target.execute_v2(bindings)
        if self.row is not None:
            self.row['execute_v2_host_ms'] += (time.perf_counter() - start) * 1000
        return ok


class CudaProxy:
    def __init__(self, target):
        self.target, self.row = target, None

    def __getattr__(self, name):
        return getattr(self.target, name)

    def call(self, name, *args):
        start = time.perf_counter()
        result = self.target.call(name, *args)
        elapsed = (time.perf_counter() - start) * 1000
        key = {'cudaMalloc': 'malloc', 'cudaFree': 'free',
               'cudaDeviceSynchronize': 'sync'}.get(name)
        if name == 'cudaMemcpy':
            key = 'h2d' if args[-1] == 1 else 'd2h'
        if self.row is not None and key:
            self.row[key + '_ms'] += elapsed
            self.row[key + '_count'] += 1
        return result


def new_row(index):
    row = dict(index=index, shape_ms=0.0, execute_v2_host_ms=0.0,
               shape_changes=0, shape_api_calls=0)
    for key in ('malloc', 'free', 'sync', 'h2d', 'd2h'):
        row[key + '_ms'], row[key + '_count'] = 0.0, 0
    return row


def patterns(feeds):
    if not feeds:
        raise ValueError('No real input sequence')
    widths = [f['x'].shape[3] for f in feeds]
    a, b = widths.index(min(widths)), widths.index(max(widths))
    if a == b:
        raise ValueError('Need two distinct widths')
    return {'repeat_a': [a, a], 'repeat_b': [b, b],
            'alternate_ab': [a, b], 'alternate_ba': [b, a],
            'original_sequence': list(range(len(feeds)))}


def describe_sequence(shapes, capacity):
    """Diagnostic exact-shape LRU simulation; not GPU memory or latency data."""
    cache, hits, changes = [], 0, 0
    previous = None
    for value in shapes:
        key = tuple(value)
        changes += int(previous is not None and previous != key)
        previous = key
        if key in cache:
            hits += 1
            cache.remove(key)
        cache.append(key)
        if len(cache) > capacity:
            cache.pop(0)
    return {'capacity': capacity, 'cold_sequence_hits': hits,
            'calls': len(shapes), 'adjacent_changes': changes,
            'unique_shapes': len({tuple(s) for s in shapes})}


def capture(a):
    plan = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
    models = None
    feeds, hashes = [], []
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = common.setup(a)
            gpu = models.models['rec'].gpu
            original = gpu.run
            def run(names, values):
                result = original(names, values)
                feeds.append({k: v.copy() for k, v in values.items()})
                hashes.append([tensor_hash(v) for v in result])
                return result
            gpu.run = run
            image = plan['calibration'][0]
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                result = pipeline.process(str(Path(plan['images']) / image['file']),
                                          image_type=3, run_ner=True)
            gpu.run = original
            if not result['success'] or not models.healthy or not feeds:
                raise RuntimeError('Original input capture failed')
            execution = models.stats()
    finally:
        common.close_models(models)
    return feeds, hashes, cfg, image['sha256'], execution


def probe_profiles(native):
    second = native.engine.create_execution_context()
    row = {'profiles': native.engine.num_optimization_profiles,
           'first_active_profile': native.context.active_optimization_profile,
           'second_created': second is not None,
           'second_active_profile': (second.active_optimization_profile if second is not None else None),
           'note': 'Read-only context capability probe. No forced profile selection or execution on an invalid profile.'}
    second = None
    gc.collect()
    return row


def measure(a):
    from trt82_backend import Session
    target = a.stage / ('runtime_%s_%s.json' % (a.variant, a.round))
    common.fresh(target)
    report = {'status': 'running', 'variant': a.variant, 'round': a.round,
              'script_sha256': q.digest(__file__), 'patterns': [], 'note': __doc__,
              'warmup_cycles': a.warmup, 'repeat_cycles': a.repeat}
    native = None
    try:
        feeds, expected, cfg, image, execution = capture(a)
        gc.collect()
        report.update(image_sha256=image, capture_execution=execution,
                      input_shapes=[list(f['x'].shape) for f in feeds],
                      input_sha256=[tensor_hash(f['x']) for f in feeds],
                      baseline_output_sha256=expected,
                      deployment_sha256=q.digest(q.RELEASE / 'edge_deploy/jetson/deployment.json'),
                      backend_sha256=q.digest(q.RELEASE / 'edge_deploy/jetson/trt82_backend.py'),
                      before=common.telemetry())
        spec = cfg['engines']['rec']
        native = Session(q.RELEASE / spec['path'], spec['sha256'], 'rec_runtime_fp32')
        report.update(engine_sha256=spec['sha256'], context_device_bytes=native.engine.device_memory_size,
                      profile_probe=probe_profiles(native))
        raw_context, raw_cuda = native.context, native.cuda
        cp, cuda = ContextProxy(raw_context, skip=a.variant == 'skip_same'), CudaProxy(raw_cuda)
        native.context, native.cuda = cp, cuda
        cases = patterns(feeds)
        order = list(cases)
        if a.round == 'r2':
            order = ['original_sequence', 'alternate_ba', 'alternate_ab', 'repeat_b', 'repeat_a']
        report['sequence_cache_simulation'] = [describe_sequence(report['input_shapes'], n) for n in (2, 4, 8, 16)]
        for name in order:
            sequence = cases[name]
            for _ in range(a.warmup):
                for index in sequence:
                    out = native.run(None, feeds[index])
                    if [tensor_hash(v) for v in out] != expected[index]:
                        raise RuntimeError('Same-engine output drift in warmup')
            rows = []
            for cycle in range(a.repeat):
                for index in sequence:
                    row = new_row(index)
                    row['cycle'] = cycle
                    cp.row = cuda.row = row
                    start = time.perf_counter()
                    out = native.run(None, feeds[index])
                    row['call_ms'] = (time.perf_counter() - start) * 1000
                    cp.row = cuda.row = None
                    row['output_exact'] = [tensor_hash(v) for v in out] == expected[index]
                    if not row['output_exact']:
                        raise RuntimeError('Same-engine output drift')
                    rows.append(row)
            report['patterns'].append({'name': name, 'rows': rows,
                'means_ms': {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k.endswith('_ms')},
                'shape_changes': sum(r['shape_changes'] for r in rows),
                'malloc_count': sum(r['malloc_count'] for r in rows)})
            print(json.dumps({'pattern': name, **{k: report['patterns'][-1][k] for k in ('means_ms', 'shape_changes', 'malloc_count')}}), flush=True)
        # Remove timing proxies from baseline entirely; candidate keeps only its
        # minimal shape-skip wrapper, no per-call timings or hash checks inside.
        native.cuda = raw_cuda
        native.context = raw_context if a.variant == 'reference' else cp
        for _ in range(a.warmup):
            for feed in feeds:
                native.run(None, feed)
        report['after_warmup'] = common.telemetry()
        bare = []
        for cycle in range(a.repeat):
            for index, feed in enumerate(feeds):
                start = time.perf_counter()
                out = native.run(None, feed)
                bare.append({'cycle': cycle, 'index': index, 'wall_ms': (time.perf_counter() - start) * 1000})
        # Post-timing hashes, no per-call hashing between measured calls.
        for index, feed in enumerate(feeds):
            if [tensor_hash(v) for v in native.run(None, feed)] != expected[index]:
                raise RuntimeError('Post-timing output drift')
        report.update(status='diagnosed_exact_same_engine_not_gt',
                      bare_rows=bare, bare_timing=common.summary([r['wall_ms'] for r in bare]),
                      after=common.telemetry(), execution=native.stats(),
                      io_buffer_bytes=sum(p[1] for p in native.buffers.values()))
        native.context = raw_context
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        if native is not None:
            native.close()
        q.save(target, report)
    print(json.dumps({'status': report['status'], 'variant': a.variant, 'round': a.round,
                      'bare_timing': report['bare_timing'], 'profile_probe': report['profile_probe']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--variant', choices=('reference', 'skip_same'), required=True)
    p.add_argument('--round', choices=('r1', 'r2'), required=True)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--repeat', type=int, default=20)
    a = p.parse_args()
    a.stage = a.stage.resolve()
    if (platform.machine() != 'aarch64' or not a.stage.is_relative_to(q.BASE / 'staging')
            or not a.stage.is_dir() or min(a.warmup, a.repeat) < 1):
        p.error('Existing board project staging and positive counts required')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(q.RELEASE / 'edge_deploy/jetson'))
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE), AUTOOCR_RUNTIME_DIR=str(a.stage / 'runtime'),
                      AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL)
    measure(a)


if __name__ == '__main__':
    main()
