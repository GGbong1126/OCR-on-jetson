"""E103: seven-model, same-engine profiling to prioritize distinct optimizations.

Original real input tensors remain RAM-only. Uninstrumented Session.run and
IProfiler diagnostics use separate Sessions. No model, license or service edits.
"""
import argparse
from collections import defaultdict
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

ROLES = ('rec', 'enhance', 'det', 'layout', 'cell', 'ner', 'yolo')


def digest_tensor(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def signature(feeds):
    return tuple((name, tuple(value.shape), str(value.dtype)) for name, value in sorted(feeds.items()))


def choose_cases(cases):
    """Prefer min/middle/max unique shape volume; same-shape use data diversity.

    Representative samples, not a workload-weighted throughput estimate.
    """
    if not cases:
        raise ValueError('No real cases')
    unique = {}
    for index, case in enumerate(cases):
        unique.setdefault(signature(case['feeds']), index)
    indices = sorted(unique.values(), key=lambda i: sum(v.size for v in cases[i]['feeds'].values()))
    if len(indices) == 1:
        indices = list(range(len(cases)))
    return list(dict.fromkeys(indices[i] for i in (0, len(indices) // 2, len(indices) - 1)))


def capture(a):
    plan = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
    cases, calls, originals = {role: [] for role in ROLES}, [], []
    budget = 256 * 1024**2
    used = 0
    models = None
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = common.setup(a)
            output_names = {}
            for role in ROLES:
                native = getattr(models.models[role], 'gpu', models.models[role])
                original = native.run
                originals.append((native, original))
                output_names[role] = [v.name for v in native.outputs]

                def run(names, feeds, role=role, original=original):
                    nonlocal used
                    size = sum(v.nbytes for v in feeds.values())
                    if used + size > budget or len(cases[role]) >= 256:
                        raise RuntimeError('RAM capture budget exceeded; do not write tensors to disk')
                    start = time.perf_counter()
                    out = original(names, feeds)
                    elapsed = (time.perf_counter() - start) * 1000
                    cases[role].append({'feeds': {k: v.copy() for k, v in feeds.items()},
                                        'outputs_sha256': [digest_tensor(v) for v in out]})
                    used += size
                    calls.append({'role': role, 'case_index': len(cases[role]) - 1,
                                  'image_type': active[0], 'call_ms': elapsed})
                    return out

                native.run = run
            image = plan['calibration'][0]
            active = [3]
            for kind in (3, 1):
                active[0] = kind
                with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                    out = pipeline.process(str(Path(plan['images']) / image['file']),
                                           image_type=kind, run_ner=True)
                if not out['success'] or not models.healthy:
                    raise RuntimeError('Incomplete original business capture')
            execution = models.stats()
    finally:
        for native, original in originals:
            native.run = original
        common.close_models(models)
    if set(role for role, values in cases.items() if values) != set(ROLES):
        raise ValueError('All seven models must have real calls')
    return cases, calls, cfg, image['sha256'], output_names, execution


def layer_summary(callbacks, calls):
    if calls < 1 or not callbacks:
        raise ValueError('Missing profiler data')
    sums, counts = defaultdict(float), defaultdict(int)
    for name, ms in callbacks:
        if not isinstance(name, str) or not name or not np.isfinite(ms) or ms < 0:
            raise ValueError('Invalid callback')
        sums[name] += float(ms)
        counts[name] += 1
    total = sum(sums.values())
    if total <= 0:
        raise ValueError('No positive layer timings')
    return {'summed_layer_ms': total / calls,
            'layers': sorted([{'name': name, 'callbacks': counts[name],
                               'mean_ms': ms / calls, 'fraction_of_layer_sum': ms / total}
                              for name, ms in sums.items()], key=lambda r: r['mean_ms'], reverse=True),
            'note': 'Fused layer callback intervals, not original ONNX operator attribution or service latency. Profiler adds overhead.'}


def describe(case):
    return {'inputs': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'sha256': digest_tensor(v)}
                       for k, v in case['feeds'].items()}, 'outputs_sha256': case['outputs_sha256']}


def verify(out, case):
    if [digest_tensor(v) for v in out] != case['outputs_sha256']:
        raise RuntimeError('Same-engine output drift; no tolerance change')


def measure_role(a, role, cases, cfg, names, image):
    from trt82_backend import Session
    target = a.stage / ('%s_%s.json' % (role, a.round))
    common.fresh(target)
    selected = choose_cases(cases)
    if a.round == 'r2':
        selected.reverse()
    spec = cfg['engines'][role]
    report = {'status': 'running', 'role': role, 'round': a.round,
              'image_sha256': image, 'engine_sha256': spec['sha256'],
              'script_sha256': q.digest(__file__), 'source_calls': len(cases),
              'unique_shapes': len({signature(c['feeds']) for c in cases}),
              'selected_indices': selected, 'cases': [], 'note': __doc__,
              'warmup': a.warmup, 'repeat_bare': a.repeat, 'repeat_profile': a.profile_repeat,
              'before': common.telemetry()}
    native = None

    def create():
        s = Session(q.RELEASE / spec['path'], spec['sha256'], role,
                    plugins=[(q.RELEASE / p['path'], p['sha256']) for p in spec.get('plugins', [])])
        # Business sorts multi-output bindings. Preserve that order for hashes.
        if set(v.name for v in s.outputs) != set(names):
            raise ValueError('Output names differ')
        s.outputs.sort(key=lambda v: names.index(v.name))
        return s

    try:
        native = create()
        report['context_device_bytes'] = native.engine.device_memory_size
        for index in selected:
            case = cases[index]
            for _ in range(a.warmup):
                verify(native.run(None, case['feeds']), case)
            times = []
            for _ in range(a.repeat):
                start = time.perf_counter()
                out = native.run(None, case['feeds'])
                times.append((time.perf_counter() - start) * 1000)
            verify(out, case)
            report['cases'].append({'source_index': index, **describe(case),
                'bare_wall_ms': times, 'bare_summary': common.summary(times), 'bare_output_exact': True})
        report['after_bare'] = common.telemetry()
        report['bare_execution'] = native.stats()
        native.close()
        native = create()

        class Profiler(native.trt.IProfiler):
            def __init__(self):
                super().__init__()
                self.rows = []

            def report_layer_time(self, name, ms):
                self.rows.append((name, ms))

        profiler = Profiler()
        native.context.profiler = profiler
        native.context.enqueue_emits_profile = True
        for row in report['cases']:
            case = cases[row['source_index']]
            for _ in range(a.warmup):
                verify(native.run(None, case['feeds']), case)
            inspector = native.engine.create_engine_inspector()
            inspector.execution_context = native.context
            info = json.loads(inspector.get_engine_information(native.trt.LayerInformationFormat.JSON))
            row['inspector'] = info
            inspector = None
            profiler.rows.clear()
            for _ in range(a.profile_repeat):
                count = len(profiler.rows)
                out = native.run(None, case['feeds'])
                if len(profiler.rows) <= count:
                    raise RuntimeError('Missing callbacks for a profiled inference')
                verify(out, case)
            row.update(profile=layer_summary(profiler.rows, a.profile_repeat), profile_output_exact=True)
        report.update(status='profiled_exact_not_optimized_or_gt', profile_execution=native.stats(),
                      after=common.telemetry(), io_buffer_bytes=sum(v[1] for v in native.buffers.values()))
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        if native is not None:
            native.close()
        q.save(target, report)
    print(json.dumps({'role': role, 'round': a.round, 'status': report['status'],
                      'means_ms': [r['bare_summary']['mean_ms'] for r in report['cases']],
                      'top_layer_ms': [r['profile']['layers'][0]['mean_ms'] for r in report['cases']]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--round', choices=('r1', 'r2'), required=True)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--repeat', type=int, default=20)
    p.add_argument('--profile-repeat', type=int, default=10)
    a = p.parse_args()
    a.stage = a.stage.resolve()
    if (platform.machine() != 'aarch64' or not a.stage.is_relative_to(q.BASE / 'staging')
            or not a.stage.is_dir() or min(a.warmup, a.repeat, a.profile_repeat) < 1):
        p.error('Existing board staging and positive repeat counts required')
    for role in ROLES:
        common.fresh(a.stage / ('%s_%s.json' % (role, a.round)))
    target = a.stage / ('capture_%s.json' % a.round)
    common.fresh(target)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(q.RELEASE / 'edge_deploy/jetson'))
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE), AUTOOCR_RUNTIME_DIR=str(a.stage / 'runtime'),
                      AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL)
    cases, calls, cfg, image, names, execution = capture(a)
    gc.collect()
    q.save(target, {'script_sha256': q.digest(__file__), 'image_sha256': image,
        'deployment_sha256': q.digest(q.RELEASE / 'edge_deploy/jetson/deployment.json'),
        'calls': calls, 'execution': execution, 'output_names': names,
        'case_metadata': {r: [describe(c) for c in values] for r, values in cases.items()},
        'note': 'Instrumented single ordinary/enhanced passes, not benchmark. Original tensors/outputs in RAM only.'})
    for role in ROLES if a.round == 'r1' else reversed(ROLES):
        measure_role(a, role, cases[role], cfg, names[role], image)
        gc.collect()


if __name__ == '__main__':
    main()
