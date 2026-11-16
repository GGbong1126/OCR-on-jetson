"""E97: matched FP32 control and separate, non-production TRT8.2 profiling.

Keeps E92 calibration, engines, releases and activation untouched. The built-in
profiler adds overhead; its timings are diagnostics, never benchmark results.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import quantize_yolo_ptq as q

PRIOR = q.BASE / 'staging/quant_yolo_e92_r1'
INT8_SHA = 'd1a1342d85f3496b6b4f6a92f2d5384d95c35bf9b03bb64110609b02d4d6c92b'


def fresh(path):
    if path.exists():
        raise ValueError('Refuse to overwrite: ' + str(path))


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid timing samples')
    return {'count': int(values.size), 'mean_ms': float(values.mean()),
            'p50_ms': float(np.percentile(values, 50)), 'p95_ms': float(np.percentile(values, 95))}


def snapshot():
    """Read-only endpoint telemetry, not a peak/frequency trace."""
    result = {'memory': q.memory(), 'devfreq_hz': {}, 'thermal_millicelsius': {}}
    for path in Path('/sys/class/devfreq').glob('*/cur_freq'):
        try:
            result['devfreq_hz'][path.parent.name] = int(path.read_text())
        except (OSError, ValueError):
            pass
    for path in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
        try:
            name = (path.parent / 'type').read_text().strip()
            result['thermal_millicelsius'][name] = int(path.read_text())
        except (OSError, ValueError):
            pass
    return result


def artifact_name(mode, variant, pattern, index, round_name):
    workload = pattern if pattern == 'mixed' else 'fixed%d' % index
    return '%s_%s_%s_%s.json' % (mode, variant, workload, round_name)


def category(name, layer):
    kind = layer.get('LayerType', 'unmapped')
    if 'reformat' in name.lower() or kind.lower() == 'reformat':
        return 'Reformat'
    return kind


def aggregate(callbacks, layers, calls):
    if calls < 1 or not callbacks:
        raise ValueError('No measured callbacks')
    lookup = {row['Name']: row for row in layers}
    times, counts = defaultdict(float), defaultdict(int)
    for name, ms in callbacks:
        if not np.isfinite(ms) or ms < 0:
            raise ValueError('Invalid layer timing')
        times[name] += float(ms)
        counts[name] += 1
    rows = []
    for name, total in times.items():
        layer = lookup.get(name, {})
        rows.append({'name': name, 'category': category(name, layer), 'callbacks': counts[name],
                     'mean_ms_per_inference': total / calls,
                     'has_int8_format': 'int8' in json.dumps(layer).lower()})
    groups = defaultdict(float)
    for row in rows:
        groups[row['category']] += row['mean_ms_per_inference']
    return {'layer_rows': sorted(rows, key=lambda r: r['mean_ms_per_inference'], reverse=True),
            'category_ms_per_inference': dict(groups),
            'summed_layer_ms_per_inference': sum(times.values()) / calls,
            'unmapped_callback_names': [name for name in times if name not in lookup]}


def provenance():
    plan = q.load_plan(PRIOR)
    inspected = json.loads((PRIOR / 'inspection.json').read_text())
    if (inspected['engine_sha256'] != INT8_SHA
            or q.digest(PRIOR / 'yolo_int8_fp32.engine') != INT8_SHA):
        raise ValueError('E92 candidate changed')
    return plan, {'plan_sha256': q.digest(PRIOR / 'plan.json'), 'int8_sha256': INT8_SHA,
                  'script_sha256': q.digest(__file__), 'helper_sha256': q.digest(q.__file__)}


def build(a):
    import tensorrt as trt
    _, meta = provenance()
    target = a.stage / 'build_fp32.json'
    engine_path = a.stage / 'yolo_fp32_ws512.engine'
    fresh(target); fresh(engine_path)
    if shutil.disk_usage(a.stage).free < 400 * 1024**2:
        raise RuntimeError('Less than 400MiB disk reserve; no cleanup authorized')
    source = q.BASE / 'staging/yolo_onnx_r3/yolo.onnx'
    if q.digest(source) != q.SOURCE_HASH:
        raise ValueError('Source changed')
    log = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(log)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, log)
    if not parser.parse(source.read_bytes()):
        raise RuntimeError('; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if network.num_inputs != 1 or network.get_input(0).name != 'images':
        raise ValueError('Unexpected input contract')
    config = builder.create_builder_config()
    config.max_workspace_size = 512 * 1024**2
    for flag in (trt.BuilderFlag.FP16, trt.BuilderFlag.INT8, trt.BuilderFlag.TF32):
        config.clear_flag(flag)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    profile = builder.create_optimization_profile()
    profile.set_shape('images', *q.PROFILE)
    if [tuple(x) for x in profile.get_shape('images')] != q.PROFILE:
        raise RuntimeError('Profile mismatch')
    config.add_optimization_profile(profile)
    report = dict(meta, status='building', onnx_sha256=q.SOURCE_HASH, profile=q.PROFILE,
                  trt_version=trt.__version__, workspace_bytes=config.max_workspace_size,
                  flags=int(config.flags), profiling_verbosity=str(config.profiling_verbosity),
                  avg_timing_iterations=config.avg_timing_iterations,
                  tactic_sources=config.get_tactic_sources(), precision='fp32_no_tf32',
                  note='Matches E92 workspace512, profile, detailed verbosity, same source/runtime; no calibration needed for FP32. Build timing/tactic variability remains.')
    start = time.perf_counter()
    try:
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            raise RuntimeError('Engine build failed')
        with engine_path.open('xb') as f:
            f.write(bytes(blob))
        report.update(status='built_pending_numeric_check', engine_sha256=q.digest(engine_path),
                      engine_bytes=engine_path.stat().st_size)
    except Exception as e:
        report.update(status='failed', error=type(e).__name__ + ': ' + str(e))
        raise
    finally:
        report['build_seconds'] = time.perf_counter() - start
        q.save(target, report)
        print(json.dumps(report), flush=True)


def session(a):
    from trt82_backend import Session
    if a.variant == 'int8':
        return q.candidate(SimpleNamespace(stage=PRIOR))
    if a.variant == 'original':
        cfg = json.loads((q.RELEASE / 'edge_deploy/jetson/deployment.json').read_text())['engines']['yolo']
        return Session(q.RELEASE / cfg['path'], cfg['sha256'], 'original_fp32')
    report = json.loads((a.stage / 'build_fp32.json').read_text())
    if report['status'] != 'built_pending_numeric_check':
        raise ValueError('No successful control build')
    return Session(a.stage / 'yolo_fp32_ws512.engine', report['engine_sha256'], 'matched_fp32')


def feeds(plan):
    return [{'images': q.tensor_from_rgb(q.business_rgb(Path(plan['images']) / r['file']))}
            for r in plan['holdout']]


def check(a):
    plan, meta = provenance()
    target = a.stage / 'check_fp32.json'
    fresh(target)
    tensors = feeds(plan)
    reference = session(SimpleNamespace(stage=a.stage, variant='original'))
    trial = session(SimpleNamespace(stage=a.stage, variant='matched'))
    report = dict(meta, status='running', rows=[], build_report_sha256=q.digest(a.stage / 'build_fp32.json'))
    try:
        for feed in tensors:
            ref, out = reference.run(None, feed)[0], trial.run(None, feed)[0]
            if ref.shape != out.shape:
                raise ValueError('Output shape differs')
            error = np.abs(ref - out)
            report['rows'].append({'input_shape': list(feed['images'].shape),
                'input_sha256': hashlib.sha256(feed['images'].tobytes()).hexdigest(),
                'output_shape': list(out.shape), 'max_abs_error': float(error.max()),
                'mae': float(error.mean()), 'allclose': bool(np.allclose(ref, out, rtol=1e-4, atol=1e-4))})
        report.update(status='numeric_control_checked' if all(r['allclose'] for r in report['rows']) else 'numeric_drift_not_accepted',
                      reference=reference.stats(), matched=trial.stats())
    finally:
        reference.close(); trial.close()
        q.save(target, report)
    print(json.dumps(report), flush=True)


def pooled(a):
    """Bounded exact-shape session routing; no padding, no rebuilt INT8 model.

    Two Sessions own two engines/contexts, so the memory cost is measured, not
    presented as free. Equality probes precede/follow the timed pass.
    """
    plan, meta = provenance()
    target = a.stage / artifact_name('pooled', a.variant, 'mixed', 0, a.round)
    fresh(target)
    if a.variant == 'matched':
        gate = json.loads((a.stage / 'check_fp32.json').read_text())
        if (gate['status'] != 'numeric_control_checked'
                or gate['build_report_sha256'] != q.digest(a.stage / 'build_fp32.json')):
            raise ValueError('Matched control check not passed')
    tensors = feeds(plan)
    keys = [tuple(t['images'].shape) for t in tensors]
    if len(set(keys)) != 2:
        raise ValueError('This bounded diagnostic expects exactly two observed shapes')
    report = dict(meta, status='running', variant=a.variant, rows=[], before=q.memory(),
                  matched_build_sha256=q.digest(a.stage / 'build_fp32.json'),
                  warmup_cycles=a.warmup, measured_cycles=a.repeat,
                  input_shapes=[list(k) for k in keys],
                  input_sha256=[hashlib.sha256(t['images'].tobytes()).hexdigest() for t in tensors],
                  note='Two independently owned engine/context Sessions, exact input shapes, no padding. Route+Session.run timed without profiler; exact output probes outside timed region. NOT quantization-only gain or business/GT acceptance.')
    reference, pool = None, {}
    try:
        reference = session(a)
        expected = [reference.run(None, t)[0] for t in tensors]
        reference.close()
        reference = None
        for key in dict.fromkeys(keys):
            pool[key] = session(a)

        def verify():
            equal = []
            for key, t, ref in zip(keys, tensors, expected):
                out = pool[key].run(None, t)
                equal.append(len(out) == 1 and np.array_equal(ref, out[0]))
            if not all(equal):
                raise ValueError('Pooled output differs from same-engine single session')
            return equal

        report['exact_equal_before'] = verify()
        for _ in range(a.warmup):
            for key, t in zip(keys, tensors):
                pool[key].run(None, t)
        report['after_warmup'] = q.memory()
        report['telemetry_start'] = snapshot()
        for cycle in range(a.repeat):
            for i, t in enumerate(tensors):
                start = time.perf_counter()
                pool[tuple(t['images'].shape)].run(None, t)
                report['rows'].append({'cycle': cycle, 'input_index': i, 'wall_ms': (time.perf_counter() - start) * 1000})
        report['telemetry_end'] = snapshot()
        report['exact_equal_after'] = verify()
        report.update(status='exact_shape_route_measured_not_deployed',
                      timing=summary([r['wall_ms'] for r in report['rows']]),
                      execution={str(k): v.stats() for k, v in pool.items()},
                      context_bytes=sum(s.engine.device_memory_size for s in pool.values()))
    except Exception as e:
        report.update(status='failed', error=type(e).__name__ + ': ' + str(e))
        raise
    finally:
        if reference is not None:
            reference.close()
        for s in pool.values():
            s.close()
        q.save(target, report)
    print(json.dumps({k: report[k] for k in ('status', 'variant', 'timing', 'before', 'after_warmup')}), flush=True)


def measure(a):
    plan, meta = provenance()
    target = a.stage / artifact_name(a.mode, a.variant, a.pattern, a.index, a.round)
    fresh(target)
    if a.variant == 'matched':
        gate = json.loads((a.stage / 'check_fp32.json').read_text())
        if (gate['status'] != 'numeric_control_checked'
                or gate['build_report_sha256'] != q.digest(a.stage / 'build_fp32.json')):
            raise ValueError('Matched control check not passed')
    tensors = feeds(plan)
    indices = list(range(len(tensors))) if a.pattern == 'mixed' else [a.index] * len(tensors)
    before = q.memory()
    native = session(a)
    report = dict(meta, status='running', variant=a.variant, pattern=a.pattern, index=a.index,
                  before=before, rows=[], warmup_cycles=a.warmup, measured_cycles=a.repeat,
                  profiler_enabled=a.mode == 'profile', input_indices=indices,
                  input_shapes=[list(t['images'].shape) for t in tensors],
                  input_sha256=[hashlib.sha256(t['images'].tobytes()).hexdigest() for t in tensors],
                  note='Normal Session.run benchmark excludes decode/pre/post; separate profiler diagnostic adds instrumentation overhead. No GT/HTTP/whole-service claim. No clock changes.')
    report['matched_build_sha256'] = q.digest(a.stage / 'build_fp32.json')
    profiler = None
    try:
        for _ in range(a.warmup):
            for i in indices:
                native.run(None, tensors[i])
        report['after_warmup'] = q.memory()
        report['telemetry_start'] = snapshot()
        layer_info = None
        if a.mode == 'profile':
            inspector = native.engine.create_engine_inspector()
            inspector.execution_context = native.context
            layer_info = json.loads(inspector.get_engine_information(native.trt.LayerInformationFormat.JSON))
            if len(layer_info.get('Layers', [])) != native.engine.num_layers:
                raise ValueError('Incomplete layer inspector')

            class Profiler(native.trt.IProfiler):
                def __init__(self):
                    super().__init__()
                    self.rows = []

                def report_layer_time(self, name, ms):
                    self.rows.append((name, ms))

            profiler = Profiler()
            native.context.profiler = profiler
            native.context.enqueue_emits_profile = True
            # Instrumented warmup separately, discarded from measured callbacks.
            for i in indices:
                native.run(None, tensors[i])
            profiler.rows.clear()
            q.save(a.stage / (target.stem + '_layers.json'), layer_info)
        for cycle in range(a.repeat):
            for i in indices:
                start = time.perf_counter()
                native.run(None, tensors[i])
                report['rows'].append({'cycle': cycle, 'input_index': i, 'wall_ms': (time.perf_counter() - start) * 1000})
        report.update(status='measured', timing=summary([r['wall_ms'] for r in report['rows']]),
                      execution=native.stats(), after_measurement=q.memory(),
                      context_bytes=native.engine.device_memory_size, telemetry_end=snapshot())
        if profiler is not None:
            report['profile'] = aggregate(profiler.rows, layer_info['Layers'], len(report['rows']))
    except Exception as e:
        report.update(status='failed', error=type(e).__name__ + ': ' + str(e))
        raise
    finally:
        native.close()
        q.save(target, report)
    print(json.dumps({k: report[k] for k in ('status', 'variant', 'pattern', 'profiler_enabled', 'timing')}), flush=True)
    if profiler is not None:
        print(json.dumps({'layer_categories': report['profile']['category_ms_per_inference'],
                          'layer_sum_ms': report['profile']['summed_layer_ms_per_inference']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('build', 'check', 'benchmark', 'profile', 'pooled'))
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--variant', choices=('matched', 'int8', 'original'), default='matched')
    p.add_argument('--pattern', choices=('mixed', 'fixed'), default='mixed')
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--round', choices=('r1', 'r2'), default='r1')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeat', type=int, default=20)
    a = p.parse_args()
    a.stage = a.stage.resolve()
    if (platform.machine() != 'aarch64' or not a.stage.is_relative_to(q.BASE / 'staging')
            or not a.stage.is_dir() or a.stage == PRIOR or a.warmup < 1 or a.repeat < 1 or not 0 <= a.index < 5):
        p.error('Use new isolated board staging directory and valid sampling parameters')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(q.RELEASE / 'edge_deploy/jetson'))
    (build if a.mode == 'build' else check if a.mode == 'check' else pooled if a.mode == 'pooled' else measure)(a)


if __name__ == '__main__':
    main()
