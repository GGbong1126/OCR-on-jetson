"""Isolated OCR recognition FP16 trial for Jetson TRT8.2, no release changes.

Same original crop tensors and CTC decoder; no padding/trimming or new tolerance
policy. Reports omit recognized text and tensors. No activation/HTTP/counters.
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
import quantize_yolo_ptq as q

SOURCE = q.RELEASE / 'pth/onnx_rec/ocr_rec.onnx'
SOURCE_SHA = 'c93ea0ae08479f3b9a3d8b3549c7a45776f4830a10a430e3d355570d1daec5f4'
PROFILE = [(1, 3, 48, 32), (1, 3, 48, 160), (1, 3, 48, 1536)]


def fresh(path):
    if path.exists():
        raise ValueError('Refuse to overwrite report or engine')


def tokens(output):
    if output.ndim != 3 or output.shape[0] != 1 or not np.isfinite(output).all():
        raise ValueError('Invalid CTC output')
    seq = output.argmax(-1)[0]
    return [int(v) for i, v in enumerate(seq) if v != 0 and (i == 0 or v != seq[i-1])]


def compare(ref, out):
    if ref.shape != out.shape or not np.isfinite(ref).all() or not np.isfinite(out).all():
        raise ValueError('Invalid output contract or nonfinite output')
    error = np.abs(ref - out)
    return {'token_sequence_equal': tokens(ref) == tokens(out),
            'reference_empty': not tokens(ref), 'candidate_empty': not tokens(out),
            'max_abs_error': float(error.max()), 'mae': float(error.mean()),
            'allclose_diagnostic': bool(np.allclose(ref, out, rtol=1e-3, atol=1e-4))}


def summary(values):
    x = np.asarray(values, dtype=np.float64)
    if not x.size or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Invalid timings')
    return {'count': len(values), 'mean_ms': float(x.mean()),
            'p50_ms': float(np.percentile(x, 50)), 'p95_ms': float(np.percentile(x, 95))}


def telemetry():
    values = {}
    for p in Path('/sys/class/devfreq').glob('*/cur_freq'):
        try:
            values[p.parent.name] = int(p.read_text())
        except (OSError, ValueError):
            pass
    return {'memory': q.memory(), 'devfreq_hz': values}


def build(a):
    import tensorrt as trt
    target, engine_path = a.stage / 'build_rec_fp16.json', a.stage / 'rec_fp16.engine'
    fresh(target); fresh(engine_path)
    if q.digest(SOURCE) != SOURCE_SHA:
        raise ValueError('Unexpected recognition model')
    log = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(log)
    if not builder.platform_has_fast_fp16:
        raise RuntimeError('FP16 not supported')
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, log)
    if not parser.parse(SOURCE.read_bytes()):
        raise RuntimeError('; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if network.num_inputs != 1 or network.get_input(0).name != 'x':
        raise ValueError('Unexpected model input')
    config = builder.create_builder_config()
    config.max_workspace_size = 512 * 1024**2
    config.avg_timing_iterations = 8
    config.clear_flag(trt.BuilderFlag.TF32)
    config.clear_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    profile = builder.create_optimization_profile()
    profile.set_shape('x', *PROFILE)
    if [tuple(s) for s in profile.get_shape('x')] != PROFILE:
        raise RuntimeError('Profile differs')
    config.add_optimization_profile(profile)
    report = {'status': 'building', 'source_sha256': SOURCE_SHA, 'profile': PROFILE,
              'script_sha256': q.digest(__file__), 'trt_version': trt.__version__,
              'workspace_bytes': config.max_workspace_size, 'flags': int(config.flags),
              'avg_timing_iterations': config.avg_timing_iterations,
              'tactic_sources': config.get_tactic_sources(),
              'note': 'FP16 enabled with FP32 fallback, external FP32 bindings; actual layers require post-execution Inspector. Not INT8 or pure FP16.'}
    start = time.perf_counter()
    try:
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            raise RuntimeError('Build failed')
        with engine_path.open('xb') as f:
            f.write(bytes(blob))
        report.update(status='built_not_validated', engine_sha256=q.digest(engine_path),
                      engine_bytes=engine_path.stat().st_size)
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        report['build_s'] = time.perf_counter() - start
        q.save(target, report)
    print(json.dumps(report), flush=True)


def candidate(stage):
    from trt82_backend import Session
    built = json.loads((stage / 'build_rec_fp16.json').read_text())
    if built['status'] != 'built_not_validated' or built['source_sha256'] != SOURCE_SHA:
        raise ValueError('Successful source-matched build required')
    class FP16(Session):
        def stats(self):
            return dict(super().stats(), precision='fp16_enabled_fp32_fallback_no_tf32')
    return FP16(stage / 'rec_fp16.engine', built['engine_sha256'], 'rec_fp16_candidate')


def setup(a):
    from serve_jetson import deployment, load_models
    server, root, cfg = deployment()
    pipeline, models = load_models(server, root, cfg)
    return server, pipeline, models, cfg


def close_models(models):
    if models is not None:
        for session in models.models.values():
            getattr(session, 'gpu', session).close()


def inspect(session, stage):
    inspector = session.engine.create_engine_inspector()
    inspector.execution_context = session.context
    layers = json.loads(inspector.get_engine_information(session.trt.LayerInformationFormat.JSON))
    count = sum('half' in json.dumps(r).lower() or 'fp16' in json.dumps(r).lower()
                for r in layers.get('Layers', []))
    if not count or len(layers['Layers']) != session.engine.num_layers:
        raise RuntimeError('Missing FP16 layer evidence')
    q.save(stage / 'rec_fp16_layers.json', layers)
    return {'layers': len(layers['Layers']), 'layers_with_half_format': count}


def check(a):
    target = a.stage / 'check_rec_fp16.json'
    fresh(target); fresh(a.stage / 'rec_fp16_layers.json')
    plan = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
    report = {'status': 'running', 'rows': [], 'source_sha256': SOURCE_SHA,
              'build_sha256': q.digest(a.stage / 'build_rec_fp16.json'),
              'script_sha256': q.digest(__file__),
              'note': 'Same crops from first three calibration images, ordinary and enhanced. Token equality is not GT accuracy; timings instrumented.'}
    models = trial = None
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = setup(a)
            gpu = models.models['rec'].gpu
            original = gpu.run
            trial = candidate(a.stage)
            active = {}

            def shadow(names, feeds):
                reference = original(names, feeds)
                out = trial.run(names, feeds)
                metrics = compare(reference[0], out[0])
                report['rows'].append(dict(active, shape=list(feeds['x'].shape),
                    input_sha256=hashlib.sha256(feeds['x'].tobytes()).hexdigest(), **metrics))
                return reference

            gpu.run = shadow
            for index, image in enumerate(plan['calibration'][:3]):
                for kind in (3, 1):
                    active.update(image_sha256=image['sha256'], image_type=kind)
                    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                        out = pipeline.process(str(Path(plan['images']) / image['file']), image_type=kind, run_ner=True)
                    if not out['success'] or not models.healthy or trial.errors:
                        raise RuntimeError('Backend error during original pipeline')
                    print('Checked calibration %d type %d, crops %d' % (index + 1, kind, len(report['rows'])), flush=True)
            gpu.run = original
            if not report['rows']:
                raise RuntimeError('No compared crops')
            report['inspection'] = inspect(trial, a.stage)
            changed = sum(not r['token_sequence_equal'] for r in report['rows'])
            report.update(status='calibration_text_consistent_not_gt' if not changed else 'text_changes_not_accepted',
                          changed_crops=changed, execution=trial.stats(), baseline_execution=models.stats())
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        if trial is not None:
            trial.close()
        close_models(models)
        q.save(target, report)
    print(json.dumps({'status': report['status'], 'crops': len(report['rows']),
                      'changed': report['changed_crops'], 'inspection': report['inspection']}), flush=True)


def gate(stage):
    checked = json.loads((stage / 'check_rec_fp16.json').read_text())
    if (checked['status'] != 'calibration_text_consistent_not_gt'
            or checked['build_sha256'] != q.digest(stage / 'build_rec_fp16.json')):
        raise ValueError('Candidate calibration regression not passed')


def evaluate(a):
    gate(a.stage)
    target = a.stage / 'evaluate_rec_fp16.json'
    fresh(target)
    plan = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
    report = {'status': 'running', 'rows': [], 'script_sha256': q.digest(__file__),
              'build_sha256': q.digest(a.stage / 'build_rec_fp16.json'),
              'note': 'Five previously held-out images, ordinary/enhanced; paired order alternates. Six other networks fixed FP32, same CPU fallback policy. No GT or HTTP.'}
    models = trial = base = None
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = setup(a)
            rec = models.models['rec']
            base, trial = rec.gpu, candidate(a.stage)
            for index, image in enumerate(plan['holdout']):
                for kind in (3, 1):
                    outputs, durations = {}, {}
                    order = ('reference', 'candidate') if (index + kind) % 2 else ('candidate', 'reference')
                    for name in order:
                        rec.gpu = base if name == 'reference' else trial
                        start = time.perf_counter()
                        with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                            out = pipeline.process(str(Path(plan['images']) / image['file']), image_type=kind, run_ner=True)
                        durations[name] = (time.perf_counter() - start) * 1000
                        if not out['success'] or not models.healthy:
                            raise RuntimeError('Partial pipeline not accepted')
                        outputs[name] = out['data']
                    row = q.strict_compare(outputs['reference'], outputs['candidate'])
                    row['fields'] = q.field_diagnostics(outputs['reference'], outputs['candidate'], server._layout_special_text)
                    row.update(image_sha256=image['sha256'], image_type=kind, order=order, wall_ms=durations)
                    report['rows'].append(row)
                    print(json.dumps({'image': index+1, 'type': kind, 'passed': row['passed']}), flush=True)
            report.update(status='holdout_text_field_consistent_not_gt' if all(r['passed'] for r in report['rows']) else 'differences_not_accepted',
                          candidate_execution=trial.stats(), reference_execution=base.stats(),
                          pipeline_execution=models.stats())
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        if models is not None and base is not None:
            models.models['rec'].gpu = base
        if trial is not None:
            trial.close()
        close_models(models)
        q.save(target, report)
    print(json.dumps({'status': report['status'], 'cases': len(report['rows']), 'passed': sum(r['passed'] for r in report['rows'])}), flush=True)


def benchmark(a):
    gate(a.stage)
    target = a.stage / ('benchmark_rec_%s_%s.json' % (a.variant, a.round))
    fresh(target)
    plan = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
    report = {'status': 'running', 'variant': a.variant, 'round': a.round, 'rows': [],
              'script_sha256': q.digest(__file__), 'build_sha256': q.digest(a.stage / 'build_rec_fp16.json'),
              'note': 'Original in-profile rec tensor sequence from first calibration image, original order/no padding. Raw sync Session.run timing, not full service or GPU kernel. Fresh process per variant, no locked clocks.',
              'warmup_cycles': a.warmup, 'measured_cycles': a.repeat}
    models = native = None
    feeds = []
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = setup(a)
            rec = models.models['rec'].gpu
            original = rec.run
            def capture(names, values):
                feeds.append({k: v.copy() for k, v in values.items()})
                return original(names, values)
            rec.run = capture
            image = plan['calibration'][0]
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                result = pipeline.process(str(Path(plan['images']) / image['file']), image_type=3, run_ner=True)
            if not result['success'] or not models.healthy or not feeds:
                raise RuntimeError('Cannot capture valid original tensor sequence')
            rec.run = original
            close_models(models)
            models = pipeline = rec = original = None
            gc.collect()
        report['before'] = telemetry()
        from trt82_backend import Session
        base = cfg['engines']['rec']
        native = candidate(a.stage) if a.variant == 'fp16' else Session(q.RELEASE / base['path'], base['sha256'], 'rec_fp32_reference')
        report.update(engine_sha256=q.digest(a.stage / 'rec_fp16.engine') if a.variant == 'fp16' else base['sha256'],
                      input_sha256=[hashlib.sha256(f['x'].tobytes()).hexdigest() for f in feeds],
                      input_shapes=[list(f['x'].shape) for f in feeds])
        for _ in range(a.warmup):
            for feed in feeds:
                native.run(None, feed)
        report['after_warmup'] = telemetry()
        for cycle in range(a.repeat):
            for i, feed in enumerate(feeds):
                start = time.perf_counter()
                native.run(None, feed)
                report['rows'].append({'cycle': cycle, 'index': i, 'wall_ms': (time.perf_counter() - start) * 1000})
        report.update(status='measured', timing=summary([r['wall_ms'] for r in report['rows']]),
                      after=telemetry(), execution=native.stats(),
                      context_device_bytes=native.engine.device_memory_size,
                      io_buffer_bytes=sum(v[1] for v in native.buffers.values()))
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        if native is not None:
            native.close()
        close_models(models)
        q.save(target, report)
    print(json.dumps({'variant': a.variant, 'round': a.round, 'timing': report['timing']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('build', 'check', 'evaluate', 'benchmark'))
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--variant', choices=('fp32', 'fp16'), default='fp16')
    p.add_argument('--round', choices=('r1', 'r2'), default='r1')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--repeat', type=int, default=10)
    a = p.parse_args()
    a.stage = a.stage.resolve()
    if (platform.machine() != 'aarch64' or not a.stage.is_relative_to(q.BASE / 'staging')
            or not a.stage.is_dir() or a.warmup < 1 or a.repeat < 1):
        p.error('Use existing isolated board project staging and positive counts')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(q.RELEASE / 'edge_deploy/jetson'))
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE), AUTOOCR_RUNTIME_DIR=str(a.stage / 'runtime'),
                      AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL)
    {'build': build, 'check': check, 'evaluate': evaluate, 'benchmark': benchmark}[a.mode](a)


if __name__ == '__main__':
    main()
