"""E100: explicit FP32 region ablation for recognition, never a release edit.

Keeps exact source, inputs, CTC, fallback policy, and failed E99 evidence.
Regions follow parsed graph order on a hash-locked model, not image/width rules.
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
import optimize_rec_fp16 as old
import quantize_yolo_ptq as q

PRIOR = q.BASE / 'staging/other_models_e99_r1'
ANCHOR = 'ReduceMean.0'
COMPUTE = {'CONVOLUTION', 'DECONVOLUTION', 'ACTIVATION', 'POOLING', 'SCALE',
           'ELEMENTWISE', 'UNARY', 'REDUCE', 'SOFTMAX', 'MATRIX_MULTIPLY', 'FULLY_CONNECTED'}


def selected(index, anchor, policy):
    if policy not in ('tail_fp32', 'prefix_fp32'):
        raise ValueError('Unknown policy')
    return index >= anchor if policy == 'tail_fp32' else index < anchor


def metadata(stage):
    return {'script_sha256': q.digest(__file__), 'helper_sha256': q.digest(old.__file__),
            'build_sha256': q.digest(stage / 'build.json'),
            'deployment_sha256': q.digest(q.RELEASE / 'edge_deploy/jetson/deployment.json')}


def build(a):
    import tensorrt as trt
    target, engine_path = a.stage / 'build.json', a.stage / 'mixed.engine'
    old.fresh(target); old.fresh(engine_path)
    if not trt.__version__.startswith('8.2.1') or q.digest(old.SOURCE) != old.SOURCE_SHA:
        raise ValueError('Unexpected model/runtime')
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(old.SOURCE.read_bytes()):
        raise RuntimeError('; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    anchors = [i for i in range(network.num_layers) if network.get_layer(i).name == ANCHOR]
    if anchors != [355] or network.num_layers != 768:
        raise ValueError('Parsed topology changed; inspect before choosing regions')
    constrained = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        kind = str(layer.type).split('.')[-1]
        outputs = [j for j in range(layer.num_outputs)
                   if layer.get_output(j) is not None
                   and layer.get_output(j).dtype == trt.float32
                   and not layer.get_output(j).is_shape_tensor]
        if not selected(i, anchors[0], a.policy) or not outputs:
            continue
        # Shape tensors stay integer. Reordering/constants get output constraints,
        # not a fictitious arithmetic precision; arithmetic also gets compute FP32.
        compute = kind in COMPUTE
        if compute:
            layer.precision = trt.float32
        for j in outputs:
            layer.set_output_type(j, trt.float32)
            if layer.get_output_type(j) != trt.float32:
                raise RuntimeError('Output constraint not retained')
        constrained.append({'index': i, 'name': layer.name, 'kind': kind,
                            'compute_fp32': compute, 'fp32_output_indices': outputs})
    if not constrained:
        raise RuntimeError('Empty precision policy')
    config = builder.create_builder_config()
    config.max_workspace_size = 512 * 1024**2
    config.avg_timing_iterations = 8
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    config.clear_flag(trt.BuilderFlag.INT8)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    profile = builder.create_optimization_profile()
    profile.set_shape('x', *old.PROFILE)
    if [tuple(s) for s in profile.get_shape('x')] != old.PROFILE:
        raise RuntimeError('Profile mismatch')
    config.add_optimization_profile(profile)
    timing_cache = config.create_timing_cache(b'')
    config.set_timing_cache(timing_cache, False)
    report = {'status': 'building', 'policy': a.policy, 'anchor': ANCHOR, 'anchor_index': anchors[0],
              'source_sha256': old.SOURCE_SHA, 'script_sha256': q.digest(__file__),
              'profile': old.PROFILE, 'trt_version': trt.__version__,
              'workspace_bytes': config.max_workspace_size, 'avg_timing_iterations': config.avg_timing_iterations,
              'flags': int(config.flags), 'tactic_sources': config.get_tactic_sources(),
              'constraints': constrained,
              'note': 'Controlled precision-region hypothesis, not proven localization. FP16 allowed elsewhere; FP32 I/O. No INT8/calibration/padding.'}
    start = time.perf_counter()
    try:
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            raise RuntimeError('Mixed build failed')
        with engine_path.open('xb') as f:
            f.write(bytes(blob))
        with (a.stage / 'timing.cache').open('xb') as f:
            f.write(bytes(config.get_timing_cache().serialize()))
        report.update(status='built_not_validated', engine_sha256=q.digest(engine_path),
                      engine_bytes=engine_path.stat().st_size, cache_sha256=q.digest(a.stage / 'timing.cache'))
    except Exception as e:
        report.update(status='failed', error=type(e).__name__ + ': ' + str(e))
        raise
    finally:
        report['build_s'] = time.perf_counter() - start
        q.save(target, report)
    print(json.dumps({k: report[k] for k in ('status', 'policy', 'engine_bytes', 'engine_sha256', 'build_s')}), flush=True)


def candidate(stage):
    from trt82_backend import Session
    report = json.loads((stage / 'build.json').read_text())
    if report['status'] != 'built_not_validated' or report['source_sha256'] != old.SOURCE_SHA:
        raise ValueError('No valid source-matched build')
    class Mixed(Session):
        def stats(self):
            return dict(super().stats(), precision='mixed_' + report['policy'])
    return Mixed(stage / 'mixed.engine', report['engine_sha256'], 'rec_mixed')


def probe(a):
    target = a.stage / 'probe.json'
    old.fresh(target); old.fresh(a.stage / 'layers.json')
    prior = json.loads((PRIOR / 'rec_difference_diagnostic.json').read_text())
    bad = prior['rows'][0]['input_sha256']
    plan = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
    image = next(r for r in plan['holdout'] if r['sha256'] == prior['image_sha256'])
    report = dict(metadata(a.stage), status='running', expected_input_sha256=bad,
                  prior_diagnostic_sha256=q.digest(PRIOR / 'rec_difference_diagnostic.json'))
    models = native = None
    captured = []
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = old.setup(a)
            rec = models.models['rec'].gpu
            original = rec.run
            def capture(names, feeds):
                result = original(names, feeds)
                if hashlib.sha256(feeds['x'].tobytes()).hexdigest() == bad:
                    captured.append(({'x': feeds['x'].copy()}, result[0].copy()))
                return result
            rec.run = capture
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                result = pipeline.process(str(Path(plan['images']) / image['file']),
                                          image_type=prior['image_type'], run_ner=True)
            rec.run = original
            if not result['success'] or not models.healthy or len(captured) != 1:
                raise RuntimeError('Exact failed tensor not reproduced')
            old.close_models(models)
            models = pipeline = rec = original = None
            gc.collect()
        feed, ref = captured[0]
        native = candidate(a.stage)
        out = native.run(None, feed)[0]
        again = native.run(None, feed)[0]
        report.update(comparison=old.compare(ref, out), repeat_exact=bool(np.array_equal(out, again)),
                      input_shape=list(feed['x'].shape))
        inspector = native.engine.create_engine_inspector()
        inspector.execution_context = native.context
        layers = json.loads(inspector.get_engine_information(native.trt.LayerInformationFormat.JSON))
        if not layers.get('Layers') or len(layers['Layers']) != native.engine.num_layers:
            raise RuntimeError('Incomplete inspector')
        q.save(a.stage / 'layers.json', layers)
        report['inspection'] = {'layers': len(layers['Layers']),
            'half_format_layers': sum('fp16' in json.dumps(r).lower() for r in layers['Layers']),
            'float_format_layers': sum('fp32' in json.dumps(r).lower() for r in layers['Layers'])}
        if not report['inspection']['half_format_layers'] or not report['inspection']['float_format_layers']:
            raise RuntimeError('Missing mixed format evidence')
        report.update(status='failed_case_recovered_not_general_acceptance'
                      if report['comparison']['token_sequence_equal'] else 'failed_case_still_changed',
                      execution=native.stats())
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        try:
            if native is not None:
                native.close()
            old.close_models(models)
        finally:
            q.save(target, report)
    print(json.dumps(report), flush=True)


def require_probe(stage):
    p = json.loads((stage / 'probe.json').read_text())
    if (p['status'] != 'failed_case_recovered_not_general_acceptance' or not p['repeat_exact']
            or p['build_sha256'] != q.digest(stage / 'build.json')):
        raise ValueError('Exact failure probe has not passed')


def evaluate(a):
    require_probe(a.stage)
    target = a.stage / 'evaluation.json'
    old.fresh(target)
    plan = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
    groups = [('development_e99_cal', plan['calibration'][:3]),
              ('development_e99_holdout', plan['holdout']),
              ('additional_not_used_by_e99_rec', plan['calibration'][3:])]
    report = dict(metadata(a.stage), status='running', rows=[],
                  note='13 convenience images x ordinary/enhanced. Five additional images not used by E99 rec, but used by earlier YOLO trials; not a globally pristine test set or GT accuracy. Text gate unchanged.')
    models = native = base = None
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = old.setup(a)
            rec = models.models['rec']
            base, native = rec.gpu, candidate(a.stage)
            for group, images in groups:
                for image in images:
                    for kind in (3, 1):
                        outputs, wall = {}, {}
                        order = ('reference', 'candidate') if len(report['rows']) % 2 == 0 else ('candidate', 'reference')
                        for name in order:
                            rec.gpu = base if name == 'reference' else native
                            start = time.perf_counter()
                            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                                result = pipeline.process(str(Path(plan['images']) / image['file']), image_type=kind, run_ner=True)
                            wall[name] = (time.perf_counter() - start) * 1000
                            if not result['success'] or not models.healthy:
                                raise RuntimeError('Pipeline/optional backend error')
                            outputs[name] = result['data']
                        row = q.strict_compare(outputs['reference'], outputs['candidate'])
                        row.update(group=group, image_sha256=image['sha256'], image_type=kind, wall_ms=wall, order=order,
                                   field_diagnostics=q.field_diagnostics(outputs['reference'], outputs['candidate'], server._layout_special_text))
                        report['rows'].append(row)
                        print(json.dumps({'case': len(report['rows']), 'group': group, 'type': kind, 'passed': row['passed']}), flush=True)
            report.update(status='all_26_consistent_not_gt' if len(report['rows']) == 26 and all(r['passed'] for r in report['rows']) else 'differences_not_accepted',
                          candidate_execution=native.stats(), reference_execution=base.stats(), all_execution=models.stats())
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        try:
            if models is not None and base is not None:
                models.models['rec'].gpu = base
            if native is not None:
                native.close()
            old.close_models(models)
        finally:
            q.save(target, report)
    print(json.dumps({'status': report['status'], 'cases': len(report['rows']), 'passed': sum(r['passed'] for r in report['rows'])}), flush=True)


def benchmark(a):
    require_probe(a.stage)
    quality = json.loads((a.stage / 'evaluation.json').read_text())
    if quality['status'] != 'all_26_consistent_not_gt' or quality['build_sha256'] != q.digest(a.stage / 'build.json'):
        raise ValueError('Business regression not passed')
    target = a.stage / ('benchmark_%s_%s.json' % (a.variant, a.round))
    old.fresh(target)
    plan = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
    report = dict(metadata(a.stage), status='running', variant=a.variant, round=a.round, rows=[],
                  warmup_cycles=a.warmup, measured_cycles=a.repeat,
                  note='Original in-profile 50-crop sequence from first calibration image, no padding; raw synchronous Session.run, not HTTP or whole OCR. Fresh process per variant, no clock changes.')
    models = native = None
    feeds = []
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = old.setup(a)
            rec = models.models['rec'].gpu
            original = rec.run
            def capture(names, values):
                feeds.append({k: v.copy() for k, v in values.items()})
                return original(names, values)
            rec.run = capture
            image = plan['calibration'][0]
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                out = pipeline.process(str(Path(plan['images']) / image['file']), image_type=3, run_ner=True)
            rec.run = original
            if not out['success'] or not models.healthy or len(feeds) != 50:
                raise RuntimeError('Original benchmark input sequence differs')
            old.close_models(models)
            models = pipeline = rec = original = None
            gc.collect()
        report['before'] = old.telemetry()
        from trt82_backend import Session
        row = cfg['engines']['rec']
        native = candidate(a.stage) if a.variant == 'mixed' else Session(q.RELEASE / row['path'], row['sha256'], 'rec_fp32_reference')
        report.update(input_sha256=[hashlib.sha256(f['x'].tobytes()).hexdigest() for f in feeds],
                      input_shapes=[list(f['x'].shape) for f in feeds])
        for _ in range(a.warmup):
            for feed in feeds:
                native.run(None, feed)
        report['after_warmup'] = old.telemetry()
        for cycle in range(a.repeat):
            for index, feed in enumerate(feeds):
                start = time.perf_counter()
                native.run(None, feed)
                report['rows'].append({'cycle': cycle, 'index': index, 'wall_ms': (time.perf_counter()-start)*1000})
        report.update(status='measured', timing=old.summary([r['wall_ms'] for r in report['rows']]),
                      execution=native.stats(), after=old.telemetry(), context_device_bytes=native.engine.device_memory_size)
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        try:
            if native is not None:
                native.close()
            old.close_models(models)
        finally:
            q.save(target, report)
    print(json.dumps({'variant': a.variant, 'round': a.round, 'timing': report['timing']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('build', 'probe', 'evaluate', 'benchmark'))
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--policy', choices=('tail_fp32', 'prefix_fp32'), default='tail_fp32')
    p.add_argument('--variant', choices=('fp32', 'mixed'), default='mixed')
    p.add_argument('--round', choices=('r1', 'r2'), default='r1')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--repeat', type=int, default=10)
    a = p.parse_args()
    a.stage = a.stage.resolve()
    if (platform.machine() != 'aarch64' or not a.stage.is_relative_to(q.BASE / 'staging')
            or not a.stage.is_dir() or a.stage == PRIOR or a.warmup < 1 or a.repeat < 1):
        p.error('Use a separate existing project staging directory and positive counts')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(q.RELEASE / 'edge_deploy/jetson'))
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE), AUTOOCR_RUNTIME_DIR=str(a.stage / 'runtime'), AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL)
    {'build': build, 'probe': probe, 'evaluate': evaluate, 'benchmark': benchmark}[a.mode](a)


if __name__ == '__main__':
    main()
