"""Bounded short-width FP32 candidate; keep original broad engine and CPU policy.

Only an isolated experiment. No padding, resizing, dictionary, CTC or release
changes. The two-engine memory cost must be measured, not described as free.
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

PROFILE = [(1, 3, 48, 32), (1, 3, 48, 128), (1, 3, 48, 256)]
DEPLOYMENT = '2ced56d7d35e429199d51282eaaf961d1408bf2bc0f2c5784fcfb2b0fecba6d8'


def metadata(stage):
    if q.digest(q.RELEASE / 'edge_deploy/jetson/deployment.json') != DEPLOYMENT:
        raise ValueError('Baseline deployment changed')
    built = json.loads((stage / 'build.json').read_text())
    if (built['status'] != 'built_not_validated' or built['source_sha256'] != old.SOURCE_SHA
            or built['script_sha256'] != q.digest(__file__) or built['profile'] != [list(s) for s in PROFILE]):
        raise ValueError('Candidate provenance mismatch')
    return built


def build(a):
    import tensorrt as trt
    target, engine = a.stage / 'build.json', a.stage / 'rec_short_fp32.engine'
    old.fresh(target); old.fresh(engine)
    if (q.digest(old.SOURCE) != old.SOURCE_SHA
            or q.digest(q.RELEASE / 'edge_deploy/jetson/deployment.json') != DEPLOYMENT):
        raise ValueError('Unexpected source/deployment')
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(old.SOURCE.read_bytes()):
        raise RuntimeError('; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if network.num_inputs != 1 or network.get_input(0).name != 'x':
        raise ValueError('Unexpected input')
    config = builder.create_builder_config()
    config.max_workspace_size = 512 * 1024**2
    config.avg_timing_iterations = 8
    for flag in (trt.BuilderFlag.FP16, trt.BuilderFlag.INT8, trt.BuilderFlag.TF32):
        config.clear_flag(flag)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    profile = builder.create_optimization_profile()
    profile.set_shape('x', *PROFILE)
    if [tuple(s) for s in profile.get_shape('x')] != PROFILE:
        raise ValueError('Profile not retained')
    config.add_optimization_profile(profile)
    cache = config.create_timing_cache(b'')
    config.set_timing_cache(cache, False)
    report = {'status': 'building', 'script_sha256': q.digest(__file__),
              'source_sha256': old.SOURCE_SHA, 'deployment_sha256': DEPLOYMENT,
              'profile': PROFILE, 'trt_version': trt.__version__, 'flags': int(config.flags),
              'workspace_bytes': config.max_workspace_size, 'avg_timing_iterations': 8,
              'tactic_sources': config.get_tactic_sources(),
              'note': 'FP32/noTF32, narrow profile and workspace512 candidate. No quantization or model surgery. A configuration trial, not pure single-variable attribution.'}
    start = time.perf_counter()
    try:
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            raise RuntimeError('Build failed')
        with engine.open('xb') as f:
            f.write(bytes(blob))
        with (a.stage / 'timing.cache').open('xb') as f:
            f.write(bytes(config.get_timing_cache().serialize()))
        report.update(status='built_not_validated', engine_sha256=q.digest(engine), engine_bytes=engine.stat().st_size)
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        report['build_s'] = time.perf_counter() - start
        q.save(target, report)
    print(json.dumps(report), flush=True)


def candidate(stage):
    from trt82_backend import Session
    built = metadata(stage)
    return Session(stage / 'rec_short_fp32.engine', built['engine_sha256'], 'rec_short_fp32')


class Route:
    """Exactly two Sessions, serial original-tensor routing, no error fallback."""
    def __init__(self, base, short):
        self.base, self.short = base, short

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def errors(self):
        return self.base.errors + self.short.errors

    def run(self, names, feeds):
        if self.errors or self.base.closed or self.short.closed:
            raise RuntimeError('Recognition route unhealthy')
        # Actual profile contracts, never a width guessed from filenames/text.
        if not self.base.in_profile(feeds):
            raise ValueError('Outer OCRSession must handle original CPU fallback')
        return (self.short if self.short.in_profile(feeds) else self.base).run(names, feeds)

    def stats(self):
        a, b = self.base.stats(), self.short.stats()
        return dict(a, backend='tensorrt82_fp32_two_engine_route',
                    calls=a['calls'] + b['calls'], errors=self.errors,
                    short_execution=b, broad_execution=a)

    def close(self):
        try:
            self.short.close()
        finally:
            self.base.close()


def report_meta(a):
    built = metadata(a.stage)
    return {'status': 'running', 'script_sha256': q.digest(__file__),
            'build_sha256': q.digest(a.stage / 'build.json'), 'engine_sha256': built['engine_sha256'],
            'deployment_sha256': DEPLOYMENT}


def check(a):
    target = a.stage / 'check.json'
    old.fresh(target)
    report = dict(report_meta(a), rows=[])
    plan = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
    models = trial = base = original = None
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = old.setup(a)
            base, trial = models.models['rec'].gpu, candidate(a.stage)
            original = base.run
            active = {}
            def run(names, feeds):
                ref = original(names, feeds)[0]
                if trial.in_profile(feeds):
                    out = trial.run(names, feeds)[0]
                    report['rows'].append(dict(active, shape=list(feeds['x'].shape),
                        input_sha256=hashlib.sha256(feeds['x'].tobytes()).hexdigest(), **old.compare(ref, out)))
                return [ref]
            base.run = run
            for image in plan['calibration'][:3]:
                for kind in (3, 1):
                    active.update(image_sha256=image['sha256'], image_type=kind)
                    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                        out = pipeline.process(str(Path(plan['images']) / image['file']), image_type=kind, run_ner=True)
                    if not out['success'] or not models.healthy or trial.errors:
                        raise RuntimeError('Capture/backend failed')
                    print('Checked',len(report['rows']),flush=True)
            report.update(status='short_ctc_consistent_not_gt' if report['rows'] and all(r['token_sequence_equal'] for r in report['rows']) else 'text_changes_not_accepted',
                          execution=trial.stats(), baseline_execution=models.stats())
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        if original is not None:
            base.run = original
        if trial is not None:
            trial.close()
        old.close_models(models)
        q.save(target, report)
    print(json.dumps({'status':report['status'], 'short_crops':len(report['rows'])}),flush=True)


def require(stage, file, status):
    result = json.loads((stage / file).read_text())
    if result['status'] != status or result['build_sha256'] != q.digest(stage / 'build.json'):
        raise ValueError('Required quality gate not passed')


def business_passed(row):
    return (all(row['strict_equal'][k] for k in ('FullText','Text','MarkdownBodyLines','NER','raw_line_texts'))
            and all(row['fields']['api_special_text_equal'][k] for k in ('title','category_no','page_no')))


def evaluate(a):
    require(a.stage, 'check.json', 'short_ctc_consistent_not_gt')
    target = a.stage / 'evaluation.json'
    old.fresh(target)
    report = dict(report_meta(a), rows=[], note='All 13 existing images ordinary/enhanced. Not independent GT. Text/fields strict, geometry diagnostic only. No new normalization.')
    plan = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
    models = trial = base = None
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = old.setup(a)
            rec = models.models['rec']
            base, trial = rec.gpu, candidate(a.stage)
            route = Route(base, trial)
            for image in plan['calibration'] + plan['holdout']:
                for kind in (3, 1):
                    outputs = {}
                    order = ('reference','candidate') if len(report['rows']) % 2 == 0 else ('candidate','reference')
                    for name in order:
                        rec.gpu = base if name == 'reference' else route
                        with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                            out = pipeline.process(str(Path(plan['images']) / image['file']), image_type=kind, run_ner=True)
                        if not out['success'] or not models.healthy:
                            raise RuntimeError('Business/backend error')
                        outputs[name] = out['data']
                    row = q.strict_compare(outputs['reference'], outputs['candidate'])
                    row.update(fields=q.field_diagnostics(outputs['reference'], outputs['candidate'], server._layout_special_text),
                               image_sha256=image['sha256'], image_type=kind, order=order,
                               strict_objects_passed=row['passed'])
                    row['passed'] = business_passed(row)
                    report['rows'].append(row)
                    print(json.dumps({'case':len(report['rows']),'passed':row['passed']}),flush=True)
            report.update(status='all_26_business_consistent_not_gt' if len(report['rows']) == 26 and all(r['passed'] for r in report['rows']) else 'differences_not_accepted',
                          short_execution=trial.stats(), broad_execution=base.stats(), execution=models.stats())
            if not trial.calls or trial.errors:
                raise RuntimeError('Short candidate not used or unhealthy')
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        if models is not None and base is not None:
            models.models['rec'].gpu = base
        if trial is not None:
            trial.close()
        old.close_models(models)
        q.save(target, report)
    print(json.dumps({'status':report['status'],'cases':len(report['rows'])}),flush=True)


def benchmark(a):
    require(a.stage, 'evaluation.json', 'all_26_business_consistent_not_gt')
    target = a.stage / ('benchmark_%s_%s.json' % (a.variant,a.round))
    old.fresh(target)
    # Reuse E102's original-order capture (no persisted tensors or padding).
    from profile_rec_runtime import capture
    feeds, expected, cfg, image, execution = capture(a)
    gc.collect()
    from trt82_backend import Session
    report = dict(report_meta(a), variant=a.variant, round=a.round, rows=[], before=old.telemetry(),
                  image_sha256=image, input_shapes=[list(v['x'].shape) for v in feeds],
                  input_sha256=[hashlib.sha256(v['x'].tobytes()).hexdigest() for v in feeds],
                  warmup=3, repeat=20, note='Same original in-profile crop order, batch1, synchronous Session.run incl route/checks/copies. Not full business/HTTP; no clocks changed.')
    native = None
    try:
        spec = cfg['engines']['rec']
        native = Session(q.RELEASE/spec['path'], spec['sha256'], 'rec_fp32_broad')
        expected_tokens=[]
        for values, hashes in zip(feeds, expected):
            out=native.run(None,values)
            if [hashlib.sha256(v.tobytes()).hexdigest() for v in out] != hashes:
                raise RuntimeError('Broad baseline output drift')
            expected_tokens.append(old.tokens(out[0]))
        if a.variant == 'routed':
            short = candidate(a.stage)
            native = Route(native,short)
        for _ in range(3):
            for values,tokens in zip(feeds,expected_tokens):
                if old.tokens(native.run(None,values)[0]) != tokens:
                    raise RuntimeError('Pre-timing token drift')
        report['after_warmup'] = old.telemetry()
        for cycle in range(20):
            for index,values in enumerate(feeds):
                start=time.perf_counter()
                native.run(None,values)
                report['rows'].append({'cycle':cycle,'index':index,'wall_ms':(time.perf_counter()-start)*1000})
        for values,tokens in zip(feeds,expected_tokens):
            if old.tokens(native.run(None,values)[0]) != tokens:
                raise RuntimeError('Post-timing token drift')
        report.update(status='measured',timing=old.summary([r['wall_ms'] for r in report['rows']]),
                      after=old.telemetry(),execution=native.stats(),pre_post_tokens_equal=True)
    except Exception as e:
        report.update(status='failed',error=type(e).__name__)
        raise
    finally:
        if native is not None:
            native.close()
        q.save(target,report)
    print(json.dumps({'variant':a.variant,'round':a.round,'timing':report['timing']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('build','check','evaluate','benchmark'))
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--variant',choices=('reference','routed'),default='routed')
    p.add_argument('--round',choices=('r1','r2'),default='r1')
    a=p.parse_args(); a.stage=a.stage.resolve()
    if (platform.machine()!='aarch64' or not a.stage.is_relative_to(q.BASE/'staging') or not a.stage.is_dir()):
        p.error('Existing isolated board staging required')
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(q.RELEASE/'edge_deploy/jetson'))
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE),AUTOOCR_RUNTIME_DIR=str(a.stage/'runtime'),AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL)
    {'build':build,'check':check,'evaluate':evaluate,'benchmark':benchmark}[a.mode](a)


if __name__=='__main__':
    main()
