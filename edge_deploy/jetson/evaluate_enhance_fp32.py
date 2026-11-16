"""E108 unchanged tile/FP32 contract, full business regression, gated timing.

No OCR text, images, tensors, license or HTTP output is saved. Original inputs
remain in RAM; prior helpers are frozen and never monkeypatched globally.
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
import re
import sys
import time

import numpy as np
import build_enhance_fp32 as build
import optimize_enhance_fp16 as legacy

q, common = build.q, legacy.common


def tensor_sha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def metadata(stage):
    b = json.loads((stage/'build.json').read_text())
    if (any(b[k] != v for k, v in build.provenance().items())
            or b['status'] != 'built_not_validated' or b['flags'] != 0
            or b['workspace_bytes'] != build.WORKSPACE or b['avg_timing_iterations'] != 8
            or b['trt_version'] != '8.2.1.8+localcp310'):
        raise ValueError('Successful unchanged FP32 build required')
    build.validate_contract([tuple(v) for v in b['inputs']], [tuple(v) for v in b['outputs']])
    return dict(script_sha256=q.digest(__file__), build_script_sha256=q.digest(build.__file__),
        build_sha256=q.digest(stage/'build.json'), engine_sha256=b['engine_sha256'],
        baseline_engine_sha256=b['baseline_engine_sha256'], onnx_sha256=b['onnx_sha256'],
        deployment_sha256=b['deployment_sha256'], plan_sha256=b['plan_sha256'],
        helpers_sha256={Path(m.__file__).name:q.digest(m.__file__) for m in (legacy, common, q, build.source)},
        diagnostic_helper_sha256=q.digest(Path(__file__).with_name('diagnose_rec_difference.py')))


def candidate(stage):
    from trt82_backend import Session
    return Session(stage/'enhance_fp32.engine', metadata(stage)['engine_sha256'], 'enhance_fp32_ws1024')


def require(stage, name, status):
    r = json.loads((stage/name).read_text())
    if r['status'] != status or any(r[k] != v for k,v in metadata(stage).items()):
        raise ValueError('Prior gate/provenance changed')
    return r


def inspect(info):
    layers = info.get('Layers', [])
    if not layers: raise ValueError('Detailed inspector required')
    convs = []
    for row in layers:
        formats = [t.get('Format/Datatype', '') for t in row.get('Inputs', [])+row.get('Outputs', [])]
        if any(re.search(r'\b(half|fp16|int8)\b', v.lower()) for v in formats):
            raise ValueError('Unexpected low precision tensor')
        if row.get('ParameterType') in ('Convolution', 'Deconvolution'):
            if not formats or not all(re.search(r'\b(float|fp32)\b', v.lower()) for v in formats):
                raise ValueError('FP32 convolution formats not evidenced')
            convs.append(dict(name=row['Name'], type=row['LayerType'], tactic=row.get('TacticValue')))
    if not convs: raise ValueError('No convolution evidence')
    return dict(layer_count=len(layers), convolutions=convs, low_precision_format_count=0)


def capture(a, count):
    """Original business-generated tiles, limited to known 9-tile pages, RAM only."""
    plan = q.load_plan(q.BASE/'staging/quant_yolo_e92_r1')
    rows = []; models = base = original = None
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = common.setup(a)
            base = models.models['enhance']; original = base.run
            active = None
            def run(names, values):
                x = values['images']
                if x.shape != build.source.INPUT_SHAPE or x.dtype != np.float32 or len(rows) >= count*9:
                    raise ValueError('Unexpected tile contract/count')
                y = original(names, values)[0]
                rows.append(dict(image_sha256=active, input_shape=list(x.shape), input_sha256=tensor_sha(x),
                                 reference_sha256=tensor_sha(y), x=x.copy(), reference=y.copy()))
                return [y]
            base.run = run
            for image in plan['calibration'][:count]:
                active = image['sha256']; before = len(rows)
                with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                    result = pipeline.process(str(Path(plan['images'])/image['file']), image_type=1, run_ner=True)
                if not result['success'] or not models.healthy or len(rows)-before != 9:
                    raise RuntimeError('Original pipeline or tile capture failed')
            return rows, cfg
    finally:
        if base is not None and original is not None: base.run = original
        common.close_models(models)


def describe(row):
    return {k:row[k] for k in ('image_sha256','input_shape','input_sha256','reference_sha256')}


def quality_ready(rows):
    return (legacy.quality_ok(rows)
            and all(legacy.business_passed(r['strict_equal'],r['field_diagnostics']) for r in rows)
            and all(sum(r['cpu_fallback_calls']['candidate'][k] for r in rows)
                    <= sum(r['cpu_fallback_calls']['reference'][k] for r in rows) for k in ('det','rec')))


def check(a):
    target=a.stage/'check.json'; common.fresh(target); common.fresh(a.stage/'layers.json')
    report=dict(metadata(a.stage), status='running', rows=[])
    native=None
    try:
        rows, cfg=capture(a, 3); gc.collect(); native=candidate(a.stage)
        for row in rows:
            out=native.run(None, {'images':row['x']})[0]
            again=native.run(None, {'images':row['x']})[0]
            if not np.array_equal(out, again): raise ValueError('Candidate not repeatable')
            report['rows'].append(dict(describe(row), **legacy.compare_arrays(row['reference'],out),
                candidate_sha256=tensor_sha(out), repeat_exact=True))
        inspector=native.engine.create_engine_inspector(); inspector.execution_context=native.context
        info=json.loads(inspector.get_engine_information(native.trt.LayerInformationFormat.JSON))
        details=inspect(info)
        if details['layer_count'] != native.engine.num_layers: raise ValueError('Incomplete inspector')
        q.save(a.stage/'layers.json',info)
        report.update(status='finite_original_tiles_not_business_accepted', inspection=details,
            execution=native.stats(), note='27 real tiles repeated, numerical/mask checks diagnostic. Original 512 tiling untouched; no GT or benchmark.')
    except Exception as e:
        report.update(status='failed',error=type(e).__name__); raise
    finally:
        if native is not None: native.close()
        q.save(target,report)
    print(json.dumps(dict(status=report['status'],tiles=len(report['rows']))),flush=True)


def evaluate(a):
    require(a.stage,'check.json','finite_original_tiles_not_business_accepted')
    target=a.stage/'evaluation.json'; common.fresh(target)
    plan=q.load_plan(q.BASE/'staging/quant_yolo_e92_r1'); images=plan['calibration']+plan['holdout']
    report=dict(metadata(a.stage),status='running',rows=[],note='13 enhanced images + ordinary negative control. Paired order alternates. No new normalization, GT or benchmark.')
    models=trial=base=original=None
    try:
        with open(os.devnull,'w') as quiet:
            with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                server,pipeline,models,cfg=common.setup(a)
            base=models.models['enhance']; original=base.run; trial=candidate(a.stage)
            active='reference'; hashes=None
            def run(names,feeds):
                if trial.errors: raise RuntimeError('Candidate unhealthy; no fallback')
                hashes[active].append(tensor_sha(feeds['images']))
                return original(names,feeds) if active=='reference' else trial.run(names,feeds)
            base.run=run
            for image,kind in [(images[0],3)]+[(v,1) for v in images]:
                hashes={v:[] for v in ('reference','candidate')}; outputs={}; wall={}; counts={}
                order=('reference','candidate') if len(report['rows'])%2==0 else ('candidate','reference')
                for active in order:
                    before=models.stats(); start=time.perf_counter()
                    with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                        result=pipeline.process(str(Path(plan['images'])/image['file']),image_type=kind,run_ner=True)
                    wall[active]=(time.perf_counter()-start)*1000; after=models.stats()
                    counts[active]={k:after[k]['cpu_fallback']['calls']-before[k]['cpu_fallback']['calls'] for k in ('det','rec')}
                    if not result['success'] or not models.healthy or trial.errors: raise RuntimeError('Pipeline/backend failure')
                    outputs[active]=result['data']
                legacy.verify_usage(kind,hashes['reference'],hashes['candidate'])
                ref,out=outputs['reference'],outputs['candidate']; row=q.strict_compare(ref,out)
                fields=q.field_diagnostics(ref,out,server._layout_special_text)
                row.update(image_sha256=image['sha256'],image_type=kind,order=order,wall_ms=wall,
                    same_original_inputs=True,tile_input_sha256=hashes['reference'],tile_calls=len(hashes['reference']),
                    cpu_fallback_calls=counts,field_diagnostics=fields,
                    full_text_diagnostics=legacy.text_diagnostics(ref['FullText'],out['FullText']),
                    strict_objects_passed=row['passed'])
                row['passed']=legacy.business_passed(row['strict_equal'],fields); report['rows'].append(row)
                print(json.dumps(dict(case=len(report['rows']),passed=row['passed'])),flush=True)
            report.update(status='all_cases_consistent_not_gt' if legacy.quality_ok(report['rows']) else 'differences_not_accepted',
                          candidate_execution=trial.stats(),original_execution=models.stats())
    except Exception as e:
        report.update(status='failed',error=type(e).__name__); raise
    finally:
        if base is not None and original is not None: base.run=original
        if trial is not None: trial.close()
        common.close_models(models); q.save(target,report)
    print(json.dumps(dict(status=report['status'],passed=sum(r['passed'] for r in report['rows']))),flush=True)


def benchmark(a):
    quality=require(a.stage,'evaluation.json','all_cases_consistent_not_gt')
    if not quality_ready(quality['rows']): raise ValueError('Incomplete quality/fallback gate')
    checked=require(a.stage,'check.json','finite_original_tiles_not_business_accepted')
    target=a.stage/('benchmark_%s_%s.json'%(a.variant,a.round)); common.fresh(target)
    report=dict(metadata(a.stage),status='running',variant=a.variant,round=a.round,rows=[],warmup=3,repeat=20,
                note='Same original 9 tiles in original order. Separate process, synchronous Session.run with checks/copies/execution. No preprocessing/merge/whole page/HTTP; tails exploratory.')
    native=None
    try:
        rows,cfg=capture(a,1); gc.collect()
        expected=checked['rows'][:9]
        if [describe(r) for r in rows]!=[describe(r) for r in expected]: raise ValueError('Original capture drift')
        report.update(inputs=[describe(r) for r in rows],before=common.telemetry())
        from trt82_backend import Session
        spec=cfg['engines']['enhance']
        native=candidate(a.stage) if a.variant=='candidate' else Session(q.RELEASE/spec['path'],spec['sha256'],'enhance_fp32_reference')
        hashes=[r['candidate_sha256' if a.variant=='candidate' else 'reference_sha256'] for r in expected]
        def probe():
            for row,h in zip(rows,hashes):
                if tensor_sha(native.run(None,{'images':row['x']})[0])!=h: raise ValueError('Output drift')
        for _ in range(3): probe()
        report['after_warmup']=common.telemetry()
        for cycle in range(20):
            for index,row in enumerate(rows):
                start=time.perf_counter(); native.run(None,{'images':row['x']})
                report['rows'].append(dict(cycle=cycle,index=index,wall_ms=(time.perf_counter()-start)*1000))
        probe()
        report.update(status='measured',pre_post_outputs_exact=True,timing=common.summary([r['wall_ms'] for r in report['rows']]),
                      execution=native.stats(),after=common.telemetry(),context_device_bytes=native.engine.device_memory_size)
    except Exception as e:
        report.update(status='failed',error=type(e).__name__); raise
    finally:
        if native is not None: native.close()
        q.save(target,report)
    print(json.dumps(dict(variant=a.variant,round=a.round,timing=report['timing'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=('check','evaluate','benchmark'))
    p.add_argument('--stage',required=True,type=Path); p.add_argument('--variant',choices=('reference','candidate'),default='candidate')
    p.add_argument('--round',choices=('r1','r2'),default='r1'); a=p.parse_args(); a.stage=a.stage.resolve()
    if platform.machine()!='aarch64' or not a.stage.is_relative_to(q.BASE/'staging') or not a.stage.is_dir(): p.error('Isolated board stage required')
    sys.path.insert(0,str(q.RELEASE/'edge_deploy/jetson'))
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE),AUTOOCR_RUNTIME_DIR=str(a.stage/'runtime'),AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL)
    globals()[a.mode](a)
