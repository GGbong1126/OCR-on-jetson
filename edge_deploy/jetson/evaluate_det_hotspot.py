"""E104 real detector validation and gated timing. Statistics only, no text."""
import argparse
import contextlib
import gc
import json
import logging
import os
from pathlib import Path
import platform
import re
import sys
import time
import unicodedata

import numpy as np
import det_hotspot_mixed as build
import optimize_rec_fp16 as old
import quantize_yolo_ptq as q


def metadata(stage):
    current=build.provenance()
    b=json.loads((stage/'build.json').read_text())
    for key in ('script_sha256','source_sha256','deployment_sha256','helper_sha256','plan_sha256'):
        if b[key]!=current[key]:raise ValueError('Build provenance changed')
    if (b['status']!='built_not_validated' or b['targets']!=list(build.TARGETS)
            or b['profile']!=[list(v) for v in build.PROFILE]
            or b['ranges_sha256']!=q.digest(stage/'ranges.json')):
        raise ValueError('Unexpected build policy')
    return dict(script_sha256=q.digest(__file__),build_script_sha256=current['script_sha256'],
                build_sha256=q.digest(stage/'build.json'),engine_sha256=b['engine_sha256'],
                source_sha256=build.SOURCE_SHA,deployment_sha256=build.DEPLOYMENT_SHA,
                plan_sha256=current['plan_sha256'],comparison_helper_sha256=q.digest(q.__file__))


def candidate(stage):
    from trt82_backend import Session
    m=metadata(stage)
    class Mixed(Session):
        def stats(self):
            return dict(super().stats(),precision='conv87_fp16_compute_fp32_outputs_other_compute_fp32')
    return Mixed(stage/'det_hotspot.engine',m['engine_sha256'],'det_hotspot_mixed')


def compare(ref,out):
    if (ref.shape!=out.shape or ref.ndim!=4 or ref.shape[:2]!=(1,1)
            or ref.dtype!=np.float32 or out.dtype!=np.float32
            or not np.isfinite(ref).all() or not np.isfinite(out).all()):
        raise ValueError('Invalid detector outputs')
    delta=np.abs(ref-out)
    return dict(output_shape=list(out.shape),mae=float(delta.mean()),max_abs_error=float(delta.max()),
                allclose_diagnostic=bool(np.allclose(ref,out,rtol=1e-3,atol=1e-4)),
                mask_changed_at_point1=int(np.count_nonzero((ref>.1)!=(out>.1))),
                mask_elements=ref.size,candidate_min=float(out.min()),candidate_max=float(out.max()))


def inspect_layers(info):
    layers=info.get('Layers',[])
    if not layers or not all(isinstance(row,dict) for row in layers):
        raise ValueError('Detailed Inspector required')
    selected=[row for row in layers if re.search(r'(?<![\w.])Conv\.87(?![\w.])',row.get('Name',''))
              and (row.get('ParameterType')=='Convolution'
                   or row.get('LayerType')=='Convolution')]
    if len(selected)!=1:raise ValueError('Unique executed hotspot convolution missing')
    def formats(tensors):
        return [str(t.get('Format/Datatype','')).lower() for t in tensors]
    def is_half(value):
        values=formats(value)
        return bool(values) and all(re.search(r'\b(?:half|fp16)\b',s) for s in values)
    def is_float(value):
        values=formats(value)
        return bool(values) and all(re.search(r'\b(?:float|fp32)\b',s) for s in values)
    row=selected[0]
    if not is_half(row.get('Inputs',[])) or not is_float(row.get('Outputs',[])):
        raise ValueError('Expected half-input float-output hotspot not evidenced')
    return dict(layer_count=len(layers),hotspot_layer=row,
                half_format_layer_count=sum(any(re.search(r'\b(?:half|fp16)\b',s)
                    for s in formats(r.get('Inputs',[])+r.get('Outputs',[]))) for r in layers),
                note='Inspector formats/tactic evidence; not proof of internal accumulation dtype. Reformat layers may contain half too.')


def check(a):
    target=a.stage/'check.json';old.fresh(target);old.fresh(a.stage/'layers.json')
    report=dict(metadata(a.stage),status='running',rows=[])
    native=None
    try:
        rows,cfg,execution=build.capture(a,3);gc.collect()
        expected=json.loads((a.stage/'ranges.json').read_text())['rows']
        if [build.describe(r) for r in rows]!=[{k:r[k] for k in build.describe(rows[0])} for r in expected]:
            raise ValueError('Original inputs or baseline outputs drifted since range probe')
        native=candidate(a.stage)
        for row in rows:
            out=native.run(None,{'x':row['x']})[0]
            again=native.run(None,{'x':row['x']})[0]
            if not np.array_equal(out,again):raise ValueError('Candidate repeat output drift')
            report['rows'].append(dict(build.describe(row),**compare(row['reference'],out),
                                      candidate_sha256=build.sha_tensor(out),repeat_exact=True))
        inspector=native.engine.create_engine_inspector();inspector.execution_context=native.context
        info=json.loads(inspector.get_engine_information(native.trt.LayerInformationFormat.JSON))
        q.save(a.stage/'layers.json',info)
        report['inspection']=inspect_layers(info)
        report.update(status='finite_same_inputs_compared_not_business_accepted',execution=native.stats(),original_execution=execution,
                      note='6 real original detector tensors, downstream business required. Numeric/mask differences diagnostic, no acceptance tolerance relaxation or benchmark.')
    except Exception as e:
        report.update(status='failed',error=type(e).__name__)
        raise
    finally:
        if native is not None:native.close()
        q.save(target,report)
    print(json.dumps({'status':report['status'],'cases':len(report['rows'])}),flush=True)


def business_passed(row):
    return (all(row['strict_equal'][k] for k in ('FullText','Text','MarkdownBodyLines','NER','raw_line_texts'))
            and all(row['fields']['api_special_text_equal'][k] for k in ('title','category_no','page_no'))
            and row['same_original_input'])


def protected_equal(ref,out):
    def seq(text):return ''.join(c for c in text if unicodedata.category(c)[0] in 'NPS')
    return seq(ref)==seq(out)


def require(stage,filename,status):
    r=json.loads((stage/filename).read_text());m=metadata(stage)
    if r['status']!=status or any(r[k]!=m[k] for k in m):raise ValueError('Required prior gate or provenance failed')
    return r


def evaluate(a):
    require(a.stage,'check.json','finite_same_inputs_compared_not_business_accepted')
    target=a.stage/'evaluation.json';old.fresh(target)
    report=dict(metadata(a.stage),status='running',rows=[],note='13 existing images x ordinary/enhanced, all other six models FP32. No GT, no new normalization. Boxes/special geometry diagnostic; text/grouping/fields strict. Single-pass timings not benchmark.')
    plan=q.load_plan(q.BASE/'staging/quant_yolo_e92_r1')
    models=trial=base=original=None
    try:
        with open(os.devnull,'w') as quiet:
            with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                server,pipeline,models,cfg=old.setup(a)
            base=models.models['det'].gpu;original=base.run;trial=candidate(a.stage)
            active='reference';hashes=None
            def run(names,values):
                if trial.errors:raise RuntimeError('Candidate unhealthy; no fallback')
                hashes[active].append(build.sha_tensor(values['x']))
                return original(names,values) if active=='reference' else trial.run(names,values)
            base.run=run
            for image in plan['calibration']+plan['holdout']:
                for kind in (3,1):
                    outputs={};wall={};counts={};hashes={'reference':[],'candidate':[]}
                    order=('reference','candidate') if len(report['rows'])%2==0 else ('candidate','reference')
                    for active in order:
                        before=models.stats();start=time.perf_counter()
                        with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                            result=pipeline.process(str(Path(plan['images'])/image['file']),image_type=kind,run_ner=True)
                        wall[active]=(time.perf_counter()-start)*1000
                        after=models.stats()
                        counts[active]={role:after[role]['cpu_fallback']['calls']-before[role]['cpu_fallback']['calls'] for role in ('det','rec')}
                        if not result['success'] or not models.healthy or trial.errors:
                            raise RuntimeError('Business or backend error')
                        outputs[active]=result['data']
                    if len(hashes['reference'])!=1 or hashes['reference']!=hashes['candidate']:
                        raise ValueError('Detector input changed or candidate bypassed')
                    ref,out=outputs['reference'],outputs['candidate']
                    row=q.strict_compare(ref,out)
                    row.update(image_sha256=image['sha256'],image_type=kind,order=order,wall_ms=wall,
                        same_original_input=True,input_sha256=hashes['reference'][0],cpu_fallback_calls=counts,
                        fields=q.field_diagnostics(ref,out,server._layout_special_text),
                        number_symbol_sequence_equal=protected_equal(ref['FullText'],out['FullText']),
                        reference_text_sha256=build.hashlib.sha256(ref['FullText'].encode()).hexdigest(),
                        candidate_text_sha256=build.hashlib.sha256(out['FullText'].encode()).hexdigest(),
                        strict_objects_passed=row['passed'])
                    row['passed']=business_passed(row);report['rows'].append(row)
                    print(json.dumps({'case':len(report['rows']),'passed':row['passed'],'number_symbol_sequence_equal':row['number_symbol_sequence_equal']}),flush=True)
            report.update(status='all_26_business_consistent_not_gt' if len(report['rows'])==26 and all(r['passed'] for r in report['rows']) else 'differences_not_accepted',
                          candidate_execution=trial.stats(),original_execution=models.stats())
    except Exception as e:
        report.update(status='failed',error=type(e).__name__)
        raise
    finally:
        if original is not None:base.run=original
        if trial is not None:trial.close()
        old.close_models(models);q.save(target,report)
    print(json.dumps({'status':report['status'],'passed':sum(r['passed'] for r in report['rows'])}),flush=True)


def benchmark(a):
    quality=require(a.stage,'evaluation.json','all_26_business_consistent_not_gt')
    if len(quality['rows'])!=26 or not all(business_passed(r) for r in quality['rows']):
        raise ValueError('Incomplete quality acceptance')
    checked=require(a.stage,'check.json','finite_same_inputs_compared_not_business_accepted')
    target=a.stage/('benchmark_%s_%s.json'%(a.variant,a.round));old.fresh(target)
    report=dict(metadata(a.stage),status='running',variant=a.variant,round=a.round,rows=[],warmup=3,repeat=10,
                note='6 original detector tensors from 3 existing images ordinary/enhanced, original order, separate processes. Synchronous Session.run incl checks/copies/execution; not pure kernel/full OCR/HTTP. Only 60 timed calls per process: tails exploratory. No clock/power changes.')
    native=None
    try:
        rows,cfg,execution=build.capture(a,3);gc.collect()
        if [build.describe(r) for r in rows]!=[{k:r[k] for k in build.describe(rows[0])} for r in checked['rows']]:
            raise ValueError('Input/reference drift since check')
        report.update(inputs=[build.describe(r) for r in rows],before=old.telemetry())
        from trt82_backend import Session
        spec=cfg['engines']['det']
        native=candidate(a.stage) if a.variant=='mixed' else Session(q.RELEASE/spec['path'],spec['sha256'],'det_fp32_reference')
        expected=[r['candidate_sha256' if a.variant=='mixed' else 'reference_sha256'] for r in checked['rows']]
        for _ in range(3):
            for row,h in zip(rows,expected):
                if build.sha_tensor(native.run(None,{'x':row['x']})[0])!=h:
                    raise RuntimeError('Pre-timing output drift')
        report['after_warmup']=old.telemetry()
        for cycle in range(10):
            for index,row in enumerate(rows):
                start=time.perf_counter();native.run(None,{'x':row['x']})
                report['rows'].append(dict(cycle=cycle,index=index,wall_ms=(time.perf_counter()-start)*1000))
        for row,h in zip(rows,expected):
            if build.sha_tensor(native.run(None,{'x':row['x']})[0])!=h:
                raise RuntimeError('Post-timing output drift')
        report.update(status='measured',pre_post_outputs_exact=True,timing=old.summary([r['wall_ms'] for r in report['rows']]),
                      execution=native.stats(),after=old.telemetry(),context_device_bytes=native.engine.device_memory_size)
    except Exception as e:
        report.update(status='failed',error=type(e).__name__)
        raise
    finally:
        if native is not None:native.close()
        q.save(target,report)
    print(json.dumps({'variant':a.variant,'round':a.round,'timing':report['timing']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('check','evaluate','benchmark'))
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--variant',choices=('reference','mixed'),default='mixed')
    p.add_argument('--round',choices=('r1','r2'),default='r1')
    a=p.parse_args();a.stage=a.stage.resolve()
    if platform.machine()!='aarch64' or not a.stage.is_relative_to(q.BASE/'staging') or not a.stage.is_dir():
        p.error('Existing isolated Jetson staging required')
    sys.dont_write_bytecode=True;sys.path.insert(0,str(q.RELEASE/'edge_deploy/jetson'))
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE),AUTOOCR_RUNTIME_DIR=str(a.stage/'runtime'),AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL)
    {'check':check,'evaluate':evaluate,'benchmark':benchmark}[a.mode](a)


if __name__=='__main__':main()
