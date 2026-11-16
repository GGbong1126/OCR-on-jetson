"""E112: independently declared effective-layout/business gate and timings.

Read-only reuse of E111 FP32 engine; its failed raw-table gate is NOT relabelled.
"""
import contextlib
import gc
import json
import os
from pathlib import Path
import tarfile
import time

import numpy as np
import evaluate_layout_outputs as e
import diagnose_layout_outputs as d

q, old, source = e.q, e.old, e.source
PRIOR=q.BASE/'staging/layout_outputs_e111_r1'
PRIOR_ARCHIVE_SHA='00cd1dddf9ded4cba6f66045f3949c893aa623fa4f8865a10305ee61c747906b'
RULES=dict(confidence=0.3,nms_iou=0.5,matching_iou_min=0.99,
    max_coordinate_delta_pixels=0.5,max_score_delta=0.001,
    require_effective_order=True,require_above_threshold_count_equal=True,
    require_count_output_exact=True,require_strict_business=True,
    raw_allclose_acceptance=False,warmup=3,repeat=10,inputs=6,
    mean_improvement_min_percent=5.,p95_ratio_max=1.05,extra_warm_nvmap_max_mib=256.)


def read(path):return json.loads(path.read_text())


def lineage():
    if q.digest(PRIOR/'technical_evidence.tar.gz')!=PRIOR_ARCHIVE_SHA:raise ValueError('Prior archive changed')
    with tarfile.open(PRIOR/'technical_evidence.tar.gz') as tar:
        manifest=tar.extractfile('manifest.json').read()
    if manifest!=(PRIOR/'manifest.json').read_bytes():raise ValueError('Prior manifest changed')
    for row in json.loads(manifest)['files']:
        if Path(row['name']).name!=row['name'] or q.digest(PRIOR/row['name'])!=row['sha256']:
            raise ValueError('Prior frozen evidence changed')
    meta=e.metadata(PRIOR)
    if read(PRIOR/'check.json')['status']!='numeric_not_accepted':raise ValueError('Historical status changed')
    return dict(prior_metadata=meta,prior_archive_sha256=PRIOR_ARCHIVE_SHA,
        prior_reports_sha256={n:q.digest(PRIOR/n) for n in ('check.json','business_diagnostic.json','output_diagnostic.json','final_audit.json')},
        script_sha256=q.digest(__file__),diagnostic_helper_sha256=q.digest(d.__file__))


def prepare(a):
    q.save(a.stage/'plan.json',dict(lineage(),rules=RULES,
        note='New E112 policy, informed by E111 diagnostic, NOT a pass of E111 raw-table allclose. Same convenience 13 images already used in development, not new/independent GT. No engine rebuild, text normalization, release/HTTP/activation change. Full 26 paired cases must pass this policy before any timing.',
        image_cases=[dict(image_sha256=r['sha256'],image_type=k) for r in
            (lambda p:p['calibration']+p['holdout'])(q.load_plan(q.BASE/'staging/quant_yolo_e92_r1')) for k in (3,1)]))
    print('E112 plan frozen; E111 failure preserved',flush=True)


def metadata(stage):
    plan=read(stage/'plan.json');current=lineage()
    if plan['rules']!=RULES or any(plan[k]!=v for k,v in current.items()):raise ValueError('Frozen E112 plan differs')
    return dict(current,plan_sha256=q.digest(stage/'plan.json'))


def geometry_passed(row):
    match=row['effective_match'];n=match['reference_count']
    if (n!=match['candidate_count'] or n!=match['unique_matched_count']
            or match['unmatched_reference_count'] or match['unmatched_candidate_count']
            or not match['order_preserved'] or not row['count_output_exact']
            or row['above_threshold_counts'][0]!=row['above_threshold_counts'][1]):return False
    if n==0:return True  # Empty is consistency, never correctness; business gate still required.
    vals=[match['matched_min_iou'],match['matched_max_coordinate_delta'],match['matched_max_score_delta']]
    return (all(v is not None and np.isfinite(v) for v in vals)
        and vals[0]>=RULES['matching_iou_min'] and 0<=vals[1]<=RULES['max_coordinate_delta_pixels']
        and 0<=vals[2]<=RULES['max_score_delta'])


def row_passed(row):
    return geometry_passed(row) and e.shared.business_passed(row) and row['outputs_finite']


def require_quality(stage):
    result=read(stage/'quality.json');meta=metadata(stage);plan=read(stage/'plan.json')
    if (result['status']!='semantic_business_consistent_not_gt' or any(result[k]!=v for k,v in meta.items())
            or result['candidate_execution']['calls']!=26 or result['candidate_execution']['errors']
            or [{k:r[k] for k in ('image_sha256','image_type')} for r in result['rows']]!=plan['image_cases']
            or len(result['rows'])!=26 or not all(row_passed(r) and r['passed'] for r in result['rows'])):
        raise ValueError('Incomplete or changed E112 quality gate')
    return result


def quality(a):
    target=a.stage/'quality.json';old.fresh(target)
    report=dict(metadata(a.stage),status='running',rows=[],old_raw_gate='numeric_not_accepted')
    images=q.load_plan(q.BASE/'staging/quant_yolo_e92_r1')
    former={(r['image_sha256'],r['image_type']):r for r in read(PRIOR/'business_diagnostic.json')['rows']}
    models=trial=base=original=None
    try:
        with open(os.devnull,'w') as quiet:
            with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                server,pipeline,models,cfg=old.setup(a)
            base=models.models['layout'];original=base.run;trial=e.candidate(PRIOR)
            active='reference';hashes=raw=feeds_used=None
            def call(names,feeds):
                if names not in (None,[]) or trial.errors:raise ValueError('Unhealthy backend/request')
                hashes[active].append(source.input_metadata(feeds))
                values=original(names,feeds) if active=='reference' else e.retained_outputs(trial,feeds)
                raw[active]=dict(zip(e.b.KEEP,values[:2]));feeds_used[active]=feeds
                return values
            guarded=source.GuardedCall(call);base.run=guarded
            for image in images['calibration']+images['holdout']:
                for kind in (3,1):
                    values={};counts={};hashes={'reference':[],'candidate':[]};raw={};feeds_used={}
                    order=('reference','candidate') if len(report['rows'])%2==0 else ('candidate','reference')
                    for active in order:
                        before=models.stats()
                        with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                            result=pipeline.process(str(Path(images['images'])/image['file']),image_type=kind,run_ner=True)
                        guarded.require_healthy();after=models.stats()
                        if not result['success'] or not models.healthy or trial.errors:raise RuntimeError('Business/backend failure')
                        counts[active]={r:after[r]['cpu_fallback']['calls']-before[r]['cpu_fallback']['calls'] for r in ('det','rec')}
                        values[active]=result['data']
                    prior=former[image['sha256'],kind]
                    if (hashes['reference']!=[prior['inputs']] or hashes['reference']!=hashes['candidate']
                            or counts['reference']!=prior['cpu_fallback_calls']['reference']):
                        raise ValueError('Input/reference fallback drift')
                    compared=e.shared.compare(raw['reference'],raw['candidate'])
                    rb,rn=d.business_boxes(raw['reference'],feeds_used['reference'],server.OCRPipeline._nms_boxes)
                    cb,cn=d.business_boxes(raw['candidate'],feeds_used['candidate'],server.OCRPipeline._nms_boxes)
                    ref,out=values['reference'],values['candidate'];row=q.strict_compare(ref,out)
                    row.update(image_sha256=image['sha256'],image_type=kind,order=order,
                        inputs=hashes['reference'][0],same_original_input=True,cpu_fallback_calls=counts,
                        raw_numeric_diagnostic=compared,outputs_finite=True,
                        count_output_exact=compared['fetch_name_1']['exactly_equal'],
                        above_threshold_counts=[len(rb),len(cb)],effective_match=d.match_boxes(rn,cn),
                        output_sha256={branch:{n:source.sha_tensor(v) for n,v in data.items()} for branch,data in raw.items()},
                        fields=q.field_diagnostics(ref,out,server._layout_special_text),
                        number_symbol_sequence_equal=e.shared.protected_equal(ref['FullText'],out['FullText']),
                        business_sha256={branch:q.hashlib.sha256(json.dumps({k:value[k] for k in ('FullText','Text','MarkdownBodyLines','NER')},sort_keys=True,ensure_ascii=False).encode()).hexdigest() for branch,value in values.items()})
                    row['passed']=row_passed(row);report['rows'].append(row)
                    print('Semantic/business',len(report['rows']),row['passed'],flush=True)
            inspector=trial.engine.create_engine_inspector();inspector.execution_context=trial.context
            info=json.loads(inspector.get_engine_information(trial.trt.LayerInformationFormat.JSON))
            q.save(a.stage/'layers.json',info);report['inspection']=e.inspect(info)
            report['status']='semantic_business_consistent_not_gt' if len(report['rows'])==26 and all(r['passed'] for r in report['rows']) else 'quality_not_accepted'
    except Exception as err:
        report.update(status='failed',error=type(err).__name__);raise
    finally:
        if original is not None:base.run=original
        if trial is not None:report['candidate_execution']=trial.stats();trial.close()
        if models is not None:report['original_execution']=models.stats()
        old.close_models(models);q.save(target,report)
    print(report['status'],flush=True)


def benchmark(a,variant,rnd):
    quality_report=require_quality(a.stage)
    target=a.stage/('benchmark_%s_%s.json'%(variant,rnd));old.fresh(target)
    report=dict(metadata(a.stage),quality_sha256=q.digest(a.stage/'quality.json'),status='running',variant=variant,round=rnd,rows=[],warmup=3,repeat=10,
        note='Same six real inputs, native synchronous Session.run all outputs including copy/checks. Reference3 vs candidate2 outputs, FP32 graph/transfer optimization not precision/pure GPU/HTTP. Separate processes, order ref/cand/cand/ref; 60 timed samples per process, p95 exploratory, clocks unlocked.')
    native=None
    try:
        rows,cfg,_=source.capture(a);gc.collect()
        checked=read(PRIOR/'check.json')
        if [source.describe(r) for r in rows]!=[{k:r[k] for k in source.describe(rows[0])} for r in checked['rows']]:raise ValueError('Capture drift')
        from trt82_backend import Session
        report.update(inputs=[source.describe(r) for r in rows],before=old.telemetry(),started_ns=time.time_ns())
        spec=cfg['engines']['layout']
        native=(e.candidate(PRIOR) if variant=='candidate' else Session(q.RELEASE/spec['path'],spec['sha256'],'layout_fp32_reference',[(source.PLUGIN,source.PLUGIN_SHA)]))
        expected=[r['candidate_sha256' if variant=='candidate' else 'reference_sha256'] for r in checked['rows']]
        def verify(row,wanted):
            out=e.shared.named_outputs(native,row['feeds'])
            if {k:source.sha_tensor(v) for k,v in out.items()}!=wanted:raise ValueError('Timing output drift')
        for _ in range(3):
            for row,wanted in zip(rows,expected):verify(row,wanted)
        report['after_warmup']=old.telemetry()
        for cycle in range(10):
            for i,row in enumerate(rows):
                start=time.perf_counter();native.run(None,row['feeds'])
                report['rows'].append(dict(cycle=cycle,index=i,wall_ms=(time.perf_counter()-start)*1000))
        for row,wanted in zip(rows,expected):verify(row,wanted)
        report.update(status='measured',pre_post_outputs_exact=True,after=old.telemetry(),ended_ns=time.time_ns(),
            context_device_bytes=native.engine.device_memory_size,timing=old.summary([r['wall_ms'] for r in report['rows']]))
    except Exception as err:
        report.update(status='failed',error=type(err).__name__);raise
    finally:
        if native is not None:report['execution']=native.stats();native.close()
        q.save(target,report)
    print(json.dumps(dict(variant=variant,round=rnd,timing=report['timing'])),flush=True)


if __name__=='__main__':
    a=source.arguments(('prepare','quality')+tuple('benchmark_'+v+'_'+r for v in ('reference','candidate') for r in ('r1','r2')))
    if a.mode.startswith('benchmark_'):
        _,variant,rnd=a.mode.split('_');benchmark(a,variant,rnd)
    else:{'prepare':prepare,'quality':quality}[a.mode](a)
