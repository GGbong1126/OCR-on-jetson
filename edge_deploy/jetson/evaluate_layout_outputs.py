"""E111 retained-output checks, offline business regression and gated timings."""
import contextlib
import gc
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import layout_outputs_fp32 as b
import evaluate_layout_hotspot as shared

q, old, source = b.q, b.old, b.source


def metadata(stage):
    b.require_prepare(stage)
    built = json.loads((stage/'build.json').read_text())
    if (built['status'] != 'built_not_validated' or built['flags'] != 0
            or any(built[k] != v for k,v in b.provenance().items())
            or built['prepare_sha256'] != q.digest(stage/'prepare.json')
            or built['engine_sha256'] != q.digest(stage/b.ENGINE)):
        raise ValueError('Build provenance differs')
    return dict(evaluator_sha256=q.digest(__file__), build_sha256=q.digest(stage/'build.json'),
        prepare_sha256=q.digest(stage/'prepare.json'), engine_sha256=built['engine_sha256'],
        helper_sha256=q.digest(shared.__file__), provenance=b.provenance())


def require(stage, name, status):
    report = json.loads((stage/name).read_text())
    if report['status'] != status or any(report[k] != v for k,v in metadata(stage).items()):
        raise ValueError('Prior gate/provenance differs')
    return report


def candidate(stage):
    from trt82_backend import Session
    return Session(stage/b.ENGINE, metadata(stage)['engine_sha256'], 'layout_two_outputs_fp32',
                   [(source.PLUGIN,source.PLUGIN_SHA)])


def retained_outputs(session, feeds):
    names = [v.name for v in session.get_outputs()]
    if len(names) != 2 or set(names) != set(b.KEEP):
        raise ValueError('Candidate must expose exactly two real outputs')
    values = session.run(None,feeds)
    if len(values) != 2: raise ValueError('Candidate output count differs')
    by = dict(zip(names,values))
    return [by[k] for k in b.KEEP]


def inspect(info):
    rows = info.get('Layers',[]); plugins = []
    if not rows: raise ValueError('Detailed Inspector required')
    for row in rows:
        tensors = row.get('Inputs',[]) + row.get('Outputs',[])
        if not tensors or any(not v.get('Format/Datatype') for v in tensors):
            raise ValueError('Missing dtype evidence')
        if any(re.search(r'\b(?:half|fp16|int8|bf16)\b',v['Format/Datatype'].lower()) for v in tensors):
            raise ValueError('Unexpected reduced precision')
        if 'GridSample' in row.get('Name',''):
            if not shared.dtype_is(tensors,'float'): raise ValueError('Plugin no longer FP32')
            plugins.append(row['Name'])
    if len(plugins) != 18: raise ValueError('Plugin count changed')
    return dict(layer_count=len(rows), fp32_grid_sample_count=len(plugins),
                reduced_precision_io_records=0,
                note='Executed I/O formats plus builder flags=0; not an internal accumulator trace.')


def numeric_passed(row):
    raw = row['raw']
    return (set(raw) == set(b.KEEP) and row['repeat_exact']
            and all(v['all_finite'] and v['allclose_diagnostic'] for v in raw.values())
            and raw['fetch_name_1']['exactly_equal'])


def check(a):
    target=a.stage/'check.json'; old.fresh(target); old.fresh(a.stage/'layers.json')
    report=dict(metadata(a.stage),status='running',rows=[]); trial=None
    try:
        rows,cfg,execution=source.capture(a); gc.collect()
        trial=candidate(a.stage)
        for row in rows:
            out=dict(zip(b.KEEP,retained_outputs(trial,row['feeds'])))
            again=dict(zip(b.KEEP,retained_outputs(trial,row['feeds'])))
            item=dict(source.describe(row),
                raw=shared.compare({k:row['reference'][k] for k in b.KEEP},out),
                repeat_exact=all(np.array_equal(out[k],again[k]) for k in b.KEEP),
                candidate_sha256={k:source.sha_tensor(v) for k,v in out.items()})
            item['passed']=numeric_passed(item); report['rows'].append(item)
            print('Checked retained outputs',len(report['rows']),item['passed'],flush=True)
        inspector=trial.engine.create_engine_inspector(); inspector.execution_context=trial.context
        info=json.loads(inspector.get_engine_information(trial.trt.LayerInformationFormat.JSON))
        q.save(a.stage/'layers.json',info); report['inspection']=inspect(info)
        report.update(status='retained_outputs_checked' if all(r['passed'] for r in report['rows']) else 'numeric_not_accepted',
                      original_execution=execution,
                      note='Same six original inputs, original three-output engine vs real two-output candidate. No synthetic masks. Numeric baseline consistency is not ground-truth accuracy.')
    except Exception as e:
        report.update(status='failed',error=type(e).__name__); raise
    finally:
        if trial is not None: report['execution']=trial.stats(); trial.close()
        q.save(target,report)
    print(json.dumps(dict(status=report['status'],cases=len(report['rows']))),flush=True)


def evaluate(a):
    require(a.stage,'check.json','retained_outputs_checked')
    target=a.stage/'evaluation.json'; old.fresh(target)
    report=dict(metadata(a.stage),status='running',rows=[])
    plan=q.load_plan(q.BASE/'staging/quant_yolo_e92_r1')
    models=trial=base=original=None
    try:
        with open(os.devnull,'w') as quiet:
            with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                server,pipeline,models,cfg=old.setup(a)
            base=models.models['layout']; original=base.run; trial=candidate(a.stage)
            active='reference'; hashes=None
            def run(names,feeds):
                if names not in (None,[]) or trial.errors: raise ValueError('Invalid request or unhealthy candidate')
                hashes[active].append(source.input_metadata(feeds))
                return original(names,feeds) if active=='reference' else retained_outputs(trial,feeds)
            guarded=source.GuardedCall(run); base.run=guarded
            for image in plan['calibration']+plan['holdout']:
                for kind in (3,1):
                    outputs={}; wall={}; counts={}; hashes={'reference':[],'candidate':[]}
                    order=('reference','candidate') if len(report['rows'])%2==0 else ('candidate','reference')
                    for active in order:
                        before=models.stats(); start=time.perf_counter()
                        with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                            result=pipeline.process(str(Path(plan['images'])/image['file']),image_type=kind,run_ner=True)
                        guarded.require_healthy(); wall[active]=(time.perf_counter()-start)*1000
                        after=models.stats()
                        if not result['success'] or not models.healthy or trial.errors: raise RuntimeError('Business/backend failure')
                        counts[active]={r:after[r]['cpu_fallback']['calls']-before[r]['cpu_fallback']['calls'] for r in ('det','rec')}
                        outputs[active]=result['data']
                    if len(hashes['reference'])!=1 or hashes['reference']!=hashes['candidate']:
                        raise ValueError('Layout input drift or bypass')
                    ref,out=outputs['reference'],outputs['candidate']; row=q.strict_compare(ref,out)
                    row.update(image_sha256=image['sha256'],image_type=kind,order=order,wall_ms=wall,
                        same_original_input=True,inputs=hashes['reference'][0],cpu_fallback_calls=counts,
                        fields=q.field_diagnostics(ref,out,server._layout_special_text),
                        number_symbol_sequence_equal=shared.protected_equal(ref['FullText'],out['FullText']),
                        reference_text_sha256=source.hashlib.sha256(ref['FullText'].encode()).hexdigest(),
                        candidate_text_sha256=source.hashlib.sha256(out['FullText'].encode()).hexdigest(),
                        strict_objects_passed=row['passed'])
                    row['passed']=shared.business_passed(row); report['rows'].append(row)
                    print('Business case',len(report['rows']),row['passed'],flush=True)
            report.update(status='all_26_business_consistent_not_gt' if len(report['rows'])==26 and all(r['passed'] for r in report['rows']) else 'differences_not_accepted',
                note='13 existing images x both paths, original remaining six models and business unchanged. Strict text/grouping/Markdown/NER/special text and fallback equality, no new normalization or GT. Single-pass walls are diagnostic, not benchmark.')
    except Exception as e:
        report.update(status='failed',error=type(e).__name__); raise
    finally:
        if original is not None: base.run=original
        if trial is not None: report['candidate_execution']=trial.stats(); trial.close()
        if models is not None: report['original_execution']=models.stats()
        old.close_models(models); q.save(target,report)
    print(json.dumps(dict(status=report['status'],passed=sum(r['passed'] for r in report['rows']))),flush=True)


def benchmark(a,variant,rnd):
    quality=require(a.stage,'evaluation.json','all_26_business_consistent_not_gt')
    checked=require(a.stage,'check.json','retained_outputs_checked')
    if len(quality['rows'])!=26 or not all(shared.business_passed(r) for r in quality['rows']):
        raise ValueError('Incomplete quality gate')
    target=a.stage/('benchmark_%s_%s.json'%(variant,rnd)); old.fresh(target)
    report=dict(metadata(a.stage),status='running',variant=variant,round=rnd,rows=[],warmup=3,repeat=10,
        note='Synchronous native Session.run including input checks/copies, GPU work, output copies and finite checks. Original THREE outputs vs candidate TWO; this deliberately measures graph/output-transfer optimization, not precision gain/pure kernel/full OCR/HTTP. Same six inputs x10, warmup3, separate processes; unlocked clocks, p95 exploratory.')
    native=None
    try:
        rows,cfg,_=source.capture(a); gc.collect()
        if [source.describe(r) for r in rows]!=[{k:r[k] for k in source.describe(rows[0])} for r in checked['rows']]:
            raise ValueError('Original input/output drift')
        from trt82_backend import Session
        report.update(inputs=[source.describe(r) for r in rows],before=old.telemetry())
        spec=cfg['engines']['layout']
        native=(candidate(a.stage) if variant=='candidate' else Session(q.RELEASE/spec['path'],spec['sha256'],
                'layout_fp32_reference',[(source.PLUGIN,source.PLUGIN_SHA)]))
        expected=[r['candidate_sha256' if variant=='candidate' else 'reference_sha256'] for r in checked['rows']]
        def verify(row,wanted):
            values=shared.named_outputs(native,row['feeds'])
            if {k:source.sha_tensor(v) for k,v in values.items()}!=wanted: raise ValueError('Timing output drift')
        for _ in range(3):
            for row,wanted in zip(rows,expected): verify(row,wanted)
        report['after_warmup']=old.telemetry()
        for cycle in range(10):
            for i,row in enumerate(rows):
                start=time.perf_counter(); native.run(None,row['feeds'])
                report['rows'].append(dict(cycle=cycle,index=i,wall_ms=(time.perf_counter()-start)*1000))
        for row,wanted in zip(rows,expected): verify(row,wanted)
        report.update(status='measured',pre_post_outputs_exact=True,after=old.telemetry(),
            context_device_bytes=native.engine.device_memory_size,timing=old.summary([r['wall_ms'] for r in report['rows']]))
    except Exception as e:
        report.update(status='failed',error=type(e).__name__); raise
    finally:
        if native is not None: report['execution']=native.stats(); native.close()
        q.save(target,report)
    print(json.dumps(dict(variant=variant,round=rnd,timing=report['timing'])),flush=True)


if __name__=='__main__':
    modes=('check','evaluate')+tuple('benchmark_'+v+'_'+r for v in ('reference','candidate') for r in ('r1','r2'))
    a=source.arguments(modes)
    if a.mode.startswith('benchmark_'):
        _,variant,rnd=a.mode.split('_'); benchmark(a,variant,rnd)
    else: {'check':check,'evaluate':evaluate}[a.mode](a)
