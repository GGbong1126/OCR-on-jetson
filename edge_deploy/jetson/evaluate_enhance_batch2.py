"""E109 original tiles, batch2+original odd tail, full business before timing.

Diagnostic output is statistics/hashes only; no HTTP, activation or release edits.
"""
import argparse
import contextlib
import gc
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import build_enhance_batch2 as build
import evaluate_enhance_fp32 as previous

q, common, legacy = build.q, previous.common, previous.legacy
sha = previous.tensor_sha
describe = previous.describe


def batches(count):
    if type(count) is not int or count <= 0: raise ValueError('Positive tile count required')
    return [list(range(i, min(i+2, count))) for i in range(0, count, 2)]


def validate_input(x):
    if x.ndim != 4 or x.shape[0] not in (1, 2) or x.shape[1:] != (3,512,512) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('Only unchanged FP32 1/2-tile tensors supported')


def dispatch(original, candidate, x):
    validate_input(x)
    if original.errors or candidate.errors: raise RuntimeError('Unhealthy route; no error fallback')
    out = (original if len(x) == 1 else candidate).run(None, {'images':x})[0]
    if out.shape != (len(x),1,512,512) or out.dtype != np.float32 or not np.isfinite(out).all():
        raise ValueError('Invalid routed output')
    return out


def verify_page(kind, ref_hashes, cand_hashes, sizes):
    legacy.verify_usage(kind, ref_hashes, cand_hashes)
    expected = [len(g) for g in batches(len(ref_hashes))] if ref_hashes else []
    if sizes != expected: raise ValueError('Missing, reordered or padded route')


def metadata(stage):
    ex = json.loads((stage/'export.json').read_text()); b = json.loads((stage/'build.json').read_text())
    if (any(b[k] != v or ex[k] != v for k,v in build.provenance().items())
            or ex['status'] != 'exported_synthetic_consistent_not_business_validated'
            or not all(v['allclose'] for v in ex['checks'].values())
            or b['status'] != 'built_not_validated' or b['flags'] != 0
            or b['workspace_bytes'] != build.WORKSPACE or b['avg_timing_iterations'] != 8
            or b['export_sha256'] != q.digest(stage/'export.json') or b['onnx_sha256'] != ex['onnx_sha256']
            or q.digest(stage/'enhance_batch2.onnx') != b['onnx_sha256']):
        raise ValueError('Build/export source gate differs')
    build.contract([tuple(v) for v in b['inputs']], [tuple(v) for v in b['outputs']])
    return dict(script_sha256=q.digest(__file__), build_script_sha256=q.digest(build.__file__),
        build_sha256=q.digest(stage/'build.json'), export_sha256=q.digest(stage/'export.json'),
        engine_sha256=b['engine_sha256'], baseline_engine_sha256=b['baseline_engine_sha256'],
        onnx_sha256=b['onnx_sha256'], deployment_sha256=b['deployment_sha256'], plan_sha256=b['plan_sha256'],
        helpers_sha256={Path(m.__file__).name:q.digest(m.__file__) for m in
                       (previous, previous.build, legacy, common, q, build.source)},
        diagnostic_helper_sha256=q.digest(Path(__file__).with_name('diagnose_rec_difference.py')))


def candidate(stage):
    from trt82_backend import Session
    return Session(stage/'enhance_batch2.engine', metadata(stage)['engine_sha256'], 'enhance_batch2_fp32')


def original(cfg):
    from trt82_backend import Session
    spec = cfg['engines']['enhance']
    return Session(q.RELEASE/spec['path'], spec['sha256'], 'enhance_original_batch1')


def require(stage, name, status):
    r = json.loads((stage/name).read_text())
    if r['status'] != status or any(r[k] != v for k,v in metadata(stage).items()):
        raise ValueError('Prior gate/source differs')
    return r


def check(a):
    target = a.stage/'check.json'; common.fresh(target); common.fresh(a.stage/'layers.json')
    report = dict(metadata(a.stage), status='running', rows=[], pairs=[])
    single = trial = None
    try:
        rows,cfg = previous.capture(a,3); gc.collect(); single=original(cfg); trial=candidate(a.stage)
        for offset in range(0,27,9):
            for group in batches(9):
                indices = [offset+i for i in group]
                x = np.concatenate([rows[i]['x'] for i in indices])
                out = dispatch(single,trial,x); repeat=dispatch(single,trial,x)
                if not np.array_equal(out,repeat): raise ValueError('Candidate repeat drift')
                swapped = dispatch(single,trial,x[::-1].copy())[::-1] if len(x)==2 else out
                swap = build.compare(out,swapped)
                report['pairs'].append(dict(indices=indices,batch=len(x),swap=swap,repeat_exact=True))
                if not swap['allclose']: raise ValueError('Batch order influenced numerical result')
                for slot,index in enumerate(indices):
                    row=rows[index]; value=out[slot:slot+1]
                    report['rows'].append(dict(describe(row),index=index,batch=len(x),slot=slot,
                        **legacy.compare_arrays(row['reference'],value),candidate_sha256=sha(value),repeat_exact=True))
        inspector=trial.engine.create_engine_inspector(); inspector.execution_context=trial.context
        info=json.loads(inspector.get_engine_information(trial.trt.LayerInformationFormat.JSON))
        details=previous.inspect(info)
        if details['layer_count'] != trial.engine.num_layers: raise ValueError('Incomplete inspection')
        q.save(a.stage/'layers.json',info)
        report.update(status='finite_original_tiles_not_business_accepted',inspection=details,
            execution=trial.stats(),tail_execution=single.stats(),
            note='27 original tiles, paired inside each page with original odd tail. Repeat/swap and numeric checks diagnostic; no GT/benchmark.')
    except Exception as exc:
        report.update(status='failed',error=type(exc).__name__); raise
    finally:
        if trial is not None: trial.close()
        if single is not None: single.close()
        q.save(target,report)
    print(json.dumps(dict(status=report['status'],tiles=len(report['rows']),allclose=sum(r['allclose_diagnostic'] for r in report['rows']))),flush=True)


def evaluate(a):
    require(a.stage,'check.json','finite_original_tiles_not_business_accepted')
    target=a.stage/'evaluation.json'; common.fresh(target)
    plan=q.load_plan(q.BASE/'staging/quant_yolo_e92_r1'); images=plan['calibration']+plan['holdout']
    report=dict(metadata(a.stage),status='running',rows=[],note='Original batch1 vs batch2+original tail. 13 enhanced +1 ordinary; no new text normalization or GT.')
    models=trial=pipeline=old_model=None
    try:
        import torch
        with open(os.devnull,'w') as quiet:
            with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                server,pipeline,models,cfg=common.setup(a)
            base=models.models['enhance']; trial=candidate(a.stage)
            old_model=pipeline._imgpre_model; old_batch=pipeline._imgpre_batch
            active='reference'; hashes=None; sizes=None
            class Routed(torch.nn.Module):
                def forward(self,value):
                    x=value.detach().cpu().numpy(); validate_input(x)
                    hashes[active].extend(sha(x[i:i+1]) for i in range(len(x)))
                    if active=='candidate':
                        sizes.append(len(x)); y=dispatch(base,trial,x)
                    else:
                        if len(x)!=1: raise ValueError('Original batching changed')
                        y=base.run(None,{'images':x})[0]
                    return torch.from_numpy(y)
            pipeline._imgpre_model=Routed().eval()
            for image,kind in [(images[0],3)]+[(r,1) for r in images]:
                hashes={v:[] for v in ('reference','candidate')}; sizes=[]; outputs={}; wall={}; counts={}
                order=('reference','candidate') if len(report['rows'])%2==0 else ('candidate','reference')
                before_trial=trial.calls
                for active in order:
                    pipeline._imgpre_batch=1 if active=='reference' else 2
                    before=models.stats(); start=time.perf_counter()
                    with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                        result=pipeline.process(str(Path(plan['images'])/image['file']),image_type=kind,run_ner=True)
                    wall[active]=(time.perf_counter()-start)*1000; after=models.stats()
                    counts[active]={k:after[k]['cpu_fallback']['calls']-before[k]['cpu_fallback']['calls'] for k in ('det','rec')}
                    if not result['success'] or not models.healthy or trial.errors: raise RuntimeError('Full pipeline/backend failed')
                    outputs[active]=result['data']
                verify_page(kind,hashes['reference'],hashes['candidate'],sizes)
                if trial.calls-before_trial != sizes.count(2): raise ValueError('Candidate bypass/count mismatch')
                ref,out=outputs['reference'],outputs['candidate']; row=q.strict_compare(ref,out)
                fields=q.field_diagnostics(ref,out,server._layout_special_text)
                row.update(image_sha256=image['sha256'],image_type=kind,order=order,wall_ms=wall,
                    same_original_inputs=True,tile_input_sha256=hashes['reference'],tile_calls=len(hashes['reference']),
                    candidate_batch_sizes=sizes,batch2_calls=sizes.count(2),original_tail_calls=sizes.count(1),
                    cpu_fallback_calls=counts,field_diagnostics=fields,
                    full_text_diagnostics=legacy.text_diagnostics(ref['FullText'],out['FullText']),strict_objects_passed=row['passed'])
                row['passed']=legacy.business_passed(row['strict_equal'],fields); report['rows'].append(row)
                print(json.dumps(dict(case=len(report['rows']),passed=row['passed'],batches=sizes)),flush=True)
            report.update(status='all_cases_consistent_not_gt' if legacy.quality_ok(report['rows']) else 'differences_not_accepted',
                          candidate_execution=trial.stats(),original_execution=models.stats())
    except Exception as exc:
        report.update(status='failed',error=type(exc).__name__); raise
    finally:
        if pipeline is not None and old_model is not None:
            pipeline._imgpre_model=old_model; pipeline._imgpre_batch=old_batch
        if trial is not None: trial.close()
        common.close_models(models); q.save(target,report)
    print(json.dumps(dict(status=report['status'],passed=sum(r['passed'] for r in report['rows']))),flush=True)


def benchmark(a):
    quality=require(a.stage,'evaluation.json','all_cases_consistent_not_gt')
    if not previous.quality_ready(quality['rows']): raise ValueError('Business/fallback gate not passed')
    checked=require(a.stage,'check.json','finite_original_tiles_not_business_accepted')
    target=a.stage/('benchmark_%s_%s.json'%(a.variant,a.round)); common.fresh(target)
    report=dict(metadata(a.stage),status='running',variant=a.variant,round=a.round,rows=[],warmup=3,repeat=20,
        note='Same 9 original tiles per sequence; candidate 4 batch2+1 original tail, reference 9 single. Time includes packing/dispatch/Session checks,copies,execute,sync. No split/merge/full OCR/HTTP, no profiler; tails exploratory.')
    single=trial=None
    try:
        rows,cfg=previous.capture(a,1); gc.collect(); expected=checked['rows'][:9]
        if [describe(r) for r in rows]!=[describe(r) for r in expected]: raise ValueError('Input drift')
        report.update(inputs=[describe(r) for r in rows],before=common.telemetry())
        single=original(cfg)
        if a.variant=='candidate': trial=candidate(a.stage)
        wanted=[r['candidate_sha256' if a.variant=='candidate' else 'reference_sha256'] for r in expected]
        def sequence():
            outputs=[]
            groups=batches(9) if trial is not None else [[i] for i in range(9)]
            for group in groups:
                x=np.concatenate([rows[i]['x'] for i in group]) if len(group)==2 else rows[group[0]]['x']
                out=dispatch(single,trial,x) if trial is not None else single.run(None,{'images':x})[0]
                outputs.extend(out[i:i+1].copy() for i in range(len(group)))
            return outputs
        def probe():
            if [sha(v) for v in sequence()] != wanted: raise ValueError('Output drift')
        for _ in range(3): probe()
        report['after_warmup']=common.telemetry()
        for cycle in range(20):
            start=time.perf_counter(); sequence()
            report['rows'].append(dict(cycle=cycle,wall_ms=(time.perf_counter()-start)*1000))
        probe()
        report.update(status='measured',pre_post_outputs_exact=True,timing=common.summary([r['wall_ms'] for r in report['rows']]),
                      execution=single.stats(),batch2_execution=trial.stats() if trial is not None else None,
                      after=common.telemetry(),context_device_bytes=single.engine.device_memory_size+(trial.engine.device_memory_size if trial is not None else 0))
    except Exception as exc:
        report.update(status='failed',error=type(exc).__name__); raise
    finally:
        if trial is not None: trial.close()
        if single is not None: single.close()
        q.save(target,report)
    print(json.dumps(dict(variant=a.variant,round=a.round,timing=report['timing'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=('check','evaluate','benchmark'))
    p.add_argument('--stage',type=Path,required=True); p.add_argument('--variant',choices=('reference','candidate'),default='candidate')
    p.add_argument('--round',choices=('r1','r2'),default='r1'); a=p.parse_args(); a.stage=a.stage.resolve()
    if platform.machine()!='aarch64' or not a.stage.is_relative_to(q.BASE/'staging') or not a.stage.is_dir(): p.error('Existing isolated board stage required')
    sys.path.insert(0,str(q.RELEASE/'edge_deploy/jetson'))
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE),AUTOOCR_RUNTIME_DIR=str(a.stage/'runtime'),AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL); globals()[a.mode](a)
