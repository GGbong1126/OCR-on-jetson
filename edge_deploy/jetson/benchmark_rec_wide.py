"""E123 quality-gated, four independent full-pipeline workers (ABBA)."""
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import contextlib
from types import SimpleNamespace
import rec_wide_fp32 as w

q,n,old,p=w.q,w.n,w.old,w.p


def validate_quality(stage):
    meta=w.require_check(stage); quality=n.read(stage/'quality.json'); plan=n.read(stage/'plan.json')
    if quality['status']!='all_26_business_consistent_not_gt' or quality['rules']!=w.RULES or any(quality[k]!=v for k,v in meta.items()):
        raise ValueError('Business quality not passed')
    if [r['index'] for r in quality['rows']]!=list(range(26)):
        raise ValueError('Quality coverage differs')
    for row,case in zip(quality['rows'],plan['cases']):
        if not row['passed'] or set(row['branches'])!={'reference','candidate'} or (row['image_sha256'],row['image_type'])!=(case['image_sha256'],case['image_type']):
            raise ValueError('Quality case differs')
        for variant,r in row['branches'].items():
            wanted=case['counters']['rec']['cpu_fallback_calls'] if variant=='candidate' else 0
            if not r['passed'] or r['signature']!=case['signature'] or r['counters']!=w.expected_counters(case,variant) or r['wide_calls']!=wanted:
                raise ValueError('Quality signature/routing differs')
    if quality['wide_execution']['calls']!=4 or quality['wide_execution']['errors'] or quality['original_gpu_execution']['errors']:
        raise ValueError('Quality backend errors')
    return dict(meta,quality_sha256=q.digest(stage/'quality.json'))


def timing(rows):
    return dict(overall=old.summary([r['wall_ms'] for r in rows]),
                by_path={str(k):old.summary([r['wall_ms'] for r in rows if r['image_type']==k]) for k in (3,1)},
                unaffected=old.summary([r['wall_ms'] for r in rows if r['index'] not in (14,15)]),
                per_case={str(i):old.summary([r['wall_ms'] for r in rows if r['index']==i]) for i in range(26)})


def worker(stage,variant,round_name,plan):
    path=stage/('pipeline_%s_%s.json'%(variant,round_name)); old.fresh(path)
    report=dict(status='running',rules=w.RULES,variant=variant,round=round_name,pid=os.getpid(),
                parent_pid=os.getppid(),spawn_method=mp.get_start_method(),startup_environment=p.b.runtime_environment(),
                before_load=p.b.snapshot(),warmup=[],rows=[])
    models=trial=None
    try:
        data=q.load_plan(q.BASE/'staging/quant_yolo_e92_r1'); images=data['calibration']+data['holdout']
        with open(os.devnull,'w') as quiet:
            with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                server,pipeline,models,cfg=old.setup(SimpleNamespace(stage=stage))
            if variant=='candidate':
                trial=w.candidate(stage); rec=models.models['rec']; rec.gpu=w.Route(rec.gpu,trial)
            report.update(after_load=p.b.snapshot(),after_setup_execution=models.stats(),after_setup_wide_calls=0)
            for phase,cycles in (('warmup',w.RULES['warmup_cycles']),('rows',w.RULES['measured_cycles'])):
                for cycle in range(cycles):
                    for i in p.page_indices(round_name):
                        case=plan['cases'][i]; image=images[i//2]
                        if image['sha256']!=case['image_sha256']:raise ValueError('Image changed')
                        before=models.stats(); wb=trial.calls if trial else 0
                        with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                            start=time.perf_counter()
                            value=pipeline.process(str(Path(data['images'])/image['file']),image_type=case['image_type'],run_ner=True)
                            wall=(time.perf_counter()-start)*1000
                        if not value['success'] or not models.healthy:raise ValueError('Incomplete page')
                        sig=p.signature(value['data'],server._layout_special_text); counts=p.counters(models.stats(),before)
                        wide=trial.calls-wb if trial else 0
                        if sig!=case['signature'] or counts!=w.expected_counters(case,variant) or wide!=(case['counters']['rec']['cpu_fallback_calls'] if trial else 0):
                            raise ValueError('Business/call routing drift')
                        report[phase].append(dict(index=i,cycle=cycle,image_sha256=case['image_sha256'],image_type=case['image_type'],
                                                wall_ms=wall,signature=sig,counters=counts,wide_calls=wide))
                        del value
                    print('pipeline',round_name,variant,phase,cycle+1,len(report[phase]),flush=True)
                if phase=='warmup':report.update(after_warmup=p.b.snapshot(),after_warmup_execution=models.stats())
            report.update(status='measured',timing=timing(report['rows']),after_measure=p.b.snapshot(),execution=models.stats(),
                          warm_nvmap_delta_mib=p.b.warm_nvmap_mib(report),wide_execution=trial.stats() if trial else None,
                          note='Offline path-to-result run_ner=True, all original pre/post; no per-call timer/hash probes. Hashes checked after page timer. Includes decode, routing, model copies/execution and business. Excludes loading/HTTP/license. Warm global NvMap delta is not peak.')
    except Exception as err:report.update(status='failed',error=type(err).__name__);raise
    finally:
        try:old.close_models(models)
        except Exception as err:report.update(status='failed',cleanup_error=type(err).__name__)
        gc.collect();report['after_close']=p.b.snapshot();q.save(path,report)
    if report['status']!='measured':raise RuntimeError('Cleanup failed')
    print(json.dumps(dict(round=round_name,variant=variant,timing=report['timing']['overall'],warm_nvmap_delta_mib=report['warm_nvmap_delta_mib'])),flush=True)


def measure(a):
    path=a.stage/'run.json';old.fresh(path);meta=validate_quality(a.stage);plan=n.read(a.stage/'plan.json')
    for r,v in p.ORDER:old.fresh(a.stage/('pipeline_%s_%s.json'%(v,r)))
    report=dict(meta,benchmark_script_sha256=q.digest(__file__),rules=w.RULES,status='running',parent_pid=os.getpid(),
                startup_environment=p.b.runtime_environment(),workers=[]);proc=None
    try:
        ctx=mp.get_context('spawn')
        for r,v in p.ORDER:
            proc=ctx.Process(target=worker,args=(a.stage,v,r,plan));proc.start();pid=proc.pid;start=time.monotonic()
            while proc.is_alive():
                proc.join(timeout=1.)
                if time.monotonic()-start>w.RULES['worker_timeout_s']:raise RuntimeError('Worker timeout')
            report['workers'].append(dict(round=r,variant=v,pid=pid,exitcode=proc.exitcode))
            if proc.exitcode:raise RuntimeError('Worker failed')
            proc.close();proc=None
        report.update(status='measured_not_summarized',reports_sha256={f'pipeline_{v}_{r}.json':q.digest(a.stage/f'pipeline_{v}_{r}.json') for r,v in p.ORDER})
    except Exception as err:report.update(status='failed',error=type(err).__name__);raise
    finally:
        if proc is not None:
            if proc.is_alive():proc.terminate();proc.join(timeout=10.)
            if proc.is_alive():proc.kill();proc.join(timeout=10.)
        q.save(path,report)


def summarize(plan,run,reports):
    if plan['rules']!=w.RULES or run['rules']!=w.RULES or run['status']!='measured_not_summarized' or len(plan['cases'])!=26:
        raise ValueError('Protocol differs')
    if set(reports)!=set(p.ORDER) or [(r['round'],r['variant']) for r in run['workers']]!=list(p.ORDER) or len({r['pid'] for r in run['workers']})!=4:
        raise ValueError('Worker order/isolation differs')
    for (r,v),worker_row in zip(p.ORDER,run['workers']):
        x=reports[(r,v)]
        if worker_row['exitcode'] or x['status']!='measured' or x['rules']!=w.RULES or (x['round'],x['variant'])!=(r,v) or x['pid']!=worker_row['pid'] or x['parent_pid']!=run['parent_pid'] or x['pid']==run['parent_pid'] or x['spawn_method']!='spawn' or x['startup_environment']!=run['startup_environment']:
            raise ValueError('Worker failed or environment differs')
        if x['timing']!=timing(x['rows']) or x['warm_nvmap_delta_mib']!=p.b.warm_nvmap_mib(x):raise ValueError('Stored timing/memory differs')
        if set(x['execution'])!={'det','rec','layout','cell','yolo','enhance','ner'}:raise ValueError('Missing business model')
        for phase,cycles,before,after in (('warmup',1,x['after_setup_execution'],x['after_warmup_execution']),('rows',2,x['after_warmup_execution'],x['execution'])):
            rows=x[phase]
            if [(z['cycle'],z['index']) for z in rows]!=[(c,i) for c in range(cycles) for i in p.page_indices(r)]:raise ValueError('Coverage differs')
            delta=p.counters(after,before)
            for role,values in delta.items():
                for key,total in values.items():
                    if total!=sum(z['counters'][role][key] for z in rows):raise ValueError('Aggregate calls differ')
            for z in rows:
                case=plan['cases'][z['index']]
                if (z['image_sha256'],z['image_type'])!=(case['image_sha256'],case['image_type']) or z['signature']!=case['signature'] or z['counters']!=w.expected_counters(case,v):
                    raise ValueError('Business/routing differs')
                if z['wide_calls']!=(case['counters']['rec']['cpu_fallback_calls'] if v=='candidate' else 0):raise ValueError('Wrong candidate usage')
                old.summary([z['wall_ms']])
        if v=='candidate':
            if x['wide_execution']['calls']!=12 or x['wide_execution']['errors'] or x['wide_execution']['profiles']!={'x':w.PROFILE}:raise ValueError('Candidate not exercised')
        elif x['wide_execution'] is not None:raise ValueError('Baseline changed')
    spread=(max(x['before_load']['memory']['NvMapMemUsed_kib'] for x in reports.values())-min(x['before_load']['memory']['NvMapMemUsed_kib'] for x in reports.values()))/1024
    rounds=[]
    for r in ('r1','r2'):
        x,y=reports[(r,'reference')],reports[(r,'candidate')];a,b=x['timing'],y['timing']
        if a['overall']['mean_ms']<=0 or a['overall']['p95_ms']<=0:raise ValueError('Invalid reference time')
        gain=1-b['overall']['mean_ms']/a['overall']['mean_ms'];ratio=b['overall']['p95_ms']/a['overall']['p95_ms']
        extra=y['warm_nvmap_delta_mib']-x['warm_nvmap_delta_mib']
        paths={k:dict(reference_ms=a['by_path'][k]['mean_ms'],candidate_ms=b['by_path'][k]['mean_ms'],ratio=b['by_path'][k]['mean_ms']/a['by_path'][k]['mean_ms']) for k in ('3','1')}
        unaffected=b['unaffected']['mean_ms']/a['unaffected']['mean_ms']
        passed=gain>=w.RULES['minimum_mean_gain'] and ratio<=w.RULES['maximum_p95_ratio'] and extra<=w.RULES['maximum_extra_warm_nvmap_mib'] and max([unaffected]+[v['ratio'] for v in paths.values()])<=w.RULES['maximum_path_mean_ratio']
        rounds.append(dict(round=r,reference=a['overall'],candidate=b['overall'],mean_gain_fraction=gain,p95_ratio=ratio,paths=paths,
            unaffected_mean_ratio=unaffected,extra_warm_nvmap_mib=extra,
            slow_cases={str(i):dict(reference=a['per_case'][str(i)],candidate=b['per_case'][str(i)]) for i in (14,15)},performance_gate_passed=passed))
    passed=all(r['performance_gate_passed'] for r in rounds) and spread<=w.RULES['maximum_baseline_spread_mib']
    return dict(status='wide_pipeline_gate_passed_not_promoted' if passed else 'wide_pipeline_gate_not_met',rounds=rounds,
                worker_baseline_spread_mib=spread,measured_pages=208,warmup_pages=104,accepted_for_release=False,
                note='Same 13 development images, no GT/independent dataset. CPU fallback to FP32 GPU coverage, not quantization. Full offline OCR+NER, not HTTP; unlocked clocks, exploratory p95. No auto release change.')


if __name__=='__main__':
    a=p.b.h.arguments(('measure',))
    if a.stage!=w.STAGE:raise ValueError('Unexpected stage')
    measure(a)
