"""E120 quality-gated same-board FP32 structural overhead, separate spawned workers."""
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np
import archive_ner_protected as previous

p,n,q,old=previous.p,previous.n,previous.q,previous.old
h=p.h
STAGE=q.BASE/'staging/ner_overhead_e120_r2'
PRIOR=p.STAGE
PRIOR_SHA='7dcc6aa99ebe09e2d7819ef33501cc642390108c9e93bd84ece74b3ceead2bc5'
AUDIT_SHA='7b5f81ce1f109b70ecccaea7d5eb5becb61b0e748a809c8c6696eff41a168cf1'
ORDER=(('r1','reference'),('r1','candidate'),('r2','candidate'),('r2','reference'))
RULES=dict(inputs=6,warmup_cycles=3,measured_cycles=10,postcheck_cycles=1,worker_calls=84,
    timed_calls=240,order=[list(v) for v in ORDER],spawn_workers=True,parent_capture_once=True,restore_spawn_environment=True,
    timed_output_hash_checks=True,hash_checks_outside_timing=True,minimum_mean_gain=.05,
    maximum_p95_ratio=1.05,maximum_extra_warm_nvmap_mib=256,maximum_worker_baseline_spread_mib=32,
    worker_timeout_s=300,accepted_for_release=False)
ENV_KEYS=('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','LD_LIBRARY_PATH','LD_PRELOAD','CUDA_CACHE_PATH')


def runtime_environment():return {k:os.environ.get(k) for k in ENV_KEYS}


def restore_environment(before):
    if set(before)!=set(ENV_KEYS):raise ValueError('Unexpected environment keys')
    for key,value in before.items():
        if value is None:os.environ.pop(key,None)
        else:os.environ[key]=value


def provenance():
    if q.digest(PRIOR/'technical_evidence.tar.gz')!=PRIOR_SHA or q.digest(PRIOR/'final_audit.json')!=AUDIT_SHA:raise ValueError('E119 evidence drift')
    meta=p.metadata(PRIOR,True)
    plan,build,check,quality,layers=(n.read(PRIOR/name) for name in ('plan.json','build.json','check.json','quality.json','layers.json'))
    result=previous.summarize(plan,build,check,quality,layers)
    if result['status']!='quality_consistent_not_benchmarked' or result['business_passed_count']!=26:raise ValueError('Full quality gate missing')
    for r in (check,quality):
        if any(r[k]!=v for k,v in meta.items()):raise ValueError('E119 sources differ')
    return dict(candidate_metadata=meta,prior_archive_sha256=PRIOR_SHA,prior_audit_sha256=AUDIT_SHA,
        benchmark_script_sha256=q.digest(__file__),prior_archive_helper_sha256=q.digest(previous.__file__),
        quality_sha256=q.digest(PRIOR/'quality.json'),check_sha256=q.digest(PRIOR/'check.json'))


def prepare(a):
    path=a.stage/'plan.json';old.fresh(path);meta=provenance()
    source=n.read(PRIOR/'plan.json');checked=n.read(PRIOR/'check.json');cfg=n.read(q.RELEASE/'edge_deploy/jetson/deployment.json')
    q.save(path,dict(meta,status='prepared',rules=RULES,inputs=source['rows'],
        expected_hashes={b:[r['output_sha256' if b=='reference' else 'candidate_output_sha256'] for r in checked['rows']] for b in ('reference','candidate')},
        engines=dict(reference=dict(path=str(q.RELEASE/cfg['engines']['ner']['path']),sha256=cfg['engines']['ner']['sha256'],plugins=[]),
            candidate=dict(path=str(PRIOR/'ner_protected.engine'),sha256=meta['candidate_metadata']['candidate_engine_sha256'],plugins=[dict(path=str(path),sha256=sha) for path,sha in p.PLUGINS])),
        note='Freeze before measurement. One parent recreates six original real token feeds in RAM; four fresh spawn workers run sequential ABBA. Parent capture GPU context remains idle and is part of each global NvMap baseline. No raw tokens/logits saved. Existing 5% two-round speed gate; this is an overhead diagnostic, never automatic promotion.'))
    print('Frozen 4 independent workers, 240 timed synchronous NER calls',flush=True)


def metadata(stage):
    meta=provenance();plan=n.read(stage/'plan.json')
    if plan['status']!='prepared' or plan['rules']!=RULES or any(plan[k]!=v for k,v in meta.items()):raise ValueError('Plan drift')
    return dict(meta,plan_sha256=q.digest(stage/'plan.json'))


def snapshot():
    result=old.telemetry();result['monotonic_s']=time.monotonic();temps={}
    for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:temps[(zone/'type').read_text().strip()]=int((zone/'temp').read_text())
        except (OSError,ValueError):pass
    result['temperature_mC']=temps
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):result['process_rss_kib']=int(line.split()[1])
    return result


def warm_nvmap_mib(report):
    before=report['before_load']['memory']['NvMapMemUsed_kib'];after=report['after_warmup']['memory']['NvMapMemUsed_kib']
    if not isinstance(before,int) or not isinstance(after,int) or after<before:raise ValueError('Invalid global warm allocation delta')
    return (after-before)/1024.


def stats(rows):
    return dict(overall=old.summary([r['wall_ms'] for r in rows]),
        per_input={str(i):old.summary([r['wall_ms'] for r in rows if r['index']==i]) for i in range(6)})


def worker(stage,which,round_name,feeds,expected,engine):
    # spawn: no inherited CUDA runtime/context/session from the capture parent.
    from trt82_backend import Session
    path=stage/('benchmark_%s_%s.json'%(which,round_name));old.fresh(path)
    report=dict(status='running',variant=which,round=round_name,pid=os.getpid(),parent_pid=os.getppid(),rows=[],rules=RULES,
        inputs=[n.input_meta(f) for f in feeds],engine=engine,expected_output_sha256=expected,spawn_method=mp.get_start_method(),
        phase_calls=dict(warmup=0,timed=0,postcheck=0),runtime_environment=runtime_environment())
    native=None
    try:
        report['before_load']=snapshot()
        native=Session(Path(engine['path']),engine['sha256'],'ner_'+which,plugins=[(Path(r['path']),r['sha256']) for r in engine['plugins']])
        report['after_load']=snapshot()
        def verify(i,values):
            if len(values)!=1 or h.sha_tensor(values[0])!=expected[i]:raise ValueError('Output drift in performance run')
        for cycle in range(RULES['warmup_cycles']):
            for i,f in enumerate(feeds):verify(i,native.run(None,f));report['phase_calls']['warmup']+=1
        report['after_warmup']=snapshot();report['context_device_bytes']=int(native.engine.device_memory_size)
        report['binding_capacity_bytes']=sum(capacity for ptr,capacity in native.buffers.values())
        for cycle in range(RULES['measured_cycles']):
            for i,f in enumerate(feeds):
                start=time.perf_counter();out=native.run(None,f);elapsed=(time.perf_counter()-start)*1000
                # Hash every measured output after stopping the timer, not just a pre/post sample.
                verify(i,out);report['phase_calls']['timed']+=1;report['rows'].append(dict(cycle=cycle,index=i,wall_ms=elapsed))
        report['after_measure']=snapshot()
        for i,f in enumerate(feeds):verify(i,native.run(None,f));report['phase_calls']['postcheck']+=1
        report.update(status='measured',timing=stats(report['rows']),all_outputs_exact=True,warm_nvmap_delta_mib=warm_nvmap_mib(report),
            note='Synchronized Session.run incl input/range checks, int64-to-int32 cast, dynamic shape, H2D/execute/sync/D2H and finite-output checks. Excludes loading, hash, tokenization, entity decoding and other OCR. Global NvMap delta is not process-private VRAM or service peak. 60 observations/worker, p95 exploratory; clocks/power not fixed.')
    except Exception as err:report.update(status='failed',error=type(err).__name__);raise
    finally:
        if native is not None:
            report['execution']=native.stats()
            try:native.close()
            except Exception as err:
                report.update(status='failed',cleanup_error=type(err).__name__)
        gc.collect();report['after_close']=snapshot();q.save(path,report)
    if report['status']!='measured':raise RuntimeError('Worker cleanup failed')
    print(json.dumps(dict(variant=which,round=round_name,timing=report['timing']['overall'],warm_nvmap_delta_mib=report['warm_nvmap_delta_mib'])),flush=True)


def measure(a):
    path=a.stage/'run.json';old.fresh(path);meta=metadata(a.stage);plan=n.read(a.stage/'plan.json')
    for r,v in ORDER:old.fresh(a.stage/('benchmark_%s_%s.json'%(v,r)))
    report=dict(meta,status='running',rules=RULES,workers=[]);proc=None
    try:
        before_environment=runtime_environment();rows,pages,execution=n.capture(a)
        if [n.describe(r) for r in rows]!=plan['inputs']:raise ValueError('Captured tokens/reference drift')
        feeds=[r['feeds'] for r in rows];del rows;gc.collect()
        after_environment=runtime_environment();restore_environment(before_environment);os.chdir(a.stage)
        report.update(capture_execution=execution,capture_pages=pages,parent_pid=os.getpid(),after_capture=snapshot(),inputs=[n.input_meta(f) for f in feeds])
        report.update(environment_before_capture=before_environment,environment_after_capture=after_environment,environment_for_workers=runtime_environment())
        ctx=mp.get_context('spawn')
        for round_name,which in ORDER:
            before=snapshot();proc=ctx.Process(target=worker,args=(a.stage,which,round_name,feeds,plan['expected_hashes'][which],plan['engines'][which]))
            proc.start();pid=proc.pid;start=time.monotonic()
            while proc.is_alive():
                proc.join(timeout=1.)
                if time.monotonic()-start>RULES['worker_timeout_s']:
                    proc.terminate();proc.join(timeout=10.)
                    if proc.is_alive():proc.kill();proc.join(timeout=10.)
                    raise RuntimeError('Worker timed out')
            report['workers'].append(dict(round=round_name,variant=which,pid=pid,exitcode=proc.exitcode,before_spawn=before,after_exit=snapshot()))
            if proc.exitcode:raise RuntimeError('Worker failed')
            proc.close();proc=None
        report.update(status='measured_not_summarized',benchmark_reports_sha256={f'benchmark_{v}_{r}.json':q.digest(a.stage/f'benchmark_{v}_{r}.json') for r,v in ORDER})
    except Exception as err:report.update(status='failed',error=type(err).__name__);raise
    finally:
        if proc is not None:
            if proc.is_alive():proc.terminate();proc.join(timeout=10.)
            if proc.is_alive():proc.kill();proc.join(timeout=10.)
        q.save(path,report)


def summarize(plan,run,reports):
    if plan['status']!='prepared' or plan['rules']!=RULES or run['rules']!=RULES or run['status']!='measured_not_summarized':raise ValueError('Protocol/run differs')
    if len(plan['inputs'])!=6 or any(len(plan['expected_hashes'][v])!=6 for v in ('reference','candidate')):raise ValueError('Input/output coverage differs')
    if set(reports)!=set(ORDER) or len({w['pid'] for w in run['workers']})!=4:raise ValueError('Worker isolation/coverage differs')
    if [(r['round'],r['variant']) for r in run['workers']]!=list(ORDER) or any(r['exitcode'] for r in run['workers']):raise ValueError('Worker sequence failed')
    if run['capture_pages']!=[dict(image_sha256=r['image_sha256'],image_type=r['image_type'],ner_calls=1) for r in plan['inputs']]:raise ValueError('Parent capture differs')
    if set(run['environment_for_workers'])!=set(ENV_KEYS) or run['environment_for_workers']!=run['environment_before_capture']:raise ValueError('Spawn environment differs')
    previous.execution(run['capture_execution']['ner'],7)
    if any(r['errors'] or r.get('cpu_fallback',{}).get('errors',0) for r in run['capture_execution'].values()):raise ValueError('Capture execution failed')
    expected=[(cycle,i) for cycle in range(10) for i in range(6)]
    for j,(r,v) in enumerate(ORDER):
        b=reports[(r,v)];w=run['workers'][j]
        if b['status']!='measured' or b['rules']!=RULES or (b['round'],b['variant'])!=(r,v) or b['pid']!=w['pid'] or b['spawn_method']!='spawn':raise ValueError('Worker drift')
        if b['parent_pid']!=run['parent_pid'] or b['pid']==run['parent_pid']:raise ValueError('Worker not independent')
        if b['runtime_environment']!=run['environment_for_workers']:raise ValueError('Worker environment differs')
        if b['inputs']!=[row['inputs'] for row in plan['inputs']] or b['inputs']!=run['inputs']:raise ValueError('Timed input drift')
        if b['engine']!=plan['engines'][v] or b['expected_output_sha256']!=plan['expected_hashes'][v] or not b['all_outputs_exact']:raise ValueError('Output/engine drift')
        if [(row['cycle'],row['index']) for row in b['rows']]!=expected or b['timing']!=stats(b['rows']):raise ValueError('Timing coverage/statistics differ')
        if b['execution']['errors'] or b['execution']['calls']!=84 or b['phase_calls']!={'warmup':18,'timed':60,'postcheck':6}:raise ValueError('Execution counts differ')
        if b['warm_nvmap_delta_mib']!=warm_nvmap_mib(b):raise ValueError('Memory arithmetic differs')
    spread=(max(b['before_load']['memory']['NvMapMemUsed_kib'] for b in reports.values())-min(b['before_load']['memory']['NvMapMemUsed_kib'] for b in reports.values()))/1024
    result=[]
    for r in ('r1','r2'):
        x,y=reports[(r,'reference')],reports[(r,'candidate')];a,b=x['timing']['overall'],y['timing']['overall']
        if a['mean_ms']<=0 or a['p95_ms']<=0:raise ValueError('Invalid reference latency')
        ratio=b['mean_ms']/a['mean_ms'];gain=1-ratio;p95=b['p95_ms']/a['p95_ms'];extra=y['warm_nvmap_delta_mib']-x['warm_nvmap_delta_mib']
        result.append(dict(round=r,reference=a,candidate=b,mean_latency_ratio=ratio,mean_gain_fraction=gain,p95_ratio=p95,
            reference_warm_nvmap_mib=x['warm_nvmap_delta_mib'],candidate_warm_nvmap_mib=y['warm_nvmap_delta_mib'],extra_warm_nvmap_mib=extra,
            reference_context_bytes=x['context_device_bytes'],candidate_context_bytes=y['context_device_bytes'],
            performance_gate_passed=gain>=RULES['minimum_mean_gain'] and p95<=RULES['maximum_p95_ratio'] and extra<=RULES['maximum_extra_warm_nvmap_mib']))
    passed=all(r['performance_gate_passed'] for r in result) and spread<=RULES['maximum_worker_baseline_spread_mib']
    return dict(status='performance_gate_passed_not_promoted' if passed else 'performance_gate_not_met',rounds=result,
        worker_baseline_spread_mib=spread,global_memory_comparison_stable=spread<=RULES['maximum_worker_baseline_spread_mib'],
        worker_count=4,timed_calls=240,total_worker_calls=336,quality_source='E119 26/26; every current benchmark output hash exact',
        accepted_for_release=False,note='Same-board synchronous raw NER, not end-to-end OCR, HTTP, kernel timing or GT accuracy. Six fixed input instances with lengths 201..322. P95 exploratory, no locked clocks/power. Global warm NvMap delta is not service peak; process baselines include idle capture parent. No half/INT8 or release change.')


if __name__=='__main__':
    a=h.arguments(('prepare','measure'))
    if a.stage!=STAGE:raise ValueError('Unexpected stage')
    {'prepare':prepare,'measure':measure}[a.mode](a)
