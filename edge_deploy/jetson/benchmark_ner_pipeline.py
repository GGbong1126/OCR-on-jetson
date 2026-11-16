"""E121 full offline OCR pipeline ABBA, one seven-model deployment per process."""
import contextlib
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from types import SimpleNamespace
import archive_ner_overhead as prior

b,n,q,old=prior.b,prior.n,prior.q,prior.old
STAGE=q.BASE/'staging/ner_pipeline_e121_r1'
PRIOR=b.STAGE
PRIOR_SHA='70a83ffbf0d11ce04e8d01cee36a3ad2a5af7d03d3b2703743a970197d6280f4'
AUDIT_SHA='54db8bd1046f7d8e58325ab57dc1dbc8522993e436286d63592343ac83ad5bc1'
ORDER=b.ORDER
RULES=dict(cases=26,warmup_cycles=1,measured_cycles=2,measured_pages=208,total_pages=312,
    order=[list(v) for v in ORDER],reverse_cases_r2=True,spawn_workers=True,one_ner_session=True,
    minimum_mean_gain=.05,maximum_p95_ratio=1.05,maximum_path_mean_ratio=1.05,
    maximum_extra_warm_nvmap_mib=256,maximum_worker_baseline_spread_mib=32,worker_timeout_s=2400,
    accepted_for_release=False)


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def provenance():
    if q.digest(PRIOR/'technical_evidence.tar.gz')!=PRIOR_SHA or q.digest(PRIOR/'final_audit.json')!=AUDIT_SHA:raise ValueError('E120 evidence changed')
    result=prior.validate_reports(PRIOR)
    if result['status']!='performance_gate_passed_not_promoted':raise ValueError('NER prerequisite not passed')
    return dict(prior_archive_sha256=PRIOR_SHA,prior_audit_sha256=AUDIT_SHA,ner_provenance=b.provenance(),
        script_sha256=q.digest(__file__),prior_helper_sha256=q.digest(prior.__file__))


def prepare(a):
    path=a.stage/'plan.json';old.fresh(path);meta=provenance()
    data=q.load_plan(q.BASE/'staging/quant_yolo_e92_r1');quality=n.read(b.PRIOR/'quality.json')
    cases=[]
    for i,(image,kind) in enumerate((im,k) for im in data['calibration']+data['holdout'] for k in (3,1)):
        row=quality['rows'][i]
        if (row['image_sha256'],row['image_type'])!=(image['sha256'],kind) or not row['passed']:raise ValueError('Quality coverage differs')
        cases.append(dict(index=i,image_sha256=image['sha256'],image_type=kind,
            expected={v:dict(inputs=row['inputs'][v],outputs=row['ner_output_sha256'][v],business_sha256=row['business_sha256'][v],
                cpu_fallback_calls=row['cpu_fallback_calls'][v]) for v in ('reference','candidate')}))
    q.save(path,dict(meta,status='prepared',rules=RULES,cases=cases,engines=n.read(PRIOR/'plan.json')['engines'],
        note='Full offline process(path,image_type,run_ner=True), including read/decode and all original modules/postprocess. No HTTP/activation/startup. Two timed repeats per case per worker; exploratory mixed-case percentiles, not production tails. E119 outputs/inputs checked outside timer. No tokens/logits/text saved.'))


def metadata(stage):
    meta=provenance();plan=n.read(stage/'plan.json')
    if plan['status']!='prepared' or plan['rules']!=RULES or any(plan[k]!=v for k,v in meta.items()):raise ValueError('Plan drift')
    return dict(meta,plan_sha256=q.digest(stage/'plan.json'))


def counters(after,before):
    result={}
    if set(after)!=set(before):raise ValueError('Model inventory changed')
    for role,row in after.items():
        if row['errors'] or row.get('cpu_fallback',{}).get('errors',0):raise ValueError('Backend error')
        calls=row['calls']-before[role]['calls']
        fallback=row.get('cpu_fallback',{}).get('calls',0)-before[role].get('cpu_fallback',{}).get('calls',0)
        if calls<0 or fallback<0:raise ValueError('Counter reset')
        result[role]=dict(calls=calls,cpu_fallback_calls=fallback)
    return result


def signature(data,special_text):
    business=digest_json({k:data[k] for k in ('FullText','Text','MarkdownBodyLines','NER')})
    # Hash only deterministic business content, never timings or raw documents.
    extra=digest_json(dict(special=data['special'],Box=data['Box'],raw_line_texts=[r['text'] for r in data['RawLines']],
        api_fields={k:special_text(data['special'],k) for k in ('title','category_no','page_no')}))
    return dict(business_sha256=business,structure_fields_sha256=extra)


def page_indices(round_name):return list(range(26)) if round_name=='r1' else list(reversed(range(26)))


def timing(rows):
    return dict(overall=old.summary([r['wall_ms'] for r in rows]),
        by_path={str(k):old.summary([r['wall_ms'] for r in rows if r['image_type']==k]) for k in (3,1)},
        ner=old.summary([r['ner_wall_ms'] for r in rows]),
        per_case={str(i):old.summary([r['wall_ms'] for r in rows if r['index']==i]) for i in range(26)})


def setup(a,variant,engine):
    # Patch only this fresh process's constructor before attach/warmup. Never load
    # original and candidate NER together, and never edit deployment.json.
    from serve_jetson import deployment,load_models
    from jetson_models import JetsonModels
    from trt82_backend import Session
    server,root,cfg=deployment();native=JetsonModels.native;loads=[]
    def selected(self,role):
        if role!='ner':return native(self,role)
        loads.append(engine['sha256'])
        return Session(Path(engine['path']),engine['sha256'],'ner',plugins=[(Path(p['path']),p['sha256']) for p in engine['plugins']])
    JetsonModels.native=selected
    try:pipeline,models=load_models(server,root,cfg)
    finally:JetsonModels.native=native
    if loads!=[engine['sha256']] or server.cfg.NER_SCORE_THRESHOLD!=.98:raise ValueError('NER loading/threshold differs')
    return server,pipeline,models,loads


def worker(stage,variant,round_name,plan):
    path=stage/('pipeline_%s_%s.json'%(variant,round_name));old.fresh(path)
    a=SimpleNamespace(stage=stage);models=base=original=None
    report=dict(status='running',rules=RULES,variant=variant,round=round_name,pid=os.getpid(),parent_pid=os.getppid(),
        spawn_method=mp.get_start_method(),startup_environment=b.runtime_environment(),engine=plan['engines'][variant],rows=[],warmup=[])
    try:
        data=q.load_plan(q.BASE/'staging/quant_yolo_e92_r1')
        images=data['calibration']+data['holdout']
        report['before_load']=b.snapshot()
        with open(os.devnull,'w') as quiet:
            with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):server,pipeline,models,loads=setup(a,variant,plan['engines'][variant])
            report.update(ner_loads=loads,after_load=b.snapshot(),after_setup_execution=models.stats())
            base=models.models['ner'];original=base.run;captured=[]
            def call(names,feeds):
                start=time.perf_counter();out=original(names,feeds);elapsed=(time.perf_counter()-start)*1000
                # Retain owned output and input references only until page ends;
                # hash/validation is outside the full-pipeline timing interval.
                captured.append((feeds,out,elapsed));return out
            guard=b.h.GuardedCall(call);base.run=guard
            for phase,cycles in (('warmup',1),('rows',2)):
                for cycle in range(cycles):
                    for i in page_indices(round_name):
                        case=plan['cases'][i];image=images[i//2];expected=case['expected'][variant]
                        if image['sha256']!=case['image_sha256']:raise ValueError('Image ordering changed')
                        captured.clear();before=models.stats()
                        with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                            start=time.perf_counter();value=pipeline.process(str(Path(data['images'])/image['file']),image_type=case['image_type'],run_ner=True)
                            wall=(time.perf_counter()-start)*1000
                        guard.require_healthy()
                        if not value['success'] or not models.healthy:raise ValueError('Incomplete business result')
                        inputs=[n.input_meta(f) for f,out,t in captured]
                        if any(len(out)!=1 for f,out,t in captured):raise ValueError('NER contract changed')
                        outputs=[b.h.sha_tensor(out[0]) for f,out,t in captured]
                        if inputs!=expected['inputs'] or outputs!=expected['outputs']:raise ValueError('NER input/output drift')
                        sig=signature(value['data'],server._layout_special_text);delta=counters(models.stats(),before)
                        fallback={k:delta[k]['cpu_fallback_calls'] for k in ('det','rec')}
                        if sig['business_sha256']!=expected['business_sha256'] or fallback!=expected['cpu_fallback_calls'] or delta['ner']['calls']!=len(inputs):raise ValueError('Business/fallback/calls drift')
                        row=dict(index=i,cycle=cycle,image_sha256=case['image_sha256'],image_type=case['image_type'],wall_ms=wall,
                            ner_wall_ms=sum(t for f,out,t in captured),signature=sig,ner_inputs=inputs,ner_outputs=outputs,counters=delta)
                        report[phase].append(row);captured.clear();del value
                    print('pipeline',round_name,variant,phase,cycle+1,len(report[phase]),flush=True)
                if phase=='warmup':report.update(after_warmup=b.snapshot(),after_warmup_execution=models.stats())
            report.update(status='measured',timing=timing(report['rows']),after_measure=b.snapshot(),execution=models.stats(),
                warm_nvmap_delta_mib=b.warm_nvmap_mib(report),
                note='One complete seven-model pipeline per process, same original front/back processing. Per-call NER timer/list append included; all hashing/result checks outside page timer. Warm page cache and two timed repeats/case. Offline file-to-result, not HTTP/startup/production-tail/GT accuracy. NvMap global warm increment is not peak.')
    except Exception as err:report.update(status='failed',error=type(err).__name__);raise
    finally:
        if original is not None:base.run=original
        try:old.close_models(models)
        except Exception as err:report.update(status='failed',cleanup_error=type(err).__name__)
        gc.collect();report['after_close']=b.snapshot();q.save(path,report)
    if report['status']!='measured':raise RuntimeError('Cleanup failed')
    print(json.dumps(dict(round=round_name,variant=variant,timing=report['timing'],warm_nvmap_delta_mib=report['warm_nvmap_delta_mib'])),flush=True)


def measure(a):
    path=a.stage/'run.json';old.fresh(path);meta=metadata(a.stage);plan=n.read(a.stage/'plan.json')
    for r,v in ORDER:old.fresh(a.stage/('pipeline_%s_%s.json'%(v,r)))
    report=dict(meta,status='running',rules=RULES,parent_pid=os.getpid(),startup_environment=b.runtime_environment(),workers=[]);proc=None
    try:
        ctx=mp.get_context('spawn')
        for r,v in ORDER:
            proc=ctx.Process(target=worker,args=(a.stage,v,r,plan));proc.start();pid=proc.pid;start=time.monotonic()
            while proc.is_alive():
                proc.join(timeout=1.)
                if time.monotonic()-start>RULES['worker_timeout_s']:raise RuntimeError('Worker timeout')
            report['workers'].append(dict(round=r,variant=v,pid=pid,exitcode=proc.exitcode))
            if proc.exitcode:raise RuntimeError('Worker failed')
            proc.close();proc=None
        report.update(status='measured_not_summarized',reports_sha256={f'pipeline_{v}_{r}.json':q.digest(a.stage/f'pipeline_{v}_{r}.json') for r,v in ORDER})
    except Exception as err:report.update(status='failed',error=type(err).__name__);raise
    finally:
        if proc is not None:
            if proc.is_alive():proc.terminate();proc.join(timeout=10.)
            if proc.is_alive():proc.kill();proc.join(timeout=10.)
        q.save(path,report)


def summarize(plan,run,reports):
    if plan['status']!='prepared' or plan['rules']!=RULES or run['rules']!=RULES or run['status']!='measured_not_summarized':raise ValueError('Protocol differs')
    if len(plan['cases'])!=26 or [r['index'] for r in plan['cases']]!=list(range(26)):raise ValueError('Cases incomplete')
    if set(reports)!=set(ORDER) or [(w['round'],w['variant']) for w in run['workers']]!=list(ORDER) or len({w['pid'] for w in run['workers']})!=4:raise ValueError('Worker order/isolation differs')
    signatures={}
    for (r,v),w in zip(ORDER,run['workers']):
        x=reports[(r,v)]
        if w['exitcode'] or x['status']!='measured' or x['rules']!=RULES or (x['round'],x['variant'])!=(r,v) or x['pid']!=w['pid'] or x['parent_pid']!=run['parent_pid'] or x['pid']==run['parent_pid'] or x['spawn_method']!='spawn':raise ValueError('Worker failed')
        if x['startup_environment']!=run['startup_environment'] or x['engine']!=plan['engines'][v] or x['ner_loads']!=[x['engine']['sha256']]:raise ValueError('Environment/engine differs')
        if x['timing']!=timing(x['rows']) or x['warm_nvmap_delta_mib']!=b.warm_nvmap_mib(x):raise ValueError('Stored statistics differ')
        if set(x['execution'])!={'det','rec','layout','cell','yolo','enhance','ner'}:raise ValueError('Full deployment missing')
        for phase,cycles,before,after in (('warmup',1,x['after_setup_execution'],x['after_warmup_execution']),('rows',2,x['after_warmup_execution'],x['execution'])):
            rows=x[phase]
            if [(z['cycle'],z['index']) for z in rows]!=[(c,i) for c in range(cycles) for i in page_indices(r)]:raise ValueError('Page coverage differs')
            delta=counters(after,before)
            for role,values in delta.items():
                for key,total in values.items():
                    if total!=sum(z['counters'][role][key] for z in rows):raise ValueError('Aggregate call counts differ')
            for z in rows:
                case=plan['cases'][z['index']];expected=case['expected'][v]
                if (z['image_sha256'],z['image_type'])!=(case['image_sha256'],case['image_type']) or z['ner_inputs']!=expected['inputs'] or z['ner_outputs']!=expected['outputs']:raise ValueError('Case/input/output differs')
                if not z['ner_inputs'] or z['counters']['ner']['calls']!=len(z['ner_inputs']) or z['signature']['business_sha256']!=expected['business_sha256']:raise ValueError('Missing NER/business drift')
                if {k:z['counters'][k]['cpu_fallback_calls'] for k in ('det','rec')}!=expected['cpu_fallback_calls']:raise ValueError('Fallback changed')
                old.summary([z['wall_ms'],z['ner_wall_ms']])
                if z['ner_wall_ms']>z['wall_ms']:raise ValueError('Nested timing invalid')
                key=z['index'];sig=z['signature']
                if key in signatures and signatures[key]!=sig:raise ValueError('Business/structure not repeated exactly')
                signatures[key]=sig
    spread=(max(x['before_load']['memory']['NvMapMemUsed_kib'] for x in reports.values())-min(x['before_load']['memory']['NvMapMemUsed_kib'] for x in reports.values()))/1024
    results=[]
    for r in ('r1','r2'):
        x,y=reports[(r,'reference')],reports[(r,'candidate')];a,c=x['timing'],y['timing']
        if a['overall']['mean_ms']<=0 or a['overall']['p95_ms']<=0:raise ValueError('Invalid reference time')
        gain=1-c['overall']['mean_ms']/a['overall']['mean_ms'];p95=c['overall']['p95_ms']/a['overall']['p95_ms']
        paths={k:dict(reference_ms=a['by_path'][k]['mean_ms'],candidate_ms=c['by_path'][k]['mean_ms'],gain_fraction=1-c['by_path'][k]['mean_ms']/a['by_path'][k]['mean_ms']) for k in ('3','1')}
        extra=y['warm_nvmap_delta_mib']-x['warm_nvmap_delta_mib']
        results.append(dict(round=r,reference=a['overall'],candidate=c['overall'],mean_gain_fraction=gain,p95_ratio=p95,paths=paths,
            reference_ner=a['ner'],candidate_ner=c['ner'],extra_warm_nvmap_mib=extra,
            performance_gate_passed=gain>=RULES['minimum_mean_gain'] and p95<=RULES['maximum_p95_ratio'] and all(1-z['gain_fraction']<=RULES['maximum_path_mean_ratio'] for z in paths.values()) and extra<=RULES['maximum_extra_warm_nvmap_mib']))
    passed=all(r['performance_gate_passed'] for r in results) and spread<=RULES['maximum_worker_baseline_spread_mib']
    return dict(status='pipeline_performance_gate_passed_not_promoted' if passed else 'pipeline_performance_gate_not_met',rounds=results,
        business_consistent_cases=len(signatures),measured_pages=208,warmup_pages=104,worker_count=4,worker_baseline_spread_mib=spread,
        accepted_for_release=False,note='13 existing development images, two paths, two repeats/case/worker. Not GT or independent evaluation data. Offline path-to-result, not HTTP/startup; warm caches, unlocked clocks. Mixed-case p95 exploratory; no service peak measurement. E120 single-NER gain is not pipeline gain.')


if __name__=='__main__':
    a=b.h.arguments(('prepare','measure'))
    if a.stage!=STAGE:raise ValueError('Unexpected stage')
    {'prepare':prepare,'measure':measure}[a.mode](a)
