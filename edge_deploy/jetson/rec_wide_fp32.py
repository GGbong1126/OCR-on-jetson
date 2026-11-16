"""E123 isolated FP32 wide rec profile. Preserve original tensors and release."""
import contextlib
import copy
import gc
import json
import os
from pathlib import Path
import re
import time
import numpy as np
import archive_rec_fallback as previous

d, q, n, old = previous.d, previous.q, previous.n, previous.old
p = d.p
STAGE = q.BASE / 'staging/rec_wide_e123_r1'
PRIOR = d.STAGE
ARCHIVE_SHA = '14743405f2158c5f132f73b1671a62f2d63186aed7dcbbf59163c8ec9378e125'
AUDIT_SHA = 'dd4117ecb01d134129d13e775ebb7742b416c7893e9ec024060d9292866ea4da'
PROFILE = [[1, 3, 48, 1537], [1, 3, 48, 1696], [1, 3, 48, 2304]]
BOUNDARIES = [1537, 1538, 1696, 2303, 2304]
RULES = dict(profile=PROFILE, boundaries=BOUNDARIES, workspace_bytes=512*1024**2,
             avg_timing_iterations=8, fp16=False, int8=False, tf32=False,
             strict_ctc=True, strict_business=True, original_inputs=True,
             warmup_cycles=1, measured_cycles=2, order=[list(v) for v in p.ORDER],
             minimum_mean_gain=.05, maximum_path_mean_ratio=1.05, maximum_p95_ratio=1.05,
             maximum_extra_warm_nvmap_mib=384, maximum_baseline_spread_mib=32,
             worker_timeout_s=1800, accepted_for_release=False)


def provenance():
    if q.digest(PRIOR/'technical_evidence.tar.gz') != ARCHIVE_SHA or q.digest(PRIOR/'final_audit.json') != AUDIT_SHA:
        raise ValueError('E122 evidence changed')
    previous.validate_reports(PRIOR)
    return dict(script_sha256=q.digest(__file__), prior_archive_sha256=ARCHIVE_SHA,
                prior_audit_sha256=AUDIT_SHA, prior_diagnostic_sha256=q.digest(PRIOR/'diagnostic.json'),
                source_sha256=old.SOURCE_SHA, deployment_sha256=d.provenance()['deployment_sha256'])


def prepare(a):
    old.fresh(a.stage/'plan.json')
    meta = provenance()
    prior = n.read(PRIOR/'diagnostic.json')
    cases = n.read(PRIOR/'plan.json')['cases']
    crops = [dict(case=row['index'], **c) for row in prior['pages'] if row['phase']=='r1'
             for c in row['calls'] if c['route']=='cpu']
    if len(crops) != 4 or sorted(c['input_shapes']['x'][3] for c in crops) != [1622,1695,1697,2081]:
        raise ValueError('Fallback set changed')
    q.save(a.stage/'plan.json',dict(meta,status='prepared',rules=RULES,cases=cases,crops=crops,
        note='Separate FP32 engine, batch1, widths1537..2304 OPT1696; original32..1536 GPU retained. Outside union keeps original CPU. Never resize/pad/truncate or fallback on errors. Fixed gates before build; no automatic promotion.'))


def metadata(stage, built=False):
    meta = provenance(); plan = n.read(stage/'plan.json')
    if plan['status']!='prepared' or plan['rules']!=RULES or any(plan[k]!=v for k,v in meta.items()):
        raise ValueError('Frozen plan drift')
    meta['plan_sha256']=q.digest(stage/'plan.json')
    if built:
        build=n.read(stage/'build.json')
        if build['status']!='built_not_validated' or any(build[k]!=v for k,v in meta.items()) or build['rules']!=RULES:
            raise ValueError('Build provenance drift')
        if q.digest(stage/'rec_wide_fp32.engine')!=build['engine_sha256']:
            raise ValueError('Candidate changed')
        meta.update(build_sha256=q.digest(stage/'build.json'),engine_sha256=build['engine_sha256'])
    return meta


def build(a):
    import tensorrt as trt
    path, engine = a.stage/'build.json', a.stage/'rec_wide_fp32.engine'
    old.fresh(path); old.fresh(engine)
    report=dict(metadata(a.stage),rules=RULES,status='building',pid=os.getpid(),trt_version=trt.__version__)
    start=time.perf_counter()
    try:
        if trt.__version__!='8.2.1.8+localcp310': raise ValueError('Unexpected TensorRT')
        log=trt.Logger(trt.Logger.INFO); builder=trt.Builder(log)
        network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser=trt.OnnxParser(network,log)
        if not parser.parse(old.SOURCE.read_bytes()):
            raise RuntimeError('ONNX parse failed')
        if network.num_inputs!=1 or network.get_input(0).name!='x' or network.num_outputs!=1:
            raise ValueError('Model interface differs')
        config=builder.create_builder_config(); config.max_workspace_size=RULES['workspace_bytes']
        config.avg_timing_iterations=RULES['avg_timing_iterations']
        for flag in (trt.BuilderFlag.FP16,trt.BuilderFlag.INT8,trt.BuilderFlag.TF32): config.clear_flag(flag)
        config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        profile=builder.create_optimization_profile(); profile.set_shape('x',*PROFILE)
        if [list(s) for s in profile.get_shape('x')]!=PROFILE: raise ValueError('Profile not retained')
        if config.add_optimization_profile(profile)!=0: raise ValueError('Profile index differs')
        cache=config.create_timing_cache(b''); config.set_timing_cache(cache,False)
        report.update(flags=int(config.flags),tactic_sources=config.get_tactic_sources(),
                      network_layers=network.num_layers,profile=[list(s) for s in profile.get_shape('x')])
        blob=builder.build_serialized_network(network,config)
        if blob is None: raise RuntimeError('Engine build failed')
        with engine.open('xb') as f: f.write(bytes(blob))
        with (a.stage/'timing.cache').open('xb') as f: f.write(bytes(config.get_timing_cache().serialize()))
        report.update(status='built_not_validated',engine_sha256=q.digest(engine),engine_bytes=engine.stat().st_size)
    except Exception as err:
        report.update(status='failed',error=type(err).__name__); raise
    finally:
        report['build_s']=time.perf_counter()-start; q.save(path,report)
    print(json.dumps(report),flush=True)


def candidate(stage):
    from trt82_backend import Session
    built=n.read(stage/'build.json')
    if built['status']!='built_not_validated' or built['rules']!=RULES: raise ValueError('Build missing')
    session=Session(stage/'rec_wide_fp32.engine',built['engine_sha256'],'rec_wide_fp32')
    if session.profiles!={'x':PROFILE} or any(v!=np.dtype('float32') for v in session.dtypes.values()):
        session.close(); raise ValueError('Engine profile/dtype differs')
    return session


class Route:
    """Union of two GPU profiles inside the unchanged original OCRSession."""
    def __init__(self, base, wide): self.base,self.wide=base,wide
    def __getattr__(self,name): return getattr(self.base,name)
    @property
    def errors(self): return self.base.errors+self.wide.errors
    def in_profile(self,feeds): return self.base.in_profile(feeds) or self.wide.in_profile(feeds)
    def run(self,names,feeds):
        if self.errors or self.base.closed or self.wide.closed: raise RuntimeError('Recognition route unhealthy')
        if self.base.in_profile(feeds): return self.base.run(names,feeds)
        if self.wide.in_profile(feeds): return self.wide.run(names,feeds)
        raise ValueError('Outer OCRSession must handle out-of-union CPU fallback')
    def stats(self):
        a,b=self.base.stats(),self.wide.stats()
        return dict(a,backend='tensorrt82_two_fp32_profiles',calls=a['calls']+b['calls'],errors=self.errors,
                    original_execution=a,wide_execution=b,wide_profiles=self.wide.profiles)
    def close(self):
        try: self.wide.close()
        finally: self.base.close()


def expected_counters(case,variant):
    expected=copy.deepcopy(case['counters'])
    if variant=='candidate':
        expected['rec']['calls']+=expected['rec']['cpu_fallback_calls']
        expected['rec']['cpu_fallback_calls']=0
    elif variant!='reference': raise ValueError('Unknown variant')
    return expected


def inspection(info, count):
    rows=info.get('Layers',[])
    low=[r.get('Name','') for r in rows if any(re.search(r'\b(half|fp16|int8)\b',v.get('Format/Datatype','').lower())
         for v in r.get('Inputs',[])+r.get('Outputs',[]))]
    return dict(layers=len(rows),low_precision_format_layers=low,
                opaque_fusions=sum('myelin' in r.get('LayerType','').lower() or 'ForeignNode' in r.get('Name','') for r in rows),
                passed=len(rows)==count and bool(rows) and not low,
                note='Build disables FP16/INT8/TF32; runtime FP32 I/O verified. Inspector visible formats checked, fused internals not independently enumerated.')


def check(a):
    target=a.stage/'check.json'; old.fresh(target); old.fresh(a.stage/'layers.json')
    report=dict(metadata(a.stage,True),rules=RULES,status='running',pid=os.getpid(),rows=[],boundaries=[])
    plan=n.read(a.stage/'plan.json'); models=trial=rec=original=None; captured=[]
    try:
        data=q.load_plan(q.BASE/'staging/quant_yolo_e92_r1'); images=data['calibration']+data['holdout']
        with open(os.devnull,'w') as quiet:
            with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                server,pipeline,models,cfg=old.setup(a)
            rec=models.models['rec']; original=rec.run; active={}
            def capture(names,feeds):
                out=original(names,feeds)
                if not rec.gpu.in_profile(feeds):
                    captured.append(dict(active,feeds={k:v.copy() for k,v in feeds.items()},output=out[0].copy()))
                return out
            guard=p.b.h.GuardedCall(capture); rec.run=guard
            for i in (14,15):
                case=plan['cases'][i]; active['case']=i; before=models.stats()
                with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                    value=pipeline.process(str(Path(data['images'])/images[i//2]['file']),image_type=case['image_type'],run_ner=True)
                guard.require_healthy()
                if not value['success'] or not models.healthy or p.signature(value['data'],server._layout_special_text)!=case['signature'] or p.counters(models.stats(),before)!=case['counters']:
                    raise ValueError('Capture business drift')
            rec.run=original; original=None
            if len(captured)!=4: raise ValueError('Capture incomplete')
            cpu=rec.cpu
            old.close_models(models); models=None; gc.collect()
            trial=candidate(a.stage)
            for v,expected in zip(captured,plan['crops']):
                feed,reference=v['feeds'],v['output']
                if v['case']!=expected['case'] or {'x':list(feed['x'].shape)}!=expected['input_shapes'] or p.b.h.sha_tensor(feed['x'])!=expected['input_sha256']['x'] or p.b.h.sha_tensor(reference)!=expected['output_sha256']:
                    raise ValueError('E122 original tensor/output drift')
                out=trial.run(None,feed)[0]; again=trial.run(None,feed)[0]
                metrics=old.compare(reference,out)
                report['rows'].append(dict(case=v['case'],shape=list(feed['x'].shape),input_sha256=p.b.h.sha_tensor(feed['x']),
                    reference_output_sha256=p.b.h.sha_tensor(reference),candidate_output_sha256=p.b.h.sha_tensor(out),
                    repeated_exact=p.b.h.sha_tensor(out)==p.b.h.sha_tensor(again),**metrics))
            for width in BOUNDARIES:
                feed={'x':np.zeros((1,3,48,width),dtype=np.float32)}
                ref=cpu.run(None,feed)[0]; out=trial.run(None,feed)[0]; again=trial.run(None,feed)[0]
                report['boundaries'].append(dict(width=width,kind='synthetic_zero_not_text',
                    repeated_exact=p.b.h.sha_tensor(out)==p.b.h.sha_tensor(again),**old.compare(ref,out)))
            inspector=trial.engine.create_engine_inspector(); inspector.execution_context=trial.context
            layers=json.loads(inspector.get_engine_information(trial.trt.LayerInformationFormat.JSON))
            q.save(a.stage/'layers.json',layers)
            report.update(inspection=inspection(layers,trial.engine.num_layers),layers_sha256=q.digest(a.stage/'layers.json'),
                          execution=trial.stats(),boundary_cpu_execution=cpu.stats())
            good=report['inspection']['passed'] and all(r['token_sequence_equal'] and r['repeated_exact'] for r in report['rows']+report['boundaries'])
            report['status']='wide_ctc_consistent_not_gt' if good else 'quality_not_accepted'
    except Exception as err: report.update(status='failed',error=type(err).__name__); raise
    finally:
        if original is not None: rec.run=original
        try:
            if trial is not None: trial.close()
            old.close_models(models)
        except Exception as err: report.update(status='failed',cleanup_error=type(err).__name__)
        q.save(target,report)
    print(json.dumps({k:report.get(k) for k in ('status','rows','boundaries','inspection')}),flush=True)


def require_check(stage):
    meta=metadata(stage,True); checked=n.read(stage/'check.json')
    if checked['status']!='wide_ctc_consistent_not_gt' or any(checked[k]!=v for k,v in meta.items()) or len(checked['rows'])!=4 or len(checked['boundaries'])!=len(BOUNDARIES) or not checked['inspection']['passed']:
        raise ValueError('Same-tensor/shape quality gate missing')
    if q.digest(stage/'layers.json')!=checked['layers_sha256']: raise ValueError('Inspector drift')
    return dict(meta,check_sha256=q.digest(stage/'check.json'))


def evaluate(a):
    path=a.stage/'quality.json'; old.fresh(path)
    report=dict(require_check(a.stage),rules=RULES,status='running',pid=os.getpid(),rows=[])
    plan=n.read(a.stage/'plan.json'); models=trial=base=rec=None
    try:
        data=q.load_plan(q.BASE/'staging/quant_yolo_e92_r1'); images=data['calibration']+data['holdout']
        with open(os.devnull,'w') as quiet:
            with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet): server,pipeline,models,cfg=old.setup(a)
            rec=models.models['rec']; base=rec.gpu; trial=candidate(a.stage); route=Route(base,trial)
            for i,case in enumerate(plan['cases']):
                row=dict(index=i,image_sha256=case['image_sha256'],image_type=case['image_type'],branches={})
                for variant in (('reference','candidate') if i%2==0 else ('candidate','reference')):
                    rec.gpu=base if variant=='reference' else route; before=models.stats(); wb=trial.calls
                    with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                        value=pipeline.process(str(Path(data['images'])/images[i//2]['file']),image_type=case['image_type'],run_ner=True)
                    if not value['success'] or not models.healthy or trial.errors: raise ValueError('Business/backend failed')
                    sig=p.signature(value['data'],server._layout_special_text); counts=p.counters(models.stats(),before)
                    wide=trial.calls-wb
                    wanted=case['counters']['rec']['cpu_fallback_calls'] if variant=='candidate' else 0
                    row['branches'][variant]=dict(signature=sig,counters=counts,wide_calls=wide,
                        passed=sig==case['signature'] and counts==expected_counters(case,variant) and wide==wanted)
                row['passed']=all(r['passed'] for r in row['branches'].values()); report['rows'].append(row)
                print(json.dumps(dict(case=i,passed=row['passed'])),flush=True)
            report.update(status='all_26_business_consistent_not_gt' if len(report['rows'])==26 and all(r['passed'] for r in report['rows']) else 'business_not_accepted',
                          wide_execution=trial.stats(),original_gpu_execution=base.stats())
    except Exception as err: report.update(status='failed',error=type(err).__name__); raise
    finally:
        if rec is not None and base is not None: rec.gpu=base
        try:
            if trial is not None: trial.close()
            old.close_models(models)
        except Exception as err: report.update(status='failed',cleanup_error=type(err).__name__)
        q.save(path,report)
    print(report['status'],flush=True)


if __name__=='__main__':
    a=p.b.h.arguments(('prepare','build','check','evaluate'))
    if a.stage!=STAGE: raise ValueError('Unexpected stage')
    {'prepare':prepare,'build':build,'check':check,'evaluate':evaluate}[a.mode](a)
