"""E119 full FP32 NER with 25 visible stable norms and 12 visible attentions."""
import ctypes
import gc
import json
from pathlib import Path
import re
import time
import numpy as np
import ner_norm_probe as norm
import ner_softmax_probe as soft
import evaluate_ner_mixed as e

n,h,q,old=norm.n,norm.h,norm.q,norm.old
STAGE=q.BASE/'staging/ner_protected_e119_r2'
PRIOR=soft.STAGE
PRIOR_SHA='cf7659a38790a9220994dac0b9d53dd6cf34e26bef14c38cc9ff6aafeac9fbd0'
AUDIT_SHA='c075ac0d3ec44c4d6672bb086be99337a1aa63efe8672f31a58aba63e807dbfc'
NORM_STAGE=q.BASE/'staging/ner_norm_stable_e117_r1'
PLUGINS=((NORM_STAGE/'libner_stable.so','5203a4284e5a9210a58276ffabff774b9d27c863ee1841cbc3bfa7dad1cff53c'),
         (PRIOR/'libner_softmax.so','7a19d9ca75a1c5e75f4f938b73162f263afd85cabf9203f93db8dbe84d945848'))
NORM_TYPE='AutoOCRNerLayerNormStableFP32'
RULES=dict(norms=25,attentions=12,real_inputs=6,boundary_cases=40,business_cases=26,fp32_only=True,
    numeric_allclose_diagnostic_only=True,strict_active_labels=True,strict_entities_threshold_fields=True,
    require_37_visible_fp32_plugins=True,no_extra_cpu_fallback=True,benchmark_calls=0,accepted_for_release=False)


def provenance():
    if q.digest(PRIOR/'technical_evidence.tar.gz')!=PRIOR_SHA or q.digest(PRIOR/'final_audit.json')!=AUDIT_SHA:raise ValueError('E118 drift')
    if q.digest(NORM_STAGE/'technical_evidence.tar.gz')!=soft.PRIOR_SHA:raise ValueError('E117 drift')
    for path,sha in PLUGINS:
        if not re.fullmatch(r'[0-9a-f]{64}',sha):raise ValueError('Malformed pinned plugin SHA256')
        if q.digest(path)!=sha:raise ValueError('Validated plugin changed')
    return dict(original=n.provenance(),prior_archive_sha256=PRIOR_SHA,prior_audit_sha256=AUDIT_SHA,
        norm_archive_sha256=soft.PRIOR_SHA,plugins=[dict(path=str(p),sha256=s) for p,s in PLUGINS],
        script_sha256=q.digest(__file__),norm_helper_sha256=q.digest(norm.__file__),softmax_helper_sha256=q.digest(soft.__file__),evaluator_sha256=q.digest(e.__file__))


def boundaries():
    for length in (2,127,128,512):
        for kind in ('full','prefix','alternating','one','none'):
            for segment in (0,1):
                ids=np.full((1,length),100,np.int64);ids[0,0]=101;ids[0,-1]=102
                mask=np.ones_like(ids)
                if kind=='prefix':mask[:,max(1,length//2):]=0
                elif kind=='alternating':mask[:,1::2]=0
                elif kind=='one':mask[:]=0;mask[:,0]=1
                elif kind=='none':mask[:]=0
                types=np.zeros_like(ids);types[:,length//2:]=segment
                yield dict(length=length,kind=kind,segment=segment),dict(input_ids=ids,attention_mask=mask,token_type_ids=types)


def prepare(a):
    path=a.stage/'plan.json';old.fresh(path)
    m,specs,raw=norm.graph();del m,raw;gc.collect();m,attention=soft.graph();del m;gc.collect()
    if specs!=n.read(norm.STAGE/'plan.json')['norms'] or attention!=n.read(PRIOR/'plan.json')['specs']:raise ValueError('Validated specs differ')
    data=q.load_plan(q.BASE/'staging/quant_yolo_e92_r1')
    q.save(path,dict(provenance(),status='prepared',rules=RULES,norms=specs,attentions=attention,
        rows=n.read(PRIOR/'plan.json')['prior_real_rows'],profile=n.PROFILE,
        boundaries=[dict(k,inputs=n.input_meta(f)) for k,f in boundaries()],
        business_pairs=[dict(image_sha256=r['sha256'],image_type=k) for r in data['calibration']+data['holdout'] for k in (3,1)],
        note='Full FP32 structure control only. Original epsilon, weights, finite mask, tokenizer, aggregation and threshold unchanged. E117 synthetic CPU differences retained. No half/INT8/benchmark/release.'))
    print('Frozen 25 norms, 12 attentions, 6 real /40 boundary /26 business cases',flush=True)


def metadata(stage,built=False):
    meta=provenance();plan=n.read(stage/'plan.json')
    if plan['status']!='prepared' or plan['rules']!=RULES or any(plan[k]!=v for k,v in meta.items()):raise ValueError('Plan drift')
    meta['plan_sha256']=q.digest(stage/'plan.json')
    if built:
        b=n.read(stage/'build.json')
        if b['status']!='built_not_validated' or b['flags']!=1024 or any(b[k]!=v for k,v in meta.items()):raise ValueError('Build drift')
        if q.digest(stage/'ner_protected.engine')!=b['candidate_engine_sha256']:raise ValueError('Engine changed')
        meta.update(build_sha256=q.digest(stage/'build.json'),candidate_engine_sha256=b['candidate_engine_sha256'])
    return meta


def plugin_names():return ['E119_norm_%02d'%i for i in range(25)]+['E119_attention_%02d'%i for i in range(12)]


def replace_output(net,src,new,old_layers):
    name=src.name
    if any(net.get_output(i).name==name for i in range(net.num_outputs)):raise ValueError('Unexpected public output replacement')
    users=[(v,j) for v in old_layers for j in range(v.num_inputs) if v.get_input(j) is not None and v.get_input(j).name==name]
    if not users:raise ValueError('No consumer for protected boundary')
    src.name=name+'__unused_e119';new.name=name
    for v,j in users:v.set_input(j,new)
    return [dict(name=v.name,index=j) for v,j in users]


def rewire(net,trt,specs,attention,raw):
    layers=[net.get_layer(i) for i in range(net.num_layers)]
    tensors={v.get_output(j).name:v.get_output(j) for v in layers for j in range(v.num_outputs) if v.get_output(j) is not None}
    registry=trt.get_plugin_registry();nc=registry.get_plugin_creator(NORM_TYPE,'1','');sc=registry.get_plugin_creator(soft.TYPE,'1','')
    if nc is None or sc is None:raise ValueError('Validated creators missing')
    kept=[];records=[]
    for i,(spec,(_,gamma,beta)) in enumerate(zip(specs,raw)):
        name='E119_norm_%02d'%i;params=[]
        for label,weight in (('gamma',gamma),('beta',beta)):
            value=np.ascontiguousarray(weight.copy());kept.append(value)
            c=net.add_constant((768,),value);c.name=name+'_'+label;params.append(c.get_output(0))
        eps=np.array([spec['epsilon']],np.float32);kept.append(eps)
        plugin=nc.create_plugin(name,trt.PluginFieldCollection([trt.PluginField('epsilon',eps,trt.PluginFieldType.FLOAT32)]))
        if plugin is None:raise ValueError('Norm creation failed')
        kept.append(plugin);layer=net.add_plugin_v2([tensors[spec['input']]]+params,plugin);layer.name=name;layer.set_output_type(0,trt.float32)
        records.append(dict(name=name,kind='norm',original=spec['name'],consumers=replace_output(net,tensors[spec['output']],layer.get_output(0),layers)))
    masks=[net.get_input(i) for i in range(net.num_inputs) if net.get_input(i).name=='attention_mask']
    if len(masks)!=1:raise ValueError('Missing public binary mask')
    cast=net.add_identity(masks[0]);cast.name='E119_mask_cast_fp32';cast.set_output_type(0,trt.float32)
    for i,spec in enumerate(attention):
        name='E119_attention_%02d'%i;plugin=sc.create_plugin(name,trt.PluginFieldCollection([]))
        if plugin is None:raise ValueError('Attention creation failed')
        kept.append(plugin);layer=net.add_plugin_v2([tensors[spec['input']],cast.get_output(0)],plugin);layer.name=name;layer.set_output_type(0,trt.float32)
        records.append(dict(name=name,kind='attention',original=spec['name'],consumers=replace_output(net,tensors[spec['output']],layer.get_output(0),layers)))
    if [r['name'] for r in records]!=plugin_names():raise ValueError('Protection coverage differs')
    return records,kept


def build(a):
    import tensorrt as trt
    path=a.stage/'build.json';engine=a.stage/'ner_protected.engine';old.fresh(path);old.fresh(engine)
    report=dict(metadata(a.stage),status='building');start=None
    try:
        if trt.__version__!=h.TRT_VERSION:raise ValueError('Runtime differs')
        handles=[ctypes.CDLL(str(p),mode=ctypes.RTLD_GLOBAL) for p,s in PLUGINS]
        logger=trt.Logger(trt.Logger.INFO);trt.init_libnvinfer_plugins(logger,'')
        m,specs,raw=norm.graph();del m;gc.collect();plan=n.read(a.stage/'plan.json')
        if specs!=plan['norms']:raise ValueError('Norm specs changed')
        builder=trt.Builder(logger);net=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH));parser=trt.OnnxParser(net,logger)
        if not parser.parse(n.SOURCE.read_bytes()):raise RuntimeError('Original parse failed')
        if {net.get_input(i).name:list(net.get_input(i).shape) for i in range(net.num_inputs)}!={k:[1,-1] for k in n.INPUTS}:raise ValueError('Public inputs differ')
        for i in range(net.num_inputs):net.get_input(i).dtype=trt.int32
        report['boundaries'],kept=rewire(net,trt,specs,plan['attentions'],raw)
        cfg=builder.create_builder_config();cfg.max_workspace_size=512*1024**2;cfg.avg_timing_iterations=1
        cfg.clear_flag(trt.BuilderFlag.FP16);cfg.clear_flag(trt.BuilderFlag.INT8);cfg.clear_flag(trt.BuilderFlag.TF32)
        cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS);cfg.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        report['constraints']=[]
        for i in range(net.num_layers):
            layer=net.get_layer(i);kind=str(layer.type).split('.')[-1]
            indices=[j for j in range(layer.num_outputs) if layer.get_output(j) is not None and layer.get_output(j).dtype==trt.float32 and not layer.get_output(j).is_shape_tensor]
            if indices and kind not in h.COMPUTE|h.STRUCTURAL:raise ValueError('Unhandled float layer '+kind)
            if indices and kind in h.COMPUTE:layer.precision=trt.float32
            for j in indices:layer.set_output_type(j,trt.float32)
            report['constraints'].append(dict(name=layer.name,kind=kind,fp32_outputs=indices))
        profile=builder.create_optimization_profile()
        for i in range(net.num_inputs):
            t=net.get_input(i);t.allowed_formats=1<<int(trt.TensorFormat.LINEAR);profile.set_shape(t.name,*n.PROFILE)
            if [list(s) for s in profile.get_shape(t.name)]!=n.PROFILE:raise ValueError('Profile differs')
        if net.num_outputs!=1 or net.get_output(0).name!='output':raise ValueError('Public output differs')
        out=net.get_output(0);out.dtype=trt.float32;out.allowed_formats=1<<int(trt.TensorFormat.LINEAR)
        cfg.add_optimization_profile(profile);cfg.set_timing_cache(cfg.create_timing_cache(b''),False)
        report.update(profile=n.PROFILE,flags=int(cfg.flags),trt_version=trt.__version__,workspace_bytes=cfg.max_workspace_size,avg_timing_iterations=cfg.avg_timing_iterations)
        start=time.perf_counter();blob=builder.build_serialized_network(net,cfg)
        if blob is None:raise RuntimeError('No candidate engine')
        with engine.open('xb') as f:f.write(bytes(blob))
        report.update(status='built_not_validated',candidate_engine_sha256=q.digest(engine),engine_bytes=engine.stat().st_size)
    except Exception as err:report.update(status='failed',error=type(err).__name__);raise
    finally:
        if start is not None:report['build_s']=time.perf_counter()-start
        q.save(path,report)
    print(json.dumps({k:report[k] for k in ('status','engine_bytes','build_s')}),flush=True)


def candidate(stage):
    from trt82_backend import Session
    return Session(stage/'ner_protected.engine',metadata(stage,True)['candidate_engine_sha256'],'ner_protected_fp32',plugins=PLUGINS)


def inspect(info):
    rows=info.get('Layers',[]);missing=[];wrong=[];found_names=[]
    for name in plugin_names():
        found=[r for r in rows if r.get('Name')==name]
        if len(found)!=1:missing.append(name);continue
        r=found[0];io=r.get('Inputs',[])+r.get('Outputs',[]);expected=4 if '_norm_' in name else 3
        if 'plugin' not in r.get('LayerType','').lower() or len(io)!=expected or any(not re.search(r'\b(float|fp32)\b',v.get('Format/Datatype','').lower()) for v in io):wrong.append(name)
        else:found_names.append(name)
    low=[r.get('Name','') for r in rows if any(re.search(r'\b(half|fp16|int8)\b',v.get('Format/Datatype','').lower()) for v in r.get('Inputs',[])+r.get('Outputs',[]))]
    opaque=[r.get('Name','') for r in rows if 'myelin' in r.get('LayerType','').lower() or 'ForeignNode' in r.get('Name','')]
    return dict(layers=len(rows),visible_protected_plugins=len(found_names),missing=missing,non_fp32_plugins=wrong,low_precision_format_layers=low,
        other_opaque_fusions=len(opaque),precision_boundary_verified=bool(rows) and not(missing or wrong or low),
        note='37 known-source FP32 plugins must be separate visible layers. Remaining network built without FP16/INT8/TF32; other fusion internals not independently enumerated.')


def check(a):
    from trt82_backend import Session
    path=a.stage/'check.json';old.fresh(path);meta=metadata(a.stage,True);plan=n.read(a.stage/'plan.json')
    report=dict(meta,status='running',rules=RULES,rows=[],boundaries=[]);trial=base=None
    try:
        rows,pages,execution=n.capture(a)
        if [n.describe(r) for r in rows]!=plan['rows']:raise ValueError('Original capture drift')
        trial=candidate(a.stage)
        for r in rows:
            y=trial.run(None,r['feeds'])[0];z=trial.run(None,r['feeds'])[0]
            if not np.array_equal(y,z):raise ValueError('Repeat drift')
            report['rows'].append(dict(n.describe(r),metrics=e.compare(r['reference'],y,r['feeds']),repeat_exact=True,candidate_output_sha256=h.sha_tensor(y)))
        cfg=n.read(q.RELEASE/'edge_deploy/jetson/deployment.json');spec=cfg['engines']['ner'];base=Session(q.RELEASE/spec['path'],spec['sha256'],'original_ner_boundary')
        for i,(k,feeds) in enumerate(boundaries()):
            if dict(k,inputs=n.input_meta(feeds))!=plan['boundaries'][i]:raise ValueError('Boundary plan drift')
            x=base.run(None,feeds)[0];y=trial.run(None,feeds)[0];z=trial.run(None,feeds)[0]
            if not np.array_equal(y,z):raise ValueError('Boundary repeat drift')
            report['boundaries'].append(dict(k,inputs=n.input_meta(feeds),metrics=e.compare(x,y,feeds),repeat_exact=True,
                all_token_labels_equal=bool(np.array_equal(x.argmax(-1),y.argmax(-1))),reference_sha256=h.sha_tensor(x),candidate_sha256=h.sha_tensor(y)))
        ins=trial.engine.create_engine_inspector();ins.execution_context=trial.context
        info=json.loads(ins.get_engine_information(trial.trt.LayerInformationFormat.JSON));q.save(a.stage/'layers.json',info)
        report.update(status='finite_checked_not_business_accepted',inspection=inspect(info),original_execution=execution,
            note='Full candidate, original GPU reference, same tokens. Raw allclose diagnostic; active labels/37 FP32 boundaries strict. All-masked active-label result alone is vacuous; all-token diagnostic retained. No benchmark or release.')
    except Exception as err:report.update(status='failed',error=type(err).__name__);raise
    finally:
        if trial is not None:report['candidate_execution']=trial.stats();trial.close()
        if base is not None:report['boundary_reference_execution']=base.stats();base.close()
        q.save(path,report)
    print(json.dumps(dict(status=report['status'],real_label_equal=sum(r['metrics']['active_token_labels_equal'] for r in report['rows']),
        boundary_label_equal=sum(r['metrics']['active_token_labels_equal'] for r in report['boundaries']),inspection=report['inspection'])),flush=True)


def quality(a):
    checked=n.read(a.stage/'check.json')
    if not checked['inspection']['precision_boundary_verified'] or not all(r['metrics']['active_token_labels_equal'] for r in checked['rows']+checked['boundaries']):raise ValueError('Structure/label gate failed; do not run business')
    e.quality(a,metadata_fn=lambda stage:metadata(stage,True),candidate_fn=candidate)


if __name__=='__main__':
    a=h.arguments(('prepare','build','check','quality'))
    if a.stage!=STAGE:raise ValueError('Unexpected stage')
    {'prepare':prepare,'build':build,'check':check,'quality':quality}[a.mode](a)
