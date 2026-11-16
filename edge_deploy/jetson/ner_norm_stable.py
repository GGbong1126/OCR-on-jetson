"""E117 shift-centered FP32 normalization: new FP64 gate, old CPU gate preserved."""
import ctypes
import gc
import json
import time
import numpy as np
import ner_norm_probe as p
import diagnose_ner_norm as d

n,h,q,old=p.n,p.h,p.q,p.old
STAGE=q.BASE/'staging/ner_norm_stable_e117_r1'
PRIOR=p.STAGE
PRIOR_SHA='b15e83117eb3a46b428da656b6b000e0d147373a0f7fa8d61a3793c6baa480a5'
AUDIT_SHA='ae4af8bd3fa73e6870dc3ade25f5b8c135ebafd29e2383cda1302a7414f41e23'
TYPE='AutoOCRNerLayerNormStableFP32'
RULES=dict(rtol=1e-3,atol=1e-4,real_cases=150,synthetic_cases=30,preserve_old_cpu_gate=True,
    new_gate='same_FP32_input_vs_FP64_formula_plus_real_CPU_regression',fp32_only=True,
    near_constant_relative_mae_at_most=.01,benchmark_calls=0,full_ner_candidate=False,accepted_for_release=False)


def synthetic():
    yield from p.synthetic()
    rng=np.random.default_rng(117)
    for length in (2,128,512):
        for base in (-1000.,-1.,0.,1.,1000.):
            # Low-variance, actually representable FP32 perturbations at each offset.
            noise=max(abs(float(np.spacing(np.float32(base))))*4,1e-6)
            x=(base+rng.normal(0,noise,(1,length,768))).astype(np.float32)
            yield length,'offset_'+str(base),x
        x=rng.normal(0,2,(1,length,768)).astype(np.float32);x[:,:,0]=20000.
        yield length,'first_element_outlier',x


def centered(x,gamma,beta,epsilon):
    shifted=x-x[...,:1];mean=d.tree_sum(shifted)/np.float32(768)
    delta=shifted-mean;var=d.tree_sum(delta*delta)/np.float32(768)
    return delta/np.sqrt(var+np.float32(epsilon))*gamma+beta


def metric(reference,y):
    if reference.shape!=y.shape or y.dtype!=np.float32 or not np.isfinite(reference).all() or not np.isfinite(y).all():raise ValueError('Invalid output')
    return dict(d.errors(reference,y),all_finite=True,allclose=bool(np.allclose(y,reference,rtol=RULES['rtol'],atol=RULES['atol'])))


def provenance():
    if q.digest(PRIOR/'technical_evidence.tar.gz')!=PRIOR_SHA or q.digest(PRIOR/'final_audit.json')!=AUDIT_SHA:raise ValueError('E116 changed')
    return dict(prior_metadata=p.plugin_metadata(PRIOR),prior_archive_sha256=PRIOR_SHA,prior_audit_sha256=AUDIT_SHA,
        script_sha256=q.digest(__file__),diagnostic_helper_sha256=q.digest(d.__file__),
        plugin_source_sha256=q.digest(STAGE/'ner_layernorm_stable.cu'),compiler_script_sha256=q.digest(STAGE/'build_grid_plugin.py'))


def prepare(a):
    path=a.stage/'plan.json';old.fresh(path)
    report=dict(provenance(),status='prepared',rules=RULES,old_plan_sha256=q.digest(PRIOR/'plan.json'),old_check_sha256=q.digest(PRIOR/'check.json'),
        epsilon=n.read(PRIOR/'plan.json')['epsilon'],synthetic=[dict(length=l,kind=k,input_sha256=h.sha_tensor(x)) for l,k,x in synthetic()],
        note='New independent mathematical-fidelity experiment, not a relabel of E116. Original CPU comparator preserved. Candidate stability does not restore lost FP16 input information or imply whole-NER equivalence/performance.')
    q.save(path,report);print('Frozen FP64/CPU comparison protocol',flush=True)


def metadata(stage):
    plan=n.read(stage/'plan.json');meta=provenance()
    if plan['status']!='prepared' or plan['rules']!=RULES or any(plan[k]!=v for k,v in meta.items()):raise ValueError('Plan changed')
    if plan['old_plan_sha256']!=q.digest(PRIOR/'plan.json') or plan['old_check_sha256']!=q.digest(PRIOR/'check.json'):raise ValueError('Old check changed')
    return dict(meta,plan_sha256=q.digest(stage/'plan.json'))


def plugin_metadata(stage):
    meta=metadata(stage);b=n.read(stage/'libner_stable.build.json')
    if b['source_sha256']!=meta['plugin_source_sha256'] or b['plugin_sha256']!=q.digest(stage/'libner_stable.so'):raise ValueError('Plugin changed')
    if '--fmad=false' not in b['command'] or '--use_fast_math' in b['command']:raise ValueError('Compiler differs')
    return dict(meta,plugin_sha256=b['plugin_sha256'],plugin_build_sha256=q.digest(stage/'libner_stable.build.json'))


def build(a):
    import tensorrt as trt
    path=a.stage/'build.json';engine=a.stage/'stable.engine';old.fresh(path);old.fresh(engine)
    result=dict(plugin_metadata(a.stage),status='running');start=time.perf_counter()
    try:
        if trt.__version__!=h.TRT_VERSION:raise ValueError('TRT changed')
        handle=ctypes.CDLL(str(a.stage/'libner_stable.so'),mode=ctypes.RTLD_GLOBAL)
        logger=trt.Logger(trt.Logger.WARNING);trt.init_libnvinfer_plugins(logger,'')
        creator=trt.get_plugin_registry().get_plugin_creator(TYPE,'1','')
        if creator is None:raise ValueError('Plugin not registered')
        eps=np.array([n.read(a.stage/'plan.json')['epsilon']],np.float32)
        plugin=creator.create_plugin('stable_norm',trt.PluginFieldCollection([trt.PluginField('epsilon',eps,trt.PluginFieldType.FLOAT32)]))
        if plugin is None:raise ValueError('Plugin creation failed')
        serialized=bytes(plugin.serialize());restored=creator.deserialize_plugin('roundtrip',serialized)
        if restored is None or bytes(restored.serialize())!=serialized:raise ValueError('Serialization differs')
        builder=trt.Builder(logger);net=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH));cfg=builder.create_builder_config()
        cfg.max_workspace_size=64*1024**2;cfg.avg_timing_iterations=1;cfg.clear_flag(trt.BuilderFlag.TF32);cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        cfg.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        x=net.add_input('x',trt.float32,(1,-1,768));gamma=net.add_input('gamma',trt.float32,(768,));beta=net.add_input('beta',trt.float32,(768,))
        layer=net.add_plugin_v2([x,gamma,beta],plugin);layer.name='isolated_norm_fp32';layer.set_output_type(0,trt.float32)
        out=layer.get_output(0);out.name='y';net.mark_output(out);out.dtype=trt.float32;out.allowed_formats=1<<int(trt.TensorFormat.LINEAR)
        profile=builder.create_optimization_profile()
        for t in (x,gamma,beta):
            shapes=[[1,2,768],[1,128,768],[1,512,768]] if t.name=='x' else [[768]]*3
            profile.set_shape(t.name,*shapes);t.allowed_formats=1<<int(trt.TensorFormat.LINEAR)
        cfg.add_optimization_profile(profile);blob=builder.build_serialized_network(net,cfg)
        if blob is None:raise RuntimeError('No engine')
        with engine.open('xb') as f:f.write(bytes(blob))
        result.update(status='built_not_validated',engine_sha256=q.digest(engine),engine_bytes=engine.stat().st_size,flags=int(cfg.flags),serialization_roundtrip=True)
    except Exception as err:result.update(status='failed',error=type(err).__name__);raise
    finally:result['build_s']=time.perf_counter()-start;q.save(path,result)
    print('Stable FP32 micro-engine built',flush=True)


def comparison(x,gamma,beta,epsilon,original,base,trial):
    feeds=dict(x=x,gamma=gamma,beta=beta);b=base.run(None,feeds)[0];y=trial.run(None,feeds)[0];z=trial.run(None,feeds)[0]
    if not np.array_equal(y,z):raise ValueError('Candidate repeat drift')
    oracle=d.precise(x,gamma,beta,epsilon);emu=centered(x,gamma,beta,epsilon)
    return dict(input_sha256=h.sha_tensor(x),input_shape=list(x.shape),old_output_sha256=h.sha_tensor(b),candidate_output_sha256=h.sha_tensor(y),repeat_exact=True,
        original_vs_fp64=metric(oracle,original),old_vs_fp64=metric(oracle,b),candidate_vs_fp64=metric(oracle,y),candidate_vs_original=metric(original,y),
        emulator_vs_candidate=metric(emu,y),emulator_exact=bool(np.array_equal(emu,y)))


def result_status(real,synthetic_rows):
    math_ok=all(r['candidate_vs_fp64']['allclose'] for r in real+synthetic_rows)
    real_ok=all(r['candidate_vs_original']['allclose'] for r in real)
    low=[r for r in synthetic_rows if r['kind']=='near_constant']
    improved=len(low)==3 and all(r['candidate_vs_fp64']['mae']<=RULES['near_constant_relative_mae_at_most']*r['old_vs_fp64']['mae'] for r in low)
    return 'isolated_stability_passed_not_integrated' if math_ok and real_ok and improved else 'isolated_stability_not_accepted'


def check(a):
    import copy
    import onnx
    import onnxruntime as ort
    from trt82_backend import Session
    path=a.stage/'check.json';old.fresh(path);meta=plugin_metadata(a.stage);built=n.read(a.stage/'build.json')
    if built['status']!='built_not_validated' or built['flags']!=1024 or any(built[k]!=v for k,v in meta.items()):raise ValueError('Build differs')
    report=dict(meta,status='running',rules=RULES,build_sha256=q.digest(a.stage/'build.json'),real_rows=[],synthetic_rows=[]);base=trial=None
    try:
        rows,pages,execution=n.capture(a);old_plan=n.read(PRIOR/'plan.json');old_check=n.read(PRIOR/'check.json')
        if [n.describe(r) for r in rows]!=old_plan['prior_real_rows']:raise ValueError('Original NER input drift')
        model,specs,raw=p.graph()
        if specs!=old_plan['norms']:raise ValueError('Norm graph changed')
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        names=[name for r in specs for name in (r['input'],r['output'])];del model.graph.output[:]
        model.graph.output.extend(onnx.helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,[1,'sequence',768]) for name in names)
        features=ort.InferenceSession(model.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider']);del model;gc.collect()
        nodes=copy.deepcopy(raw[0][0]);s=specs[0];rename={s['input']:'x',s['gamma_name']:'gamma',s['beta_name']:'beta',s['output']:'y'}
        for node in nodes:
            for i,name in enumerate(node.input):node.input[i]=rename.get(name,name)
            for i,name in enumerate(node.output):node.output[i]=rename.get(name,name)
        v=onnx.helper.make_tensor_value_info;tp=onnx.TensorProto.FLOAT
        micro=onnx.helper.make_model(onnx.helper.make_graph(nodes,'cpu_control',[v('x',tp,[1,'S',768]),v('gamma',tp,[768]),v('beta',tp,[768])],[v('y',tp,[1,'S',768])]),opset_imports=[onnx.helper.make_opsetid('',13)]);micro.ir_version=7
        original=ort.InferenceSession(micro.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider'])
        old_build=n.read(PRIOR/'engines.json');old_plugin=meta['prior_metadata']['plugin_sha256']
        base=Session(PRIOR/'fp32.engine',old_build['variants']['fp32']['engine_sha256'],'norm_v1',plugins=[(PRIOR/'libner_norm.so',old_plugin)])
        trial=Session(a.stage/'stable.engine',built['engine_sha256'],'norm_stable',plugins=[(a.stage/'libner_stable.so',meta['plugin_sha256'])])
        for i,row in enumerate(rows):
            vals=features.run(names,row['feeds'])
            for j,(_,gamma,beta) in enumerate(raw):
                x,ref=vals[2*j:2*j+2];r=comparison(x,gamma,beta,specs[j]['epsilon'],ref,base,trial)
                previous=old_check['real_rows'][i*25+j]
                if r['input_sha256']!=previous['input_sha256'] or r['old_output_sha256']!=previous['variants']['fp32']['output_sha256']:raise ValueError('E116 original input/output drift')
                r.update(image_sha256=row['image_sha256'],image_type=row['image_type'],norm_index=j);report['real_rows'].append(r)
            print('Stable norm real input',i+1,'/6',flush=True)
        del features,vals,rows;gc.collect();gamma,beta=raw[0][1:];plan=n.read(a.stage/'plan.json')
        for i,(length,kind,x) in enumerate(synthetic()):
            if dict(length=length,kind=kind,input_sha256=h.sha_tensor(x))!=plan['synthetic'][i]:raise ValueError('Synthetic plan drift')
            ref=original.run(None,dict(x=x,gamma=gamma,beta=beta))[0];r=comparison(x,gamma,beta,specs[0]['epsilon'],ref,base,trial)
            r.update(length=length,kind=kind)
            if i<12:
                previous=old_check['synthetic_rows'][i]['variants']['fp32']
                if r['input_sha256']!=previous['input_sha256'] or r['old_output_sha256']!=previous['output_sha256']:raise ValueError('Old synthetic drift')
            report['synthetic_rows'].append(r)
        ins=trial.engine.create_engine_inspector();ins.execution_context=trial.context
        info=json.loads(ins.get_engine_information(trial.trt.LayerInformationFormat.JSON));q.save(a.stage/'layers.json',info)
        report.update(status=result_status(report['real_rows'],report['synthetic_rows']),inspection=p.inspect(info),original_execution=execution,
            note='Independent FP64 formula criterion plus real CPU regression; old synthetic CPU-equality failures remain reported, not overwritten. No full NER/Softmax/mask/entity/performance/release claim. All intermediate data kept in RAM.')
    except Exception as err:report.update(status='failed',error=type(err).__name__);raise
    finally:
        report['execution']={name:s.stats() for name,s in (('old',base),('candidate',trial)) if s is not None}
        for s in (base,trial):
            if s is not None:s.close()
        q.save(path,report)
    print(json.dumps(dict(status=report['status'],real=len(report['real_rows']),synthetic=len(report['synthetic_rows']),
        real_cpu_passed=sum(r['candidate_vs_original']['allclose'] for r in report['real_rows']),synthetic_math_passed=sum(r['candidate_vs_fp64']['allclose'] for r in report['synthetic_rows']))),flush=True)


if __name__=='__main__':
    a=h.arguments(('prepare','build','check'))
    if a.stage!=STAGE:raise ValueError('Unexpected stage')
    {'prepare':prepare,'build':build,'check':check}[a.mode](a)
