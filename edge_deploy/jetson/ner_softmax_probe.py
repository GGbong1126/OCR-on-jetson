"""E118 visible FP32 scale/mask/Softmax boundary, no full-NER replacement."""
import ctypes
import gc
import json
import re
import time
import numpy as np
import ner_mixed_trial as n

h,q,old=n.h,n.q,n.old
STAGE=q.BASE/'staging/ner_softmax_e118_r1'
PRIOR=q.BASE/'staging/ner_norm_stable_e117_r1'
PRIOR_SHA='758bf30acd010f34f277c2e9959b20a6c29284937b60c7db060e1da0f1ba9849'
AUDIT_SHA='2c6f19f763d16b2f210345cf6ccd3c264786e326a182e3fce89323539ed8c09a'
TYPE='AutoOCRNerMaskSoftmaxFP32'
RULES=dict(real_cases=72,synthetic_cases=32,rtol=1e-3,atol=1e-4,row_sum_atol=2e-6,
    fp32_only=True,preserve_finite_mask=True,all_masked_uniform=True,benchmark_calls=0,accepted_for_release=False)


def inputs(x,mask):
    if x.dtype!=np.float32 or x.ndim!=4 or x.shape[:2]!=(1,12) or x.shape[2]!=x.shape[3] or not 2<=x.shape[3]<=512:raise ValueError('Score contract')
    if mask.dtype!=np.float32 or mask.shape!=(1,x.shape[3]) or not np.isin(mask,[0.,1.]).all():raise ValueError('Binary mask contract')
    if not np.isfinite(x).all() or np.abs(x).max()>1e20:raise ValueError('Score range outside tested contract')
    return dict(scores=x,mask=mask)


def logits(x,mask):
    inputs(x,mask)
    return x/np.float32(8)+(np.float32(1)-mask[:,None,None,:])*np.float32(-np.finfo(np.float32).max)


def oracle(x,mask):
    # Preserve the original FP32 mask/add rounding; only Softmax uses FP64.
    z=logits(x,mask).astype(np.float64);z-=z.max(axis=-1,keepdims=True)
    y=np.exp(z);return y/y.sum(axis=-1,keepdims=True)


def synthetic():
    rng=np.random.default_rng(118)
    for length in (2,127,128,512):
        for kind in ('normal','prefix_padding','alternating_mask','one_active','all_masked','constant','large_alternating','large_offset'):
            x=rng.normal(0,8,(1,12,length,length)).astype(np.float32);mask=np.ones((1,length),np.float32)
            if kind=='prefix_padding':mask[:,max(1,length//2):]=0
            elif kind=='alternating_mask':mask[:,1::2]=0
            elif kind=='one_active':mask[:]=0;mask[:,length//2]=1
            elif kind=='all_masked':mask[:]=0
            elif kind=='constant':x[:]=.125
            elif kind=='large_alternating':x[:]=60000.;x[:,:,:,1::2]=-60000.
            elif kind=='large_offset':x+=np.float32(60000.)
            yield length,kind,x,mask


def describe_input(x,mask):
    return dict(input_shape=list(x.shape),input_sha256=h.sha_tensor(x),mask_sha256=h.sha_tensor(mask),
        score_abs_max=float(np.abs(x).max()),active_keys=int(mask.sum()))


def graph():
    import onnx
    from onnx import numpy_helper as nh
    if q.digest(n.SOURCE)!=n.SOURCE_SHA:raise ValueError('Original ONNX changed')
    model=onnx.load(str(n.SOURCE));onnx.checker.check_model(model)
    producers={v:node for node in model.graph.node for v in node.output}
    def node(name,op):
        out=producers[name]
        if out.op_type!=op or out.domain or len(out.output)!=1:raise ValueError('Unexpected graph operation')
        return out
    def constant(name,value,dtype):
        c=node(name,'Constant');v=nh.to_array(next(a.t for a in c.attribute if a.name=='value'))
        if v.dtype!=dtype or not np.array_equal(v,np.asarray(value,dtype=dtype)):raise ValueError('Constant differs')
    soft=[v for v in model.graph.node if v.op_type=='Softmax'];specs=[]
    if len(soft)!=12:raise ValueError('Softmax coverage differs')
    for i,s in enumerate(soft):
        prefix='/inner/bert/encoder/layer.%d/attention/self/'%i
        if s.name!=prefix+'Softmax' or {a.name:onnx.helper.get_attribute_value(a) for a in s.attribute}!={'axis':-1}:raise ValueError('Softmax axis/name differs')
        add=node(s.input[0],'Add');div=node(add.input[0],'Div');mul=node(add.input[1],'Mul')
        if add.name!=prefix+'Add' or div.name!=prefix+'Div' or node(div.input[0],'MatMul').name!=prefix+'MatMul':raise ValueError('Attention wiring differs')
        constant(div.input[1],8.,np.float32);constant(mul.input[1],-np.finfo(np.float32).max,np.float32)
        sub=node(mul.input[0],'Sub');constant(sub.input[0],1.,np.float32)
        cast=node(sub.input[1],'Cast')
        if {a.name:onnx.helper.get_attribute_value(a) for a in cast.attribute}!={'to':onnx.TensorProto.FLOAT}:raise ValueError('Mask cast differs')
        u2=node(cast.input[0],'Unsqueeze');u1=node(u2.input[0],'Unsqueeze')
        constant(u2.input[1],[2],np.int64);constant(u1.input[1],[1],np.int64)
        if list(u1.input[:1])!=['attention_mask']:raise ValueError('Mask source differs')
        specs.append(dict(name=s.name,input=div.input[0],output=s.output[0],mask='attention_mask',divisor=8.,mask_value=float(-np.finfo(np.float32).max)))
    return model,specs


def provenance():
    if q.digest(PRIOR/'technical_evidence.tar.gz')!=PRIOR_SHA or q.digest(PRIOR/'final_audit.json')!=AUDIT_SHA:raise ValueError('E117 changed')
    return dict(original=n.provenance(),previous_archive_sha256=PRIOR_SHA,previous_audit_sha256=AUDIT_SHA,
        script_sha256=q.digest(__file__),plugin_source_sha256=q.digest(STAGE/'ner_mask_softmax.cu'),compiler_sha256=q.digest(STAGE/'build_grid_plugin.py'))


def prepare(a):
    old.fresh(a.stage/'plan.json');model,specs=graph();del model;gc.collect()
    cases=[dict(length=l,kind=k,**describe_input(x,m)) for l,k,x,m in synthetic()]
    q.save(a.stage/'plan.json',dict(provenance(),status='prepared',rules=RULES,specs=specs,synthetic=cases,
        prior_real_rows=n.read(q.BASE/'staging/ner_norm_e116_r1/plan.json')['prior_real_rows'],
        note='Freeze original FP32 finite-mask semantics; FP64 Softmax reference uses already-rounded FP32 logits. No -inf replacement, no full-NER/GT/benchmark/promotion.'))
    print('Frozen 12 attention blocks and 32 synthetic cases',flush=True)


def metadata(stage,plugin=False):
    meta=provenance();plan=n.read(stage/'plan.json')
    if plan['status']!='prepared' or plan['rules']!=RULES or any(plan[k]!=v for k,v in meta.items()):raise ValueError('Plan drift')
    meta['plan_sha256']=q.digest(stage/'plan.json')
    if plugin:
        b=n.read(stage/'libner_softmax.build.json')
        if b['source_sha256']!=meta['plugin_source_sha256'] or b['plugin_sha256']!=q.digest(stage/'libner_softmax.so'):raise ValueError('Plugin drift')
        if '--fmad=false' not in b['command'] or '--use_fast_math' in b['command'] or 'arch=compute_72,code=sm_72' not in b['command']:raise ValueError('Compiler flags differ')
        meta.update(plugin_sha256=b['plugin_sha256'],plugin_build_sha256=q.digest(stage/'libner_softmax.build.json'))
    return meta


def build(a):
    import tensorrt as trt
    path=a.stage/'build.json';engine=a.stage/'softmax.engine';old.fresh(path);old.fresh(engine)
    report=dict(metadata(a.stage,True),status='running');start=time.perf_counter()
    try:
        if trt.__version__!=h.TRT_VERSION:raise ValueError('TRT changed')
        handle=ctypes.CDLL(str(a.stage/'libner_softmax.so'),mode=ctypes.RTLD_GLOBAL)
        logger=trt.Logger(trt.Logger.WARNING);trt.init_libnvinfer_plugins(logger,'')
        creator=trt.get_plugin_registry().get_plugin_creator(TYPE,'1','')
        if creator is None:raise ValueError('Creator missing')
        plugin=creator.create_plugin('fp32_mask_softmax',trt.PluginFieldCollection([]))
        if plugin is None:raise ValueError('Plugin missing')
        data=bytes(plugin.serialize());again=creator.deserialize_plugin('roundtrip',data)
        if again is None or bytes(again.serialize())!=data:raise ValueError('Serialization differs')
        builder=trt.Builder(logger);net=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH));cfg=builder.create_builder_config()
        cfg.max_workspace_size=64*1024**2;cfg.avg_timing_iterations=1;cfg.clear_flag(trt.BuilderFlag.TF32)
        cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS);cfg.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        x=net.add_input('scores',trt.float32,(1,12,-1,-1));m=net.add_input('mask',trt.float32,(1,-1))
        layer=net.add_plugin_v2([x,m],plugin);layer.name='fp32_mask_softmax';layer.set_output_type(0,trt.float32)
        y=layer.get_output(0);y.name='y';net.mark_output(y);y.dtype=trt.float32
        profile=builder.create_optimization_profile()
        for t in (x,m):
            shapes=[(1,12,s,s) if t.name=='scores' else (1,s) for s in (2,128,512)]
            profile.set_shape(t.name,*shapes)
            if [tuple(s) for s in profile.get_shape(t.name)]!=shapes:raise ValueError('Profile not applied')
        for t in (x,m,y):t.allowed_formats=1<<int(trt.TensorFormat.LINEAR)
        cfg.add_optimization_profile(profile);blob=builder.build_serialized_network(net,cfg)
        if blob is None:raise RuntimeError('No engine')
        with engine.open('xb') as f:f.write(bytes(blob))
        report.update(status='built_not_validated',engine_sha256=q.digest(engine),engine_bytes=engine.stat().st_size,flags=int(cfg.flags),serialization_roundtrip=True)
    except Exception as e:report.update(status='failed',error=type(e).__name__);raise
    finally:report['build_s']=time.perf_counter()-start;q.save(path,report)
    print('FP32 mask/Softmax engine built',flush=True)


def inspect(info):
    rows=info.get('Layers',[])
    if len(rows)!=1 or rows[0].get('Name')!='fp32_mask_softmax' or 'plugin' not in rows[0].get('LayerType','').lower():raise ValueError('Opaque/extra arithmetic')
    io=rows[0].get('Inputs',[])+rows[0].get('Outputs',[])
    if len(io)!=3 or any(not re.search(r'\b(float|fp32)\b',v.get('Format/Datatype','').lower()) for v in io):raise ValueError('Not FP32')
    return dict(layers=1,fp32_inputs=2,fp32_outputs=1,visible_plugin=True)


def metrics(ref,y):
    if ref.shape!=y.shape or y.dtype!=np.float32 or not np.isfinite(ref).all() or not np.isfinite(y).all():raise ValueError('Invalid output')
    d=np.abs(ref.astype(np.float64)-y.astype(np.float64))
    return dict(all_finite=True,allclose=bool(np.allclose(y,ref,rtol=RULES['rtol'],atol=RULES['atol'])),mae=float(d.mean()),max_abs_error=float(d.max()))


def comparison(x,mask,ref,session):
    feeds=inputs(x,mask);y=session.run(None,feeds)[0];z=session.run(None,feeds)[0]
    if not np.array_equal(y,z):raise ValueError('Repeat differs')
    total=float(np.max(np.abs(y.astype(np.float64).sum(axis=-1)-1)))
    masked=y[...,mask[0]==0];active=int(mask.sum())
    masked_max=float(np.max(masked)) if masked.size and active else None
    uniform=bool(np.array_equal(y,np.full_like(y,1./x.shape[-1]))) if active==0 else None
    valid=bool(y.min()>=0 and y.max()<=1 and total<=RULES['row_sum_atol'] and (masked_max is None or masked_max==0) and (uniform is None or uniform))
    return dict(describe_input(x,mask),output_sha256=h.sha_tensor(y),reference_sha256=h.sha_tensor(ref),repeat_exact=True,
        cpu=metrics(ref,y),fp64_softmax=metrics(oracle(x,mask),y),row_sum_error=total,masked_max=masked_max,
        all_masked_uniform=uniform,probability_contract_passed=valid)


def result_status(real,syn):
    valid=len(real)==72 and len(syn)==32 and all(r['cpu']['allclose'] and r['fp64_softmax']['allclose'] and r['probability_contract_passed'] and r['repeat_exact'] for r in real+syn)
    return 'isolated_fp32_mask_softmax_passed_not_integrated' if valid else 'isolated_not_accepted'


def cpu_reference(opts):
    import onnx
    import onnxruntime as ort
    H=onnx.helper;T=onnx.TensorProto;v=H.make_tensor_value_info
    def const(name,value,dtype):return H.make_node('Constant',[],[name],value=onnx.numpy_helper.from_array(np.asarray(value,dtype=dtype)))
    nodes=[const('eight',8.,np.float32),const('one',1.,np.float32),const('low',-np.finfo(np.float32).max,np.float32),const('axes',[1,2],np.int64),
        H.make_node('Unsqueeze',['mask','axes'],['mask4']),H.make_node('Sub',['one','mask4'],['inverted']),H.make_node('Mul',['inverted','low'],['bias']),
        H.make_node('Div',['scores','eight'],['scaled']),H.make_node('Add',['scaled','bias'],['logits']),H.make_node('Softmax',['logits'],['y'],axis=-1)]
    model=H.make_model(H.make_graph(nodes,'fp32_original_formula',[v('scores',T.FLOAT,[1,12,'S','S']),v('mask',T.FLOAT,[1,'S'])],[v('y',T.FLOAT,[1,12,'S','S'])]),opset_imports=[H.make_opsetid('',13)]);model.ir_version=7
    onnx.checker.check_model(model)
    return ort.InferenceSession(model.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider'])


def check(a):
    import onnx
    import onnxruntime as ort
    from trt82_backend import Session
    path=a.stage/'check.json';old.fresh(path);meta=metadata(a.stage,True);built=n.read(a.stage/'build.json');plan=n.read(a.stage/'plan.json')
    if built['status']!='built_not_validated' or built['flags']!=1024 or any(built[k]!=v for k,v in meta.items()):raise ValueError('Build drift')
    result=dict(meta,status='running',rules=RULES,build_sha256=q.digest(a.stage/'build.json'),real_rows=[],synthetic_rows=[]);trial=None
    try:
        rows,pages,execution=n.capture(a)
        if [n.describe(r) for r in rows]!=plan['prior_real_rows']:raise ValueError('Original input drift')
        model,specs=graph()
        if specs!=plan['specs']:raise ValueError('Graph drift')
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        names=[v for r in specs for v in (r['input'],r['output'])];del model.graph.output[:]
        model.graph.output.extend(onnx.helper.make_tensor_value_info(v,onnx.TensorProto.FLOAT,[1,12,'S','S']) for v in names)
        features=ort.InferenceSession(model.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider']);del model;gc.collect()
        cpu=cpu_reference(opts)
        trial=Session(a.stage/'softmax.engine',built['engine_sha256'],'mask_softmax',plugins=[(a.stage/'libner_softmax.so',meta['plugin_sha256'])])
        for i,row in enumerate(rows):
            values=features.run(names,row['feeds']);mask=row['feeds']['attention_mask'].astype(np.float32)
            for j in range(12):
                x,ref=values[2*j:2*j+2];r=comparison(x,mask,ref,trial)
                r.update(image_sha256=row['image_sha256'],image_type=row['image_type'],layer=j);result['real_rows'].append(r)
            print('Checked real input',i+1,'/6, 12 attention blocks',flush=True)
        del features,values,rows;gc.collect()
        for i,(length,kind,x,mask) in enumerate(synthetic()):
            if dict(length=length,kind=kind,**describe_input(x,mask))!=plan['synthetic'][i]:raise ValueError('Synthetic drift')
            ref=cpu.run(None,inputs(x,mask))[0];r=comparison(x,mask,ref,trial);r.update(length=length,kind=kind);result['synthetic_rows'].append(r)
        ins=trial.engine.create_engine_inspector();ins.execution_context=trial.context
        info=json.loads(ins.get_engine_information(trial.trt.LayerInformationFormat.JSON));q.save(a.stage/'layers.json',info)
        result.update(status=result_status(result['real_rows'],result['synthetic_rows']),inspection=inspect(info),original_execution=execution,
            note='CPU original-graph intermediate scores, not original TRT hidden features. FP64 Softmax on rounded FP32 logits. Finite -FLT_MAX mask and all-masked uniform semantics preserved. No full NER/business/benchmark/promotion.')
    except Exception as e:result.update(status='failed',error=type(e).__name__);raise
    finally:
        if trial is not None:result['execution']=trial.stats();trial.close()
        q.save(path,result)
    print(json.dumps(dict(status=result['status'],real=len(result['real_rows']),synthetic=len(result['synthetic_rows']))),flush=True)


if __name__=='__main__':
    a=h.arguments(('prepare','build','check'))
    if a.stage!=STAGE:raise ValueError('Unexpected stage')
    {'prepare':prepare,'build':build,'check':check}[a.mode](a)
