"""E116 isolated FP32 LayerNorm boundary; no whole-NER replacement or benchmark."""
import copy
import ctypes
import gc
import json
from pathlib import Path
import re
import time

import numpy as np
import ner_mixed_trial as n

h,q,old=n.h,n.q,n.old
STAGE=q.BASE/'staging/ner_norm_e116_r1'
PRIOR=q.BASE/'staging/ner_mixed_e115_r1'
PRIOR_SHA='81d42901ddd23ed89b08148ba5feb57ffdf790b4256e2a169d149c9800a78e80'
AUDIT_SHA='626a87666ac49eeab9069470ce788324fa41c351bab26fda1fe95c099d65a61d'
TYPE='AutoOCRNerLayerNormFP32'
OPS=('ReduceMean','Sub','Constant','Pow','ReduceMean','Constant','Add','Sqrt','Div','Mul','Add')
RULES=dict(all_25_norms=True,original_epsilon=True,fp32_plugin_only=True,real_inputs=6,rtol=1e-3,atol=1e-4,
    original_precision_unchanged=True,synthetic_diagnostic_separate=True,benchmark_calls=0,accepted_for_release=False)


def provenance():
    meta=n.provenance()
    if q.digest(PRIOR/'technical_evidence.tar.gz')!=PRIOR_SHA or q.digest(PRIOR/'final_audit.json')!=AUDIT_SHA:raise ValueError('E115 changed')
    return dict(prior_metadata=meta,prior_archive_sha256=PRIOR_SHA,prior_audit_sha256=AUDIT_SHA,
        script_sha256=q.digest(__file__),plugin_source_sha256=q.digest(STAGE/'ner_layernorm_plugin.cu'),
        compiler_script_sha256=q.digest(STAGE/'build_grid_plugin.py'))


def validate_spec(r):
    if tuple(v['op'] for v in r['nodes'])!=OPS:raise ValueError('Unexpected LayerNorm pattern')
    nodes=r['nodes'];x=nodes[0]['inputs'][0];o=[v['outputs'][0] for v in nodes]
    expected=[[x],[x,o[0]],[],[o[1],o[2]],[o[3]],[],[o[4],o[5]],[o[6]],[o[1],o[7]],[o[8],r['gamma_name']],[o[9],r['beta_name']]]
    if any(v['inputs']!=want or len(v['outputs'])!=1 for v,want in zip(nodes,expected)):raise ValueError('LayerNorm wiring differs')
    if len(set(o))!=11 or any(v['axes']!=[-1] or v['keepdims']!=1 for v in (nodes[0],nodes[4])):raise ValueError('LayerNorm reduction differs')
    if r['power']!=2. or not np.isfinite(r['epsilon']) or not 0<r['epsilon']<=1:raise ValueError('Invalid constants')
    if r['gamma_shape']!=[768] or r['beta_shape']!=[768]:raise ValueError('Wrong learned parameter shape')
    return x,o[-1]


def graph():
    import onnx
    from onnx import numpy_helper as nh
    if q.digest(n.SOURCE)!=n.SOURCE_SHA:raise ValueError('Original ONNX changed')
    model=onnx.load(str(n.SOURCE));onnx.checker.check_model(model)
    weights={v.name:nh.to_array(v) for v in model.graph.initializer}
    prefixes=[v.name.rsplit('/',1)[0] for v in model.graph.node if v.name.endswith('/LayerNorm/ReduceMean')]
    if len(prefixes)!=25:raise ValueError('Missing norms')
    specs=[];raw=[]
    for prefix in prefixes:
        nodes=[v for v in model.graph.node if v.name.startswith(prefix+'/')]
        if len(nodes)!=11:raise ValueError('Nonlocal norm nodes')
        def scalar(node):
            values=[a.t for a in node.attribute if a.name=='value']
            if len(values)!=1:raise ValueError('Constant differs')
            v=nh.to_array(values[0])
            if v.dtype!=np.float32 or v.shape!=():raise ValueError('Constant not scalar FP32')
            return float(v)
        desc=[]
        for node in nodes:
            r=dict(name=node.name,op=node.op_type,inputs=list(node.input),outputs=list(node.output))
            if node.op_type=='ReduceMean':
                attrs={a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
                r.update(axes=attrs.get('axes'),keepdims=attrs.get('keepdims',1))
            desc.append(r)
        gamma,beta=(weights[nodes[i].input[1]] for i in (9,10))
        if any(v.dtype!=np.float32 or not np.isfinite(v).all() for v in (gamma,beta)):raise ValueError('Parameter dtype/range differs')
        spec=dict(name=prefix,nodes=desc,gamma_name=nodes[9].input[1],beta_name=nodes[10].input[1],
            gamma_shape=list(gamma.shape),beta_shape=list(beta.shape),gamma_sha256=h.sha_tensor(gamma),beta_sha256=h.sha_tensor(beta),
            power=scalar(nodes[2]),epsilon=scalar(nodes[5]))
        spec['input'],spec['output']=validate_spec(spec)
        specs.append(spec);raw.append((nodes,gamma,beta))
    if len({s['epsilon'] for s in specs})!=1:raise ValueError('Distinct epsilon requires separate engines')
    return model,specs,raw


def prepare(a):
    path=a.stage/'plan.json';old.fresh(path)
    report=dict(provenance(),status='running',rules=RULES)
    try:
        model,specs,raw=graph();del model,raw;gc.collect()
        report.update(status='prepared',norms=specs,epsilon=specs[0]['epsilon'],profile=[[1,2,768],[1,128,768],[1,512,768]],
            prior_real_rows=n.read(PRIOR/'plan.json')['rows'],
            note='Only isolated normalization. CPU original-graph features are not original TRT internal activations. No complete-NER, entity, speedup or release claim.')
    except Exception as err:report.update(status='failed',error=type(err).__name__);raise
    finally:q.save(path,report)
    print('Prepared',len(specs),'original norms',flush=True)


def metadata(stage):
    plan=n.read(stage/'plan.json');meta=provenance()
    if plan['status']!='prepared' or plan['rules']!=RULES or any(plan[k]!=v for k,v in meta.items()):raise ValueError('Plan drift')
    return dict(meta,plan_sha256=q.digest(stage/'plan.json'))


def plugin_metadata(stage):
    meta=metadata(stage);build=n.read(stage/'libner_norm.build.json')
    if build['source_sha256']!=meta['plugin_source_sha256'] or build['plugin_sha256']!=q.digest(stage/'libner_norm.so'):raise ValueError('Plugin changed')
    command=build['command']
    if '--fmad=false' not in command or '--use_fast_math' in command or 'arch=compute_72,code=sm_72' not in command:raise ValueError('Compiler options differ')
    return dict(meta,plugin_sha256=build['plugin_sha256'],plugin_build_sha256=q.digest(stage/'libner_norm.build.json'))


def build(a):
    import tensorrt as trt
    path=a.stage/'engines.json';old.fresh(path)
    result=dict(plugin_metadata(a.stage),status='running',variants={})
    try:
        if trt.__version__!=h.TRT_VERSION:raise ValueError('TensorRT changed')
        handle=ctypes.CDLL(str(a.stage/'libner_norm.so'),mode=ctypes.RTLD_GLOBAL)
        logger=trt.Logger(trt.Logger.WARNING);trt.init_libnvinfer_plugins(logger,'')
        creator=trt.get_plugin_registry().get_plugin_creator(TYPE,'1','')
        if creator is None:raise ValueError('Plugin not registered')
        eps=np.array([n.read(a.stage/'plan.json')['epsilon']],np.float32)
        for variant in ('fp32','half_boundary'):
            target=a.stage/(variant+'.engine');old.fresh(target);start=time.perf_counter()
            builder=trt.Builder(logger);net=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            x=net.add_input('x',trt.float32 if variant=='fp32' else trt.float16,(1,-1,768))
            gamma=net.add_input('gamma',trt.float32,(768,));beta=net.add_input('beta',trt.float32,(768,))
            cfg=builder.create_builder_config();cfg.max_workspace_size=64*1024**2;cfg.avg_timing_iterations=1
            cfg.clear_flag(trt.BuilderFlag.TF32);cfg.clear_flag(trt.BuilderFlag.INT8);cfg.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
            cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            if variant=='half_boundary':
                cfg.set_flag(trt.BuilderFlag.FP16)
                identity=net.add_identity(x);identity.name='explicit_half_to_float';identity.set_output_type(0,trt.float32);inp=identity.get_output(0)
            else:inp=x
            plugin=creator.create_plugin('isolated_norm',trt.PluginFieldCollection([trt.PluginField('epsilon',eps,trt.PluginFieldType.FLOAT32)]))
            if plugin is None:raise ValueError('Plugin creation failed')
            serialized=bytes(plugin.serialize());restored=creator.deserialize_plugin('roundtrip',serialized)
            if restored is None or bytes(restored.serialize())!=serialized:raise ValueError('Serialization differs')
            layer=net.add_plugin_v2([inp,gamma,beta],plugin);layer.name='isolated_norm_fp32';layer.set_output_type(0,trt.float32)
            out=layer.get_output(0);out.name='y';out.dtype=trt.float32;net.mark_output(out)
            profile=builder.create_optimization_profile()
            for t in (x,gamma,beta):
                shape=[[1,2,768],[1,128,768],[1,512,768]] if t.name=='x' else [[768]]*3
                profile.set_shape(t.name,*shape);t.allowed_formats=1<<int(trt.TensorFormat.LINEAR)
            cfg.add_optimization_profile(profile);out.allowed_formats=1<<int(trt.TensorFormat.LINEAR)
            blob=builder.build_serialized_network(net,cfg)
            if blob is None:raise RuntimeError('No engine')
            with target.open('xb') as f:f.write(bytes(blob))
            result['variants'][variant]=dict(engine_sha256=q.digest(target),engine_bytes=target.stat().st_size,build_s=time.perf_counter()-start,
                flags=int(cfg.flags),serialization_hex=serialized.hex(),plugin_serialization_roundtrip=True)
        result['status']='built_not_validated'
    except Exception as err:result.update(status='failed',error=type(err).__name__);raise
    finally:q.save(path,result)
    print('Built both isolated precision paths',flush=True)


def inspect(info):
    layers=info.get('Layers',[]);plugins=[r for r in layers if r.get('Name')=='isolated_norm_fp32']
    if len(plugins)!=1 or 'plugin' not in plugins[0].get('LayerType','').lower():raise ValueError('Plugin invisible')
    p=plugins[0];io=p.get('Inputs',[])+p.get('Outputs',[])
    if len(io)!=4 or any(not re.search(r'\b(float|fp32)\b',r.get('Format/Datatype','').lower()) for r in io):raise ValueError('Plugin not FP32')
    if any('myelin' in r.get('LayerType','').lower() or 'ForeignNode' in r.get('Name','') for r in layers):raise ValueError('Opaque arithmetic')
    return dict(layers=len(layers),plugin_inputs=3,plugin_outputs=1,fp32_plugin_boundary=True)


def metrics(x,y):
    if x.shape!=y.shape or x.dtype!=np.float32 or y.dtype!=np.float32 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid outputs')
    d=np.abs(x.astype(np.float64)-y.astype(np.float64))
    return dict(all_finite=True,mae=float(d.mean()),max_abs_error=float(d.max()),allclose=bool(np.allclose(x,y,rtol=RULES['rtol'],atol=RULES['atol'])))


def synthetic():
    rng=np.random.default_rng(116)
    for length in (2,128,512):
        for kind in ('normal','constant','near_constant','alternating_large'):
            if kind=='normal':x=rng.normal(0,3,(1,length,768)).astype(np.float32)
            elif kind=='constant':x=np.full((1,length,768),.125,np.float32)
            elif kind=='near_constant':x=(.125+rng.normal(0,1e-6,(1,length,768))).astype(np.float32)
            else:x=np.tile(np.array([1000.,-1000.],np.float32),length*384).reshape(1,length,768)
            yield length,kind,x


def check(a):
    import onnx
    import onnxruntime as ort
    from trt82_backend import Session
    path=a.stage/'check.json';old.fresh(path);built=n.read(a.stage/'engines.json');meta=plugin_metadata(a.stage)
    if built['status']!='built_not_validated' or any(built[k]!=v for k,v in meta.items()):raise ValueError('Build changed')
    result=dict(meta,status='running',rules=RULES,engine_report_sha256=q.digest(a.stage/'engines.json'),real_rows=[],synthetic_rows=[]);sessions={}
    try:
        plan=n.read(a.stage/'plan.json');rows,pages,execution=n.capture(a)
        if [n.describe(r) for r in rows]!=plan['prior_real_rows']:raise ValueError('Real NER input drift')
        model,specs,raw=graph()
        if specs!=plan['norms']:raise ValueError('Graph specs differ')
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        # Same original operation graph in RAM, with all normalization input/output probes.
        names=[name for r in specs for name in (r['input'],r['output'])]
        del model.graph.output[:]
        model.graph.output.extend(onnx.helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,[1,'sequence',768]) for name in names)
        feature_session=ort.InferenceSession(model.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider'])
        nodes=copy.deepcopy(raw[0][0]);rename={specs[0]['input']:'x',specs[0]['gamma_name']:'gamma',specs[0]['beta_name']:'beta',specs[0]['output']:'y'}
        for node in nodes:
            for i,name in enumerate(node.input):node.input[i]=rename.get(name,name)
            for i,name in enumerate(node.output):node.output[i]=rename.get(name,name)
        v=onnx.helper.make_tensor_value_info;tp=onnx.TensorProto.FLOAT
        micro=onnx.helper.make_model(onnx.helper.make_graph(nodes,'original_norm',[v('x',tp,[1,'S',768]),v('gamma',tp,[768]),v('beta',tp,[768])],[v('y',tp,[1,'S',768])]),opset_imports=[onnx.helper.make_opsetid('',13)]);micro.ir_version=7
        onnx.checker.check_model(micro)
        reference=ort.InferenceSession(micro.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider'])
        del model;gc.collect()
        for variant,b in built['variants'].items():
            sessions[variant]=Session(a.stage/(variant+'.engine'),b['engine_sha256'],'norm_'+variant,plugins=[(a.stage/'libner_norm.so',meta['plugin_sha256'])])
        for i,row in enumerate(rows):
            values=feature_session.run(names,row['feeds'])
            for j,(spec,(_,gamma,beta)) in enumerate(zip(specs,raw)):
                x,original=values[2*j:2*j+2]
                if not np.isfinite(x).all() or np.abs(x).max()>65504:raise ValueError('Unsafe half-boundary input')
                record=dict(image_sha256=row['image_sha256'],image_type=row['image_type'],norm_index=j,input_shape=list(x.shape),input_sha256=h.sha_tensor(x),input_abs_max=float(np.abs(x).max()),variants={})
                for variant,session in sessions.items():
                    actual=x if variant=='fp32' else x.astype(np.float16)
                    feeds=dict(x=actual,gamma=gamma,beta=beta);y=session.run(None,feeds)[0];z=session.run(None,feeds)[0]
                    if not np.array_equal(y,z):raise ValueError('Plugin repeat differs')
                    ref=reference.run(None,dict(feeds,x=actual.astype(np.float32)))[0]
                    record['variants'][variant]=dict(reference=metrics(ref,y),original_output=metrics(original,y),
                        repeat_exact=True,input_sha256=h.sha_tensor(actual),output_sha256=h.sha_tensor(y),reference_sha256=h.sha_tensor(ref))
                result['real_rows'].append(record)
            print('Compared original NER input',i+1,'25 LayerNorms',flush=True)
        del feature_session,values,rows;gc.collect()
        gamma,beta=raw[0][1:]
        for length,kind,x in synthetic():
            record=dict(length=length,kind=kind,variants={})
            for variant,session in sessions.items():
                actual=x if variant=='fp32' else x.astype(np.float16)
                feeds=dict(x=actual,gamma=gamma,beta=beta);y=session.run(None,feeds)[0];z=session.run(None,feeds)[0]
                ref=reference.run(None,dict(feeds,x=actual.astype(np.float32)))[0]
                if not np.array_equal(y,z):raise ValueError('Synthetic repeat differs')
                record['variants'][variant]=dict(reference=metrics(ref,y),repeat_exact=True,input_sha256=h.sha_tensor(actual),output_sha256=h.sha_tensor(y))
            result['synthetic_rows'].append(record)
        result['inspection']={}
        for variant,session in sessions.items():
            inspector=session.engine.create_engine_inspector();inspector.execution_context=session.context
            info=json.loads(inspector.get_engine_information(session.trt.LayerInformationFormat.JSON));q.save(a.stage/('layers_'+variant+'.json'),info)
            result['inspection'][variant]=inspect(info)
        real_ok=all(r['variants'][v]['reference']['allclose'] for r in result['real_rows'] for v in sessions)
        control_ok=all(r['variants']['fp32']['original_output']['allclose'] for r in result['real_rows'])
        synth_ok=all(r['variants'][v]['reference']['allclose'] for r in result['synthetic_rows'] for v in sessions)
        result.update(status='isolated_passed_not_integrated' if real_ok and control_ok and synth_ok else 'isolated_numeric_not_accepted',
            real_same_input_passed=real_ok,original_fp32_control_passed=control_ok,synthetic_same_input_passed=synth_ok,original_execution=execution,
            note='CPU ONNX intermediate inputs; not original TRT hidden activations. Half-boundary comparison uses identically rounded CPU input; original-vs-half difference is separately diagnostic. No complete NER, Softmax/mask protection, entities, timing or release.')
    except Exception as err:result.update(status='failed',error=type(err).__name__);raise
    finally:
        result['execution']={k:s.stats() for k,s in sessions.items()}
        for s in sessions.values():s.close()
        q.save(path,result)
    print(json.dumps({k:result[k] for k in ('status','real_same_input_passed','original_fp32_control_passed','synthetic_same_input_passed')}),flush=True)


if __name__=='__main__':
    a=h.arguments(('prepare','build','check'))
    if a.stage!=STAGE:raise ValueError('Unexpected stage')
    {'prepare':prepare,'build':build,'check':check}[a.mode](a)
