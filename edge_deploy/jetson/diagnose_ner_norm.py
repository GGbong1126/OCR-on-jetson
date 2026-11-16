"""E116 fixed synthetic low-variance diagnosis; preserves the failed gate."""
import copy
import gc
import json
import numpy as np
import ner_norm_probe as p

n,h,q,old=p.n,p.h,p.q,p.old


def tree_sum(x):
    s=np.zeros(x.shape[:-1]+(256,),np.float32)
    for j in range(3):s=s+x[...,j*256:(j+1)*256]
    for size in (128,64,32,16,8,4,2,1):s=s[...,:size]+s[...,size:2*size]
    return s


def emulate(x,gamma,beta,epsilon):
    mean=tree_sum(x)/np.float32(768);d=x-mean
    var=tree_sum(d*d)/np.float32(768)
    return ((d/np.sqrt(var+np.float32(epsilon)))*gamma)+beta


def precise(x,gamma,beta,epsilon):
    v=x.astype(np.float64);d=v-v.mean(axis=-1,keepdims=True)
    return d/np.sqrt((d*d).mean(axis=-1,keepdims=True)+epsilon)*gamma.astype(np.float64)+beta.astype(np.float64)


def errors(reference,value):
    if reference.shape!=value.shape or not np.isfinite(reference).all() or not np.isfinite(value).all():raise ValueError('Invalid diagnostic output')
    d=np.abs(reference-value.astype(np.float64))
    return dict(mae=float(d.mean()),max_abs_error=float(d.max()))


def run(a):
    import onnx
    import onnxruntime as ort
    from trt82_backend import Session
    path=a.stage/'diagnostic.json';old.fresh(path)
    meta=p.plugin_metadata(a.stage);checked=n.read(a.stage/'check.json');built=n.read(a.stage/'engines.json')
    if any(checked[k]!=v for k,v in meta.items()) or checked['status']!='isolated_numeric_not_accepted':raise ValueError('Expected preserved failed check')
    result=dict(meta,diagnostic_sha256=q.digest(__file__),check_sha256=q.digest(a.stage/'check.json'),status='running',rows=[]);sessions={}
    try:
        model,specs,raw=p.graph();del model;gc.collect()
        nodes=copy.deepcopy(raw[0][0]);gamma,beta=raw[0][1:];eps=specs[0]['epsilon']
        rename={specs[0]['input']:'x',specs[0]['gamma_name']:'gamma',specs[0]['beta_name']:'beta',specs[0]['output']:'y'}
        for node in nodes:
            for i,name in enumerate(node.input):node.input[i]=rename.get(name,name)
            for i,name in enumerate(node.output):node.output[i]=rename.get(name,name)
        v=onnx.helper.make_tensor_value_info;tp=onnx.TensorProto.FLOAT
        m=onnx.helper.make_model(onnx.helper.make_graph(nodes,'norm_diagnostic',[v('x',tp,[1,'S',768]),v('gamma',tp,[768]),v('beta',tp,[768])],[v('y',tp,[1,'S',768])]),opset_imports=[onnx.helper.make_opsetid('',13)]);m.ir_version=7
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        ort_session=ort.InferenceSession(m.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider'])
        for name,b in built['variants'].items():sessions[name]=Session(a.stage/(name+'.engine'),b['engine_sha256'],'norm_diag_'+name,plugins=[(a.stage/'libner_norm.so',meta['plugin_sha256'])])
        for length,kind,x in p.synthetic():
            if kind!='near_constant':continue
            record=dict(length=length,kind=kind,variants={})
            prior=next(r for r in checked['synthetic_rows'] if r['length']==length and r['kind']==kind)
            for name,s in sessions.items():
                actual=x if name=='fp32' else x.astype(np.float16);f=actual.astype(np.float32)
                if h.sha_tensor(actual)!=prior['variants'][name]['input_sha256']:raise ValueError('Synthetic input drift')
                feeds=dict(x=actual,gamma=gamma,beta=beta);y=s.run(None,feeds)[0];ref=ort_session.run(None,dict(feeds,x=f))[0]
                if h.sha_tensor(y)!=prior['variants'][name]['output_sha256']:raise ValueError('Diagnostic GPU drift')
                oracle=precise(f,gamma,beta,eps);emulated=emulate(f,gamma,beta,eps)
                mean64=f.astype(np.float64).mean(-1,keepdims=True)
                var64=((f.astype(np.float64)-mean64)**2).mean(-1)
                record['variants'][name]=dict(input_sha256=h.sha_tensor(actual),output_sha256=h.sha_tensor(y),
                    cpu_fp32_vs_fp64=errors(oracle,ref),gpu_fp32_vs_fp64=errors(oracle,y),cpu_tree_vs_gpu=p.metrics(emulated,y),
                    cpu_tree_gpu_exact=bool(np.array_equal(emulated,y)),variance64_min=float(var64.min()),variance64_max=float(var64.max()),
                    tree_mean_max_abs_error=float(np.max(np.abs(tree_sum(f)/np.float32(768)-mean64))),
                    constant_rows=int(np.count_nonzero(var64==0)),rows=int(f.shape[1]))
            result['rows'].append(record)
        result.update(status='diagnosed_not_acceptance',
            note='Synthetic-only high precision reference and exact FP32 reduction-tree emulation. No changes to plugin, inputs, tolerance or failed original report. Identically rounded half inputs can lose all low-variance detail; a passing normalization comparison is not evidence that this lost signal is harmless in full NER.')
    except Exception as err:result.update(status='failed',error=type(err).__name__);raise
    finally:
        result['execution']={k:s.stats() for k,s in sessions.items()}
        for s in sessions.values():s.close()
        q.save(path,result)
    print(json.dumps(dict(status=result['status'],rows=result['rows'])),flush=True)


if __name__=='__main__':
    a=h.arguments(('check',))
    if a.stage!=p.STAGE:raise ValueError('Unexpected stage')
    run(a)
