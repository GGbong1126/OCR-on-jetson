"""E115: isolated BERT linear-projection half trial, no release/HTTP edits."""
from collections import Counter
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import layout_hotspot_mixed as h

q,old=h.q,h.old
SOURCE=q.BASE/'staging/ner_onnx_r1/ner.onnx'
SOURCE_SHA='63357952a8e37f5e0e35827f94a0c2da76b2e26570ef2a9c8a23c02dfde05f0b'
PRIOR=q.BASE/'staging/cell_partition_e114_r2'
PRIOR_SHA='0c73de6570bdcf1dc2364821961f5d1d626c6607b90e33c54fc53e8e2fe58720'
MODEL_DIR=q.RELEASE/'pth/ner/ner_model_cadre_docs_continuous_v3/ner_model_cadre_docs_continuous_v3'
INPUTS=('input_ids','attention_mask','token_type_ids')
PROFILE=[[1,2],[1,128],[1,512]]
TARGETS=tuple('/inner/bert/encoder/layer.%d/%s/MatMul'%(i,s) for i in range(12) for s in (
    'attention/self/query','attention/self/key','attention/self/value','attention/output/dense','intermediate/dense','output/dense'))
RULES=dict(half_projection_count=72,protect_other_float_compute=True,obey_precision_constraints=True,
    require_visible_protected_fp32=True,strict_active_token_labels=True,strict_raw_and_cleaned_entity_spans=True,
    strict_threshold_membership=True,strict_business=True,no_extra_cpu_fallback=True,no_error_fallback=True,
    numeric_allclose_diagnostic_only=True,benchmark_authorized=False,accepted_for_release=False)


def read(path):return json.loads(Path(path).read_text())


def provenance():
    cfg_path=q.RELEASE/'edge_deploy/jetson/deployment.json';cfg=read(cfg_path)
    if q.digest(cfg_path)!=h.DEPLOYMENT_SHA or q.digest(SOURCE)!=SOURCE_SHA or q.digest(PRIOR/'technical_evidence.tar.gz')!=PRIOR_SHA:
        raise ValueError('Source/release/history changed')
    prior=read(PRIOR/'final_audit.json')
    for k,spec in cfg['engines'].items():
        if q.digest(q.RELEASE/spec['path'])!=prior['engines'][k] or spec['sha256']!=prior['engines'][k]:raise ValueError('Original engine changed')
    if q.digest(q.BASE/'staging/quant_yolo_e92_r1/plan.json')!=h.PLAN_SHA:raise ValueError('Dataset plan changed')
    artifacts={str(p.relative_to(q.RELEASE)):q.digest(p) for p in MODEL_DIR.iterdir() if p.is_file()}
    if q.digest(MODEL_DIR/'model.safetensors')!=cfg['engines']['ner']['source_sha256']:raise ValueError('Original weights changed')
    for n in ('server.py','server_config.py','ner_postprocess.py','edge_deploy/jetson/jetson_models.py','edge_deploy/jetson/trt82_backend.py'):
        artifacts[n]=q.digest(q.RELEASE/n)
    return dict(source_sha256=SOURCE_SHA,deployment_sha256=h.DEPLOYMENT_SHA,prior_archive_sha256=PRIOR_SHA,
        prior_audit_sha256=q.digest(PRIOR/'final_audit.json'),engine_sha256=cfg['engines']['ner']['sha256'],
        dataset_plan_sha256=h.PLAN_SHA,artifacts_sha256=artifacts,script_sha256=q.digest(__file__),
        helpers_sha256={Path(m.__file__).name:q.digest(m.__file__) for m in (h,old,q)})


def input_meta(feeds):
    if set(feeds)!=set(INPUTS):raise ValueError('NER inputs differ')
    shape=feeds['input_ids'].shape
    if len(shape)!=2 or shape[0]!=1 or not 2<=shape[1]<=512:raise ValueError('Sequence profile violated')
    for k,v in feeds.items():
        if v.shape!=shape or v.dtype!=np.int64:raise ValueError('Original token dtype/shape changed')
        if k=='input_ids' and (v.min()<0 or v.max()>=21128):raise ValueError('Token ID outside vocabulary')
        if k!='input_ids' and not np.isin(v,[0,1]).all():raise ValueError('Mask/type invalid')
    return {k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=h.sha_tensor(v)) for k,v in sorted(feeds.items())}


def capture(a):
    images=q.load_plan(q.BASE/'staging/quant_yolo_e92_r1');rows=[];pages=[]
    models=base=original=None;active={};used=0
    try:
        with open(os.devnull,'w') as quiet:
            with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):server,pipeline,models,cfg=old.setup(a)
            if server.cfg.NER_SCORE_THRESHOLD!=.98:raise ValueError('Original threshold differs')
            base=models.models['ner'];original=base.run
            def call(names,feeds):
                nonlocal used
                im=input_meta(feeds);out=original(names,feeds)
                if len(out)!=1 or out[0].shape!=(1,feeds['input_ids'].shape[1],43) or out[0].dtype!=np.float32:raise ValueError('Logits contract changed')
                used+=sum(v.nbytes for v in feeds.values())+out[0].nbytes
                if used>32*1024**2 or len(rows)>256:raise ValueError('Capture budget exceeded')
                rows.append(dict(active,inputs=im,feeds={k:v.copy() for k,v in feeds.items()},reference=out[0].copy(),output_sha256=h.sha_tensor(out[0])))
                return out
            guard=h.GuardedCall(call);base.run=guard
            for image in images['calibration'][:3]:
                for kind in (3,1):
                    active.update(image_sha256=image['sha256'],image_type=kind);start=len(rows)
                    with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):result=pipeline.process(str(Path(images['images'])/image['file']),image_type=kind,run_ner=True)
                    guard.require_healthy()
                    if not result['success'] or not models.healthy:raise RuntimeError('Original business error')
                    pages.append(dict(active,ner_calls=len(rows)-start))
            execution=models.stats()
    finally:
        if original is not None:base.run=original
        old.close_models(models)
    if not rows or len(pages)!=6:raise ValueError('Missing real NER coverage')
    return rows,pages,execution


def describe(row):return {k:row[k] for k in ('image_sha256','image_type','inputs','output_sha256')}


def prepare(a):
    target=a.stage/'plan.json';old.fresh(target);result=dict(provenance(),status='running',rules=RULES)
    try:
        import onnx
        from onnx import numpy_helper as nh
        m=onnx.load(str(SOURCE));g=m.graph;nodes={n.name:n for n in g.node};weights={v.name:v for v in g.initializer}
        if len(g.node)!=1129 or len(nodes)!=1129 or any(n.domain for n in g.node):raise ValueError('Graph changed')
        if [v.name for v in g.input]!=list(INPUTS) or [v.name for v in g.output]!=['output']:raise ValueError('ONNX interface changed')
        onnx.checker.check_model(m)
        params=[]
        for name in TARGETS:
            n=nodes[name]
            if n.op_type!='MatMul' or len(n.input)!=2 or n.input[1] not in weights:raise ValueError('Dense target changed')
            v=nh.to_array(weights[n.input[1]])
            if v.dtype!=np.float32 or not np.isfinite(v).all() or np.abs(v).max()>32752:raise ValueError('Unsafe projection weights')
            params.append(dict(name=name,weight_shape=list(v.shape),weight_sha256=h.sha_tensor(v),weight_abs_max=float(np.abs(v).max())))
        protected=[n.name for n in g.node if '/LayerNorm/' in n.name and n.op_type not in ('Constant',) or n.op_type=='Softmax']
        result['graph']=dict(nodes=len(g.node),ops=dict(Counter(n.op_type for n in g.node)),parameters=params,protected_nodes=protected)
        del m,g,nodes,weights
        rows,pages,execution=capture(a)
        result.update(status='prepared',rows=[describe(r) for r in rows],pages=pages,original_execution=execution,
            profile=PROFILE,targets=list(TARGETS),
            note='Unchanged 12-layer BERT, tokenizer, chunk500/overlap50, aggregation and threshold0.98. Only 72 learned encoder projections request half output, explicit restoration to float. QK/PV attention, mask, LayerNorm, Softmax, GELU and classifier request float; actual fusion may prevent independent verification and must be reported. No GT, benchmark, HTTP or promotion.')
    except Exception as err:result.update(status='failed',error=type(err).__name__);raise
    finally:q.save(target,result)
    print('Prepared NER plan: calls',len(rows),'lengths',sorted({r['feeds']['input_ids'].shape[1] for r in rows}),flush=True)


def metadata(stage):
    r=read(stage/'plan.json');meta=provenance()
    if r['status']!='prepared' or r['rules']!=RULES or r['targets']!=list(TARGETS) or any(r[k]!=v for k,v in meta.items()):raise ValueError('Frozen plan changed')
    return dict(meta,plan_sha256=q.digest(stage/'plan.json'))


def rewire(net,trt):
    layers=[net.get_layer(i) for i in range(net.num_layers)];pending=[]
    for name in TARGETS:
        found=[v for v in layers if v.name==name]
        if len(found)!=1 or str(found[0].type).split('.')[-1]!='MATRIX_MULTIPLY':raise ValueError('Parsed projection changed')
        src=found[0].get_output(0)
        consumers=[(v,j) for v in layers for j in range(v.num_inputs) if v.get_input(j) is not None and v.get_input(j).name==src.name]
        if not consumers:raise ValueError('Unused projection')
        pending.append((name,src,consumers))
    rows=[]
    for name,src,users in pending:
        identity=net.add_identity(src);identity.name=name+'_restore_fp32';identity.set_output_type(0,trt.float32)
        former=src.name;src.name=former+'__half';identity.get_output(0).name=former
        for layer,i in users:layer.set_input(i,identity.get_output(0))
        rows.append(dict(name=name,output=former,consumers=[dict(name=v.name,index=i) for v,i in users]))
    return rows


def policy(name,kind,indices):
    if name in TARGETS:
        if kind!='MATRIX_MULTIPLY' or indices!=[0]:raise ValueError('Half target differs')
        return 'half_projection'
    if indices and kind not in h.COMPUTE|h.STRUCTURAL:raise ValueError('Unhandled float layer '+kind)
    return 'fp32' if indices else 'integer_shape_unchanged'


def build(a):
    import tensorrt as trt
    target=a.stage/'build.json';engine=a.stage/'ner_mixed.engine';old.fresh(target);old.fresh(engine)
    result=dict(metadata(a.stage),status='building',constraints=[]);start=None
    try:
        if trt.__version__!=h.TRT_VERSION:raise ValueError('Runtime changed')
        logger=trt.Logger(trt.Logger.INFO);trt.init_libnvinfer_plugins(logger,'')
        builder=trt.Builder(logger);net=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH));parser=trt.OnnxParser(net,logger)
        if not parser.parse(SOURCE.read_bytes()):raise RuntimeError('NER parse failed')
        if {net.get_input(i).name:list(net.get_input(i).shape) for i in range(net.num_inputs)}!={n:[1,-1] for n in INPUTS}:raise ValueError('Parsed inputs differ')
        result['boundaries']=rewire(net,trt)
        for i in range(net.num_layers):
            layer=net.get_layer(i);kind=str(layer.type).split('.')[-1]
            indices=[j for j in range(layer.num_outputs) if layer.get_output(j) is not None and layer.get_output(j).dtype==trt.float32 and not layer.get_output(j).is_shape_tensor]
            choice=policy(layer.name,kind,indices)
            if indices:
                dtype=trt.float16 if choice=='half_projection' else trt.float32
                if kind in h.COMPUTE:layer.precision=dtype
                for j in indices:layer.set_output_type(j,dtype)
            result['constraints'].append(dict(name=layer.name,kind=kind,output_indices=indices,policy=choice))
        cfg=builder.create_builder_config();cfg.max_workspace_size=512*1024**2;cfg.avg_timing_iterations=1
        cfg.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        cfg.set_flag(trt.BuilderFlag.FP16);cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        cfg.clear_flag(trt.BuilderFlag.TF32);cfg.clear_flag(trt.BuilderFlag.INT8)
        profile=builder.create_optimization_profile()
        for i in range(net.num_inputs):
            t=net.get_input(i);t.dtype=trt.int32;t.allowed_formats=1<<int(trt.TensorFormat.LINEAR)
            profile.set_shape(t.name,*PROFILE)
            if [list(v) for v in profile.get_shape(t.name)]!=PROFILE:raise ValueError('Profile differs')
        if net.num_outputs!=1 or net.get_output(0).name!='output':raise ValueError('Output differs')
        net.get_output(0).dtype=trt.float32;net.get_output(0).allowed_formats=1<<int(trt.TensorFormat.LINEAR)
        cfg.add_optimization_profile(profile);cfg.set_timing_cache(cfg.create_timing_cache(b''),False)
        result.update(profile=PROFILE,trt_version=trt.__version__,flags=int(cfg.flags),workspace_bytes=cfg.max_workspace_size,avg_timing_iterations=cfg.avg_timing_iterations)
        start=time.perf_counter();blob=builder.build_serialized_network(net,cfg)
        if blob is None:result['status']='no_engine_returned'
        else:
            with engine.open('xb') as f:f.write(bytes(blob))
            result.update(status='built_not_validated',candidate_engine_sha256=q.digest(engine),engine_bytes=engine.stat().st_size)
    except Exception as err:result.update(status='failed',error=type(err).__name__);raise
    finally:
        if start is not None:result['build_s']=time.perf_counter()-start
        q.save(target,result)
    print(json.dumps({k:result.get(k) for k in ('status','engine_bytes','build_s')}),flush=True)


if __name__=='__main__':
    a=h.arguments(('prepare','build'))
    if a.stage!=q.BASE/'staging/ner_mixed_e115_r1':raise ValueError('Unexpected E115 stage')
    {'prepare':prepare,'build':build}[a.mode](a)
