"""E113: audit cell consumers and test an isolated three-convolution boundary.

Not a replacement cell model. Original business inputs/features stay in RAM.
FP32 and mixed micro-controls share topology/weights; no HTTP or release edits.
"""
import ast
import contextlib
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import layout_hotspot_mixed as helper

q = helper.q
old = helper.old
SOURCE = q.BASE / 'staging/deformable-r1/cell/trt.onnx'
SOURCE_SHA = '9659f6daa6c2910fdf9f51e630323755de1bcef311a9d0ce72d5b073986a525a'
ENGINE_SHA = '0fe6af5c9c228c003cfae920c2b477aa0aa50150563a0277d1bca423ab2e22c5'
CONSUMER_SHA = 'de6cdb587b189678247c9850fbeb4782e5972c9f0627ec31972d57250c6dbfd1'
SHAPES = dict(image=[1, 3, 640, 640], im_shape=[1, 2], scale_factor=[1, 2])
SHAPE = [1, 256, 80, 80]
TARGETS = ('Conv.91', 'Conv.92', 'Conv.93')
BLOCK = ('Conv.91', 'Sigmoid.11', 'Mul.22', 'Conv.92', 'Sigmoid.13', 'Mul.23', 'Conv.93')
FEATURES = ('Mul.21', 'Add.43', 'p2o.pd_op.silu.3.0', 'Add.45', 'p2o.pd_op.silu.4.0', 'Add.47')
OUTPUTS = ('Add.43', 'Add.45', 'Add.47')
RULES = dict(scope='isolated_block_only_no_full_model_acceptance',
             range_abs_max_limit=32752, control_rtol=1e-3, control_atol=1e-4,
             finite_and_repeat_exact=True, actual_half_convolutions_only=list(TARGETS),
             fp32_silu_and_external_io=True, benchmark_authorized=False,
             accepted_for_release=False)


def read(path):
    return json.loads(Path(path).read_text())


def metadata():
    cfg_path = q.RELEASE / 'edge_deploy/jetson/deployment.json'
    cfg = read(cfg_path)
    expected = {SOURCE: SOURCE_SHA, cfg_path: helper.DEPLOYMENT_SHA,
                helper.PLUGIN: helper.PLUGIN_SHA,
                q.BASE/'staging/quant_yolo_e92_r1/plan.json': helper.PLAN_SHA,
                q.RELEASE/cfg['engines']['cell']['path']: ENGINE_SHA}
    if any(q.digest(p) != h for p, h in expected.items()):
        raise ValueError('Source/release/plugin/plan changed')
    return dict(source_sha256=SOURCE_SHA, engine_sha256=ENGINE_SHA,
                deployment_sha256=helper.DEPLOYMENT_SHA, plugin_sha256=helper.PLUGIN_SHA,
                dataset_plan_sha256=helper.PLAN_SHA, script_sha256=q.digest(__file__),
                helpers_sha256={Path(m.__file__).name:q.digest(m.__file__) for m in (helper, old, q)})


def consumer_audit(text):
    tree = ast.parse(text)
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_detect_cells_onnx']
    if len(nodes) != 1:
        raise ValueError('Unique cell consumer missing')
    node = nodes[0]
    indices = []
    for n in ast.walk(node):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == 'outputs':
            if not isinstance(n.slice, ast.Constant) or type(n.slice.value) is not int:
                raise ValueError('Dynamic output indexing')
            indices.append(n.slice.value)
    if sorted(indices) != [0, 1]:
        raise ValueError('Output consumer differs')
    return dict(method_sha256=hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest(),
                indices=indices, both_model_outputs_consumed=True)


def inputs_meta(feeds):
    if set(feeds) != set(SHAPES):
        raise ValueError('Cell inputs differ')
    for name, value in feeds.items():
        if list(value.shape) != SHAPES[name] or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Invalid cell inputs')
    return {k:dict(shape=list(v.shape), dtype=str(v.dtype), sha256=helper.sha_tensor(v)) for k,v in sorted(feeds.items())}


def describe(row):
    return dict(image_sha256=row['image_sha256'], image_type=row['image_type'], call_index=row['call_index'],
                inputs=inputs_meta(row['feeds']),
                original_output_sha256={k:helper.sha_tensor(v) for k,v in row['reference'].items()})


def capture(a):
    plan = q.load_plan(q.BASE/'staging/quant_yolo_e92_r1')
    rows, pages = [], []
    models = base = original = None
    used = 0
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                _, pipeline, models, cfg = old.setup(a)
            base = models.models['cell']; original = base.run; active = {}
            def run(names, feeds):
                nonlocal used
                inputs_meta(feeds)
                values = original(names, feeds)
                used += sum(v.nbytes for v in list(feeds.values()) + values)
                if used > 384*1024**2 or len(rows) >= 64:
                    raise RuntimeError('RAM capture budget exceeded')
                rows.append(dict(active, call_index=len(rows), feeds={k:v.copy() for k,v in feeds.items()},
                                 reference={k:v.copy() for k,v in zip(helper.output_names(base, names),values)}))
                return values
            guard = helper.GuardedCall(run); base.run = guard
            for image in plan['calibration'][:3]:
                for kind in (3, 1):
                    active.update(image_sha256=image['sha256'], image_type=kind)
                    start = len(rows)
                    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                        result = pipeline.process(str(Path(plan['images'])/image['file']), image_type=kind, run_ner=True)
                    guard.require_healthy()
                    if not result['success'] or not models.healthy:
                        raise RuntimeError('Original business failed')
                    pages.append(dict(active, cell_calls=len(rows)-start))
            execution = models.stats()
    finally:
        if original is not None: base.run = original
        old.close_models(models)
    if len(pages) != 6 or not rows:
        raise ValueError('Missing real cell coverage')
    return rows, pages, execution


def graph_parts():
    import onnx
    from onnx import helper as oh, numpy_helper as nh
    model = onnx.load(str(SOURCE)); g = model.graph
    if len(g.node) != 1195 or [v.name for v in g.output] != ['fetch_name_0', 'fetch_name_1']:
        raise ValueError('Cell graph changed')
    if model.functions or g.sparse_initializer or any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for n in g.node for a in n.attribute):
        raise ValueError('Nested graph not audited')
    nodes = {n.name:n for n in g.node}; weights = {v.name:v for v in g.initializer}
    if len(nodes) != len(g.node): raise ValueError('Duplicate node names')
    roots = set(weights) | {v.name for v in g.input}
    ancestry = helper.ancestor_indices(g.node, FEATURES, roots)
    prefix = [g.node[i] for i in ancestry]
    if any(n.domain for n in prefix): raise ValueError('Custom operator in prefix')
    needed = {v for n in prefix for v in n.input}
    prefix_inputs = [v for v in g.input if v.name in needed]
    if [v.name for v in prefix_inputs] != ['image']: raise ValueError('Prefix requires additional inputs')
    p = copy.deepcopy(model)
    p.graph.CopyFrom(oh.make_graph(prefix, 'cell_original_fp32_prefix', prefix_inputs,
        [oh.make_tensor_value_info(n,onnx.TensorProto.FLOAT,SHAPE) for n in FEATURES],
        [v for v in g.initializer if v.name in needed]))
    onnx.checker.check_model(p)
    block = [nodes[n] for n in BLOCK]; parameters = []
    for i, name in enumerate(TARGETS):
        n = nodes[name]
        if n.op_type != 'Conv' or n.input[0] != FEATURES[i*2] or list(n.output) != [OUTPUTS[i]]:
            raise ValueError('Hotspot edge changed')
        attrs = {a.name:oh.get_attribute_value(a) for a in n.attribute}
        if (attrs.get('kernel_shape') != [3,3] or attrs.get('pads') != [1,1,1,1]
                or attrs.get('strides') != [1,1] or attrs.get('dilations',[1,1]) != [1,1] or attrs.get('group',1) != 1):
            raise ValueError('Hotspot convolution contract differs')
        if len(n.input) != 3: raise ValueError('Weight/bias missing')
        arrays = [nh.to_array(weights[k]) for k in n.input[1:]]
        if [list(v.shape) for v in arrays] != [[256,256,3,3],[256]]: raise ValueError('Parameters differ')
        parameters.append(dict(name=name, input=n.input[0], output=n.output[0],
            parameters={k:dict(helper.tensor_range(v),sha256=helper.sha_tensor(v)) for k,v in zip(n.input[1:],arrays)}))
    used = {v for n in block for v in n.input}
    produced = {v for n in block for v in n.output}
    if used-set(weights)-produced != {FEATURES[0]}: raise ValueError('Block is not closed')
    b = copy.deepcopy(model)
    b.graph.CopyFrom(oh.make_graph(block,'cell_three_convolutions',
        [oh.make_tensor_value_info(FEATURES[0],onnx.TensorProto.FLOAT,SHAPE)],
        [oh.make_tensor_value_info(n,onnx.TensorProto.FLOAT,SHAPE) for n in OUTPUTS],
        [v for v in g.initializer if v.name in used]))
    onnx.checker.check_model(b)
    live = helper.ancestor_indices(g.node,[v.name for v in g.output],roots)
    return p.SerializeToString(), b.SerializeToString(), dict(graph_nodes=len(g.node), live_nodes=len(live),
        unconsumed_graph_outputs=[], original_outputs=[v.name for v in g.output],
        plugin_count=sum(n.op_type=='AutoOCRGridSample' for n in g.node),
        prefix_nodes=len(prefix), block_nodes=BLOCK, parameters=parameters,
        source_block_node_bytes_equal=all(a.SerializeToString()==c.SerializeToString() for a,c in zip(block,b.graph.node)))


def prefix_session(data):
    import onnxruntime as ort
    opt = ort.SessionOptions(); opt.intra_op_num_threads=2; opt.inter_op_num_threads=1
    opt.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(data,sess_options=opt,providers=['CPUExecutionProvider'])


def range_gate(report):
    if report['status'] != 'original_cell_ranges_measured' or report['rules'] != RULES or not report['rows']:
        raise ValueError('Missing original ranges/rules')
    ranges = [v for n in report['graph']['parameters'] for v in n['parameters'].values()]
    ranges += [v for r in report['rows'] for v in r['ranges'].values()]
    if any(not np.isfinite(v['abs_max']) or v['abs_max']>RULES['range_abs_max_limit'] or v['abs_gt_fp16_max'] for v in ranges):
        raise ValueError('Insufficient measured FP16 headroom')


def require_ranges(stage):
    report=read(stage/'ranges.json')
    if any(report[k]!=v for k,v in metadata().items()): raise ValueError('Range provenance changed')
    range_gate(report)
    if q.digest(stage/'cell_hotspot_block.onnx')!=report['block_sha256']: raise ValueError('Block changed')
    return report


def ranges(a):
    target=a.stage/'ranges.json'; old.fresh(target); old.fresh(a.stage/'cell_hotspot_block.onnx')
    report=dict(metadata(),status='running',rules=RULES,rows=[])
    try:
        audit=consumer_audit((q.RELEASE/'server.py').read_text())
        if audit['method_sha256']!=CONSUMER_SHA: raise ValueError('Consumer source changed')
        report['consumer']=audit
        refs=[]
        for path in sorted(q.RELEASE.rglob('*.py')):
            hits=[i+1 for i,line in enumerate(path.read_text(encoding='utf-8').splitlines()) if '_table_cell_processor' in line]
            if hits: refs.append(dict(file=str(path.relative_to(q.RELEASE)),sha256=q.digest(path),lines=hits))
        report['consumer_references']=refs
        if {r['file'] for r in refs}!={'server.py'}: raise ValueError('Cell consumer references differ')
        # Adapter uses the path/role factory, not the pipeline attribute name.
        cfg=read(q.RELEASE/'edge_deploy/jetson/deployment.json')
        adapter=q.RELEASE/'edge_deploy/jetson/jetson_models.py'
        if q.digest(adapter)!=cfg['adapter_sha256']['jetson_models.py']:raise ValueError('Adapter changed')
        report['factory_adapter_sha256']=q.digest(adapter)
        prefix,block,graph=graph_parts()
        if graph['plugin_count']!=18: raise ValueError('Unexpected plugin count')
        with (a.stage/'cell_hotspot_block.onnx').open('xb') as f: f.write(block)
        report.update(graph=graph,prefix_sha256=hashlib.sha256(prefix).hexdigest(),block_sha256=hashlib.sha256(block).hexdigest())
        rows,pages,execution=capture(a); gc.collect(); cpu=prefix_session(prefix)
        for row in rows:
            values=cpu.run(list(FEATURES),{'image':row['feeds']['image']})
            if any(list(v.shape)!=SHAPE for v in values): raise ValueError('Feature shapes differ')
            report['rows'].append(dict(describe(row),ranges={k:helper.tensor_range(v) for k,v in zip(FEATURES,values)},
                feature_sha256={k:helper.sha_tensor(v) for k,v in zip(FEATURES,values)}))
            print('Original cell feature case',len(report['rows']),flush=True)
        report.update(status='original_cell_ranges_measured',pages=pages,original_execution=execution,
            providers=cpu.get_providers(),note='Existing 3 development images x two paths; CPU FP32 prefix, optimizer disabled. Range safety is not mixed numerical/GT/business acceptance. No tensors/text saved, no timing benchmark.')
        range_gate(report)
    except Exception as e:
        report.update(status='failed',error=type(e).__name__); raise
    finally: q.save(target,report)


def build(a,variant):
    import tensorrt as trt
    require_ranges(a.stage)
    target=a.stage/('build_'+variant+'.json');engine=a.stage/(variant+'.engine')
    old.fresh(target);old.fresh(engine)
    report=dict(metadata(),status='building',variant=variant,rules=RULES,ranges_sha256=q.digest(a.stage/'ranges.json'))
    start=None
    try:
        if trt.__version__!=helper.TRT_VERSION: raise ValueError('TRT version differs')
        logger=trt.Logger(trt.Logger.INFO);builder=trt.Builder(logger)
        net=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH));parser=trt.OnnxParser(net,logger)
        if not parser.parse((a.stage/'cell_hotspot_block.onnx').read_bytes()): raise RuntimeError('Block parse failed')
        if net.num_inputs!=1 or net.num_outputs!=3 or list(net.get_input(0).shape)!=SHAPE: raise ValueError('Block interface differs')
        layers=[net.get_layer(i) for i in range(net.num_layers)]
        for name,outname in zip(TARGETS,OUTPUTS):
            found=[n for n in layers if n.name==name]
            if len(found)!=1: raise ValueError('Target convolution missing')
            layer=found[0];tensor=layer.get_output(0)
            if tensor.name!=outname: raise ValueError('Target output differs')
            consumers=[(n,i) for n in layers for i in range(n.num_inputs) if n.get_input(i) is not None and n.get_input(i).name==outname]
            net.unmark_output(tensor);tensor.name=outname+'__compute'
            restore=net.add_identity(tensor);restore.name=name.replace('.','')+'_restore_fp32';restore.set_output_type(0,trt.float32)
            output=restore.get_output(0);output.name=outname;output.dtype=trt.float32;net.mark_output(output)
            for n,i in consumers:n.set_input(i,output)
        for i in range(net.num_layers):
            n=net.get_layer(i);kind=str(n.type).split('.')[-1]
            if kind not in {'CONVOLUTION','ACTIVATION','ELEMENTWISE','IDENTITY','CONSTANT'}:raise ValueError('Unexpected block layer type')
            dtype=trt.float16 if variant=='mixed' and n.name in TARGETS else trt.float32
            if kind in {'CONVOLUTION','ACTIVATION','ELEMENTWISE'}:n.precision=dtype
            for j in range(n.num_outputs):n.set_output_type(j,dtype)
        for i in range(net.num_outputs):net.get_output(i).allowed_formats=1<<int(trt.TensorFormat.LINEAR)
        net.get_input(0).allowed_formats=1<<int(trt.TensorFormat.LINEAR)
        config=builder.create_builder_config();config.max_workspace_size=512*1024**2;config.avg_timing_iterations=1
        config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        for f in (trt.BuilderFlag.TF32,trt.BuilderFlag.INT8,trt.BuilderFlag.FP16):config.clear_flag(f)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        if variant=='mixed':config.set_flag(trt.BuilderFlag.FP16)
        profile=builder.create_optimization_profile();profile.set_shape(FEATURES[0],SHAPE,SHAPE,SHAPE)
        if [list(v) for v in profile.get_shape(FEATURES[0])]!=[SHAPE]*3:raise ValueError('Profile differs')
        config.add_optimization_profile(profile);config.set_timing_cache(config.create_timing_cache(b''),False)
        report.update(trt_version=trt.__version__,flags=int(config.flags),workspace_bytes=config.max_workspace_size,
            profile=SHAPE,avg_timing_iterations=config.avg_timing_iterations,network_layers=net.num_layers,
            note='Only isolated Conv91/92/93 requested half; SiLU and external boundaries FP32. Three intermediate outputs retained for diagnostics. Not deployable model, no accuracy/speed claim.')
        start=time.perf_counter();blob=builder.build_serialized_network(net,config)
        if blob is None:raise RuntimeError('No engine returned')
        with engine.open('xb') as f:f.write(bytes(blob))
        report.update(status='built_not_validated',candidate_engine_sha256=q.digest(engine),engine_bytes=engine.stat().st_size)
    except Exception as e:
        report.update(status='failed',error=type(e).__name__);raise
    finally:
        if start is not None:report['build_s']=time.perf_counter()-start
        q.save(target,report)
    print(json.dumps({k:report[k] for k in ('status','variant','build_s','engine_bytes')}),flush=True)


def compare(x,y):
    if x.dtype!=np.float32 or y.dtype!=np.float32 or x.shape!=y.shape or not x.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid micro output')
    delta=np.abs(x.astype(np.float64)-y.astype(np.float64))
    return dict(mae=float(delta.mean()),max_abs_error=float(delta.max()),
                allclose_diagnostic=bool(np.allclose(x,y,rtol=RULES['control_rtol'],atol=RULES['control_atol'])))


def inspect(info,variant):
    rows=info.get('Layers',[])
    if not rows or not all(isinstance(r,dict) for r in rows):raise ValueError('Missing detailed Inspector')
    def dtype(tensors,kind):
        pattern=r'\b(?:half|fp16)\b' if kind=='half' else r'\b(?:float|fp32)\b'
        return bool(tensors) and all(re.search(pattern,t.get('Format/Datatype','').lower()) for t in tensors)
    selected=[]
    for name in TARGETS:
        found=[r for r in rows if re.search(r'(?<![\w.])'+re.escape(name)+r'(?![\w.])',r.get('Name','')) and r.get('ParameterType')=='Convolution']
        if len(found)!=1:raise ValueError('Unique executed convolution missing')
        row=found[0];kind='half' if variant=='mixed' else 'float'
        if not dtype(row.get('Inputs',[]),kind) or not dtype(row.get('Outputs',[]),kind):raise ValueError('Executed convolution format differs')
        selected.append(row)
    for row in rows:
        tensors=row.get('Inputs',[])+row.get('Outputs',[])
        if any(re.search(r'\bint8\b',t.get('Format/Datatype','').lower()) for t in tensors):raise ValueError('Unexpected INT8')
        half=any(re.search(r'\b(?:half|fp16)\b',t.get('Format/Datatype','').lower()) for t in tensors)
        if half and (variant=='fp32' or (row not in selected and row.get('LayerType') not in ('Reformat','NoOp'))):
            raise ValueError('Unexpected half computation')
    for name in OUTPUTS:
        bindings=[t for r in rows for t in r.get('Outputs',[]) if t.get('Name')==name]
        if not dtype(bindings,'float'):raise ValueError('FP32 boundary missing')
    return dict(layers=len(rows),convolutions=selected,external_fp32=True,
                note='Inspector I/O formats do not prove internal accumulator precision.')


def check(a):
    from trt82_backend import Session
    prior=require_ranges(a.stage);target=a.stage/'check.json';old.fresh(target)
    report=dict(metadata(),status='running',rules=RULES,ranges_sha256=q.digest(a.stage/'ranges.json'),rows=[])
    sessions={}
    try:
        builds={v:read(a.stage/('build_'+v+'.json')) for v in ('fp32','mixed')}
        for v,b in builds.items():
            if b['status']!='built_not_validated' or b['rules']!=RULES or b['ranges_sha256']!=report['ranges_sha256'] or any(b[k]!=val for k,val in metadata().items()):
                raise ValueError('Build provenance differs')
        report['build_sha256']={v:q.digest(a.stage/('build_'+v+'.json')) for v in builds}
        raw,pages,execution=capture(a)
        if [describe(r) for r in raw]!=[{k:r[k] for k in describe(raw[0])} for r in prior['rows']] or pages!=prior['pages']:
            raise ValueError('Original cell capture drift')
        prefix,_,_=graph_parts();cpu=prefix_session(prefix)
        for v,b in builds.items():
            sessions[v]=Session(a.stage/(v+'.engine'),b['candidate_engine_sha256'],'cell_block_'+v)
            sessions[v].outputs.sort(key=lambda t:OUTPUTS.index(t.name))
        for i,row in enumerate(raw):
            features=cpu.run(list(FEATURES),{'image':row['feeds']['image']})
            if {k:helper.sha_tensor(v) for k,v in zip(FEATURES,features)}!=prior['rows'][i]['feature_sha256']:
                raise ValueError('Original CPU feature changed')
            values={};pair=dict(describe(row),feature_sha256=prior['rows'][i]['feature_sha256'],variants={})
            for v,native in sessions.items():
                out=native.run(None,{FEATURES[0]:features[0]});again=native.run(None,{FEATURES[0]:features[0]})
                if not all(np.array_equal(x,y) for x,y in zip(out,again)):raise ValueError('Repeat output changed')
                pair['variants'][v]={name:dict(compare(features[FEATURES.index(name)],value),output_sha256=helper.sha_tensor(value)) for name,value in zip(OUTPUTS,out)}
                values[v]=out
            pair['repeat_exact']=True
            pair['mixed_vs_fp32']={name:compare(x,y) for name,x,y in zip(OUTPUTS,values['fp32'],values['mixed'])}
            report['rows'].append(pair);print('Cell block compared',i+1,flush=True)
        report['inspection']={}
        for v,native in sessions.items():
            inspector=native.engine.create_engine_inspector();inspector.execution_context=native.context
            info=json.loads(inspector.get_engine_information(native.trt.LayerInformationFormat.JSON))
            q.save(a.stage/('layers_'+v+'.json'),info);report['inspection'][v]=inspect(info,v)
        report.update(status='isolated_boundary_compared_not_full_model_accepted',original_execution=execution,
            execution={v:dict(n.stats(),precision='fp32_control' if v=='fp32' else 'three_convs_half_fp32_silu') for v,n in sessions.items()},
            control_allclose_cases=sum(all(x['allclose_diagnostic'] for x in r['variants']['fp32'].values()) for r in report['rows']),
            mixed_allclose_cases=sum(all(x['allclose_diagnostic'] for x in r['mixed_vs_fp32'].values()) for r in report['rows']),
            note='CPU FP32 prefix features, not GPU-original hidden activations. Matched-topology GPU micro-control. Numeric tolerance diagnostic; not effective boxes/text/GT or performance gate. No benchmark or release promotion.')
    except Exception as e:
        report.update(status='failed',error=type(e).__name__);raise
    finally:
        for n in sessions.values():n.close()
        q.save(target,report)
    print(json.dumps({k:report[k] for k in ('status','control_allclose_cases','mixed_allclose_cases')}),flush=True)


if __name__=='__main__':
    a=helper.arguments(('ranges','build_fp32','build_mixed','check'))
    if a.stage!=q.BASE/'staging/cell_probe_e113_r2':raise ValueError('Unexpected E113 staging path')
    if a.mode=='ranges':ranges(a)
    elif a.mode=='check':check(a)
    else:build(a,a.mode.removeprefix('build_'))
