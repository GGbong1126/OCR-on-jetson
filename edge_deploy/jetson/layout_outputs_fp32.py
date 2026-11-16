"""E111: remove only the unconsumed layout mask output and its dead ancestors.

Isolated FP32 graph/engine; no release, activation, HTTP or business rule edits.
"""
import ast
import copy
import ctypes
import json
import time

import layout_hotspot_mixed as source

q, old = source.q, source.old
KEEP = ('fetch_name_0', 'fetch_name_1')
DROP = 'fetch_name_2'
ENGINE = 'layout_outputs_fp32.engine'


def provenance():
    p = source.provenance()
    p['source_helper_sha256'] = p.pop('script_sha256')
    p['script_sha256'] = q.digest(__file__)
    p['server_sha256'] = q.digest(q.RELEASE/'server.py')
    return p


def audit_consumer(text):
    tree = ast.parse(text)
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
             and n.name == '_detect_layout_onnx']
    if len(found) != 1:
        raise ValueError('Unique layout consumer required')
    function = found[0]
    indices = []
    parents = {child: n for n in ast.walk(function) for child in ast.iter_child_nodes(n)}
    for n in ast.walk(function):
        if not isinstance(n, ast.Name) or n.id != 'outputs' or not isinstance(n.ctx, ast.Load):
            continue
        parent = parents[n]
        if isinstance(parent, ast.Subscript) and parent.value is n:
            if not isinstance(parent.slice, ast.Constant) or type(parent.slice.value) is not int:
                raise ValueError('Dynamic output consumer')
            indices.append(parent.slice.value)
        elif not (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name)
                  and parent.func.id == 'len' and parent.args == [n]):
            raise ValueError('Unexpected output escape')
    if sorted(indices) != [0, 1]:
        raise ValueError('Only output positions 0 and 1 may be consumed')
    return dict(function=function.name, line=function.lineno, consumed_indices=indices,
                length_check_allowed=True)


def prune(model):
    """Preserve retained protobuf nodes/weights exactly; no simplifier or re-export."""
    import onnx
    g = model.graph
    if (model.functions or g.sparse_initializer
            or any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                   for n in g.node for a in n.attribute)):
        raise ValueError('Nested/functions/sparse graph requires separate audit')
    if [v.name for v in g.output] != list(KEEP)+( [DROP] ):
        raise ValueError('Original three-output contract changed')
    roots = {v.name for v in g.input} | {v.name for v in g.initializer}
    indices = source.ancestor_indices(g.node, KEEP, roots)
    nodes = [g.node[i] for i in indices]
    needed = set(KEEP) | {v for n in nodes for v in n.input if v}
    inputs = [v for v in g.input if v.name in needed]
    if {v.name for v in inputs} != set(source.SHAPES):
        raise ValueError('Input dependency contract changed; investigate first')
    initializers = [v for v in g.initializer if v.name in needed]
    retained_values = needed | {v for n in nodes for v in n.output}
    out = copy.deepcopy(model)
    for field, values in (
        ('node', nodes), ('input', inputs), ('output', list(g.output)[:2]),
        ('initializer', initializers),
        ('value_info', [v for v in g.value_info if v.name in retained_values]),
    ):
        del getattr(out.graph, field)[:]
        getattr(out.graph, field).extend(values)
    if ([n.SerializeToString() for n in out.graph.node] != [n.SerializeToString() for n in nodes]
            or [n.SerializeToString() for n in out.graph.initializer] != [n.SerializeToString() for n in initializers]):
        raise ValueError('Retained graph bytes changed')
    onnx.checker.check_model(out)
    plugins = [n.name for n in nodes if n.op_type == 'AutoOCRGridSample']
    if len(plugins) != 18: raise ValueError('Plugin dependency changed')
    removed = [n for i, n in enumerate(g.node) if i not in set(indices)]
    return out, dict(original_nodes=len(g.node), retained_nodes=len(nodes),
        removed_nodes=[dict(name=n.name, op=n.op_type) for n in removed],
        original_initializer_count=len(g.initializer), retained_initializer_count=len(initializers),
        removed_initializer_bytes=sum(len(v.SerializeToString()) for v in g.initializer if v.name not in needed),
        retained_node_and_weight_bytes_equal=True, plugin_names=plugins,
        inputs=[v.name for v in inputs], kept_outputs=list(KEEP), removed_output=DROP,
        removed_output_shape=[300,200,200], removed_output_dtype='int32', removed_output_bytes=48000000)


def prepare(a):
    import onnx
    target = a.stage/'prepare.json'; old.fresh(target); old.fresh(a.stage/'layout_two_outputs.onnx')
    report = dict(provenance(), status='running')
    try:
        report['consumer'] = audit_consumer((q.RELEASE/'server.py').read_text())
        refs = []
        for path in sorted(q.RELEASE.rglob('*.py')):
            if path.is_symlink(): raise ValueError('Unexpected source symlink')
            lines = path.read_text(encoding='utf-8').splitlines()
            hits = [i+1 for i,line in enumerate(lines) if '_layout_detector' in line or DROP in line]
            if hits: refs.append(dict(file=str(path.relative_to(q.RELEASE)), sha256=q.digest(path), lines=hits))
        report['consumer_reference_files'] = refs
        if {v['file'] for v in refs} != {'server.py','edge_deploy/jetson/jetson_models.py'}:
            raise ValueError('Additional output consumer requires manual audit')
        model, audit = prune(onnx.load(str(source.SOURCE)))
        with (a.stage/'layout_two_outputs.onnx').open('xb') as f: f.write(model.SerializeToString())
        report.update(status='dead_output_graph_prepared_not_validated', graph=audit,
            candidate_onnx_sha256=q.digest(a.stage/'layout_two_outputs.onnx'), onnx_version=onnx.__version__,
            gate='Same 6 real inputs: finite, repeat exact, retained outputs allclose rtol=1e-3 atol=1e-4, int count exact, actual FP32/no half/int8. Then 26 strict full-business cases and unchanged fallback counts. Only then reverse-order 2 rounds x 6 inputs x 10 timed cycles, 3 warmup; mean >=5% faster both, p95 <=1.05x, extra warm NvMap <=256MiB. No auto release.',
            note='Only unused mask graph-output and exclusive ancestors/weights removed. No training/channel pruning, quantization, threshold, input or text normalization change. Candidate has TWO outputs, not a fabricated third output; original release adapter unchanged.')
    except Exception as e:
        report.update(status='failed', error=type(e).__name__); raise
    finally: q.save(target, report)
    print(json.dumps(dict(status=report['status'], graph=report['graph'])), flush=True)


def require_prepare(stage):
    report = json.loads((stage/'prepare.json').read_text())
    if (report['status'] != 'dead_output_graph_prepared_not_validated'
            or any(report[k] != v for k,v in provenance().items())
            or report['candidate_onnx_sha256'] != q.digest(stage/'layout_two_outputs.onnx')):
        raise ValueError('Preparation provenance changed')
    return report


def build(a):
    import tensorrt as trt
    prepared = require_prepare(a.stage)
    target = a.stage/'build.json'; engine = a.stage/ENGINE
    old.fresh(target); old.fresh(engine)
    report = dict(provenance(), prepare_sha256=q.digest(a.stage/'prepare.json'), status='running')
    start = None
    try:
        if trt.__version__ != source.TRT_VERSION: raise ValueError('TensorRT version changed')
        handle = ctypes.CDLL(str(source.PLUGIN), mode=ctypes.RTLD_GLOBAL)
        logger = trt.Logger(trt.Logger.INFO); trt.init_libnvinfer_plugins(logger, '')
        builder = trt.Builder(logger)
        net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(net, logger)
        if not parser.parse((a.stage/'layout_two_outputs.onnx').read_bytes()):
            raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        inputs = {net.get_input(i).name:list(net.get_input(i).shape) for i in range(net.num_inputs)}
        outputs = {net.get_output(i).name:list(net.get_output(i).shape) for i in range(net.num_outputs)}
        if inputs != source.SHAPES or set(outputs) != set(KEEP): raise ValueError('Parsed I/O differs')
        config = builder.create_builder_config(); config.max_workspace_size = 512*1024**2
        for flag in (trt.BuilderFlag.TF32,trt.BuilderFlag.FP16,trt.BuilderFlag.INT8): config.clear_flag(flag)
        config.avg_timing_iterations = 1
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        profile = builder.create_optimization_profile()
        for i in range(net.num_inputs):
            item = net.get_input(i); shape = list(item.shape)
            if item.dtype != trt.float32: raise ValueError('Input is not FP32')
            profile.set_shape(item.name, shape, shape, shape)
            if [list(s) for s in profile.get_shape(item.name)] != [shape]*3:
                raise ValueError('Profile readback differs')
        config.add_optimization_profile(profile)
        config.set_timing_cache(config.create_timing_cache(b''), False)
        report.update(trt_version=trt.__version__, inputs=inputs, outputs=outputs,
            candidate_onnx_sha256=prepared['candidate_onnx_sha256'], layers=net.num_layers,
            flags=int(config.flags), workspace_bytes=config.max_workspace_size,
            avg_timing_iterations=config.avg_timing_iterations, tactic_sources=config.get_tactic_sources())
        if report['flags'] != 0: raise ValueError('Expected FP32/noTF32 default flags')
        start = time.perf_counter(); blob = builder.build_serialized_network(net,config)
        if blob is None: raise RuntimeError('No engine returned')
        with engine.open('xb') as f: f.write(bytes(blob))
        report.update(status='built_not_validated',engine_sha256=q.digest(engine),engine_bytes=engine.stat().st_size)
    except Exception as e:
        report.update(status='failed',error=type(e).__name__); raise
    finally:
        if start is not None: report['build_s'] = time.perf_counter()-start
        q.save(target,report)
    print(json.dumps({k:report[k] for k in ('status','build_s','engine_bytes')}),flush=True)


if __name__ == '__main__':
    a = source.arguments(('prepare','build')); {'prepare':prepare,'build':build}[a.mode](a)
