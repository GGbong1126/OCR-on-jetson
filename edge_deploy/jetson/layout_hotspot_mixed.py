"""E110: three profiled layout convolutions, unchanged FP32 sampling/business.

Board-only, isolated artifacts. Probe original graph ancestors in RAM, without
plugins or downstream detections; this is not TRT accumulation/accuracy proof.
"""
import argparse
import contextlib
import ctypes
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import optimize_rec_fp16 as old
import quantize_yolo_ptq as q

SOURCE = q.BASE / 'staging/deformable-r1/layout/trt_v3.onnx'
SOURCE_SHA = '11136202dd84166ba70bc6df3a46a2e2cf69d95e0b037992fa3a72f7e9049ece'
PLUGIN = q.RELEASE / 'pth/tensorrt/libautoocr_grid.so'
PLUGIN_SHA = 'e4c9181f9e9b870ea2b01e84a7518b01b98da66b7bc4a0c01799af745855f62b'
DEPLOYMENT_SHA = '2ced56d7d35e429199d51282eaaf961d1408bf2bc0f2c5784fcfb2b0fecba6d8'
PLAN_SHA = '6aec96dddfba1a9fe75175258681351fe05902593e075d5be75a24d4c8dff0e0'
TRT_VERSION = '8.2.1.8+localcp310'
TARGETS = ('Conv.91', 'Conv.92', 'Conv.93')
CONSUMERS = {'Conv.91': [('Mul.18', 0), ('Sigmoid.11', 0)],
             'Conv.92': [('Mul.19', 0), ('Sigmoid.13', 0)],
             'Conv.93': [('Mul.20', 0), ('Sigmoid.15', 0)]}
OUTPUTS = {'Conv.91': 'Add.35', 'Conv.92': 'Add.37', 'Conv.93': 'Add.39'}
SHAPES = {'image': [1, 3, 800, 800], 'im_shape': [1, 2], 'scale_factor': [1, 2]}
COMPUTE = {'CONVOLUTION', 'DECONVOLUTION', 'ACTIVATION', 'POOLING', 'SCALE',
           'ELEMENTWISE', 'UNARY', 'REDUCE', 'SOFTMAX', 'MATRIX_MULTIPLY',
           'FULLY_CONNECTED', 'EINSUM'}
STRUCTURAL = {'CONSTANT', 'SHUFFLE', 'IDENTITY', 'CONCATENATION', 'RESIZE',
              'SLICE', 'GATHER', 'SELECT', 'TOPK', 'FILL', 'SCATTER', 'PLUGIN_V2'}


def provenance():
    expected = {SOURCE: SOURCE_SHA, PLUGIN: PLUGIN_SHA,
                q.RELEASE/'edge_deploy/jetson/deployment.json': DEPLOYMENT_SHA,
                q.BASE/'staging/quant_yolo_e92_r1/plan.json': PLAN_SHA}
    if any(q.digest(p) != h for p, h in expected.items()):
        raise ValueError('Original source/plugin/deployment/plan differs')
    return dict(script_sha256=q.digest(__file__), source_sha256=SOURCE_SHA,
                plugin_sha256=PLUGIN_SHA, deployment_sha256=DEPLOYMENT_SHA,
                plan_sha256=PLAN_SHA, helper_sha256=q.digest(old.__file__),
                business_helper_sha256=q.digest(q.__file__))


def sha_tensor(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def input_metadata(feeds):
    if set(feeds) != set(SHAPES):
        raise ValueError('Unexpected layout input names')
    for name, value in feeds.items():
        if (list(value.shape) != SHAPES[name] or value.dtype != np.float32
                or not np.isfinite(value).all()):
            raise ValueError('Invalid layout input contract')
    return {k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=sha_tensor(v))
            for k, v in sorted(feeds.items())}


def describe(row):
    return dict(image_sha256=row['image_sha256'], image_type=row['image_type'],
                inputs=input_metadata(row['feeds']),
                reference_sha256={k: sha_tensor(v) for k, v in sorted(row['reference'].items())})


def output_names(session, names=None):
    return list(names) if names is not None else [v.name for v in session.get_outputs()]


class GuardedCall:
    """Latch instrumentation errors even if original business catches them."""
    def __init__(self, function):
        self.function = function
        self.errors = []

    def __call__(self, *args, **kwargs):
        self.require_healthy()
        try:
            return self.function(*args, **kwargs)
        except Exception as e:
            self.errors.append(type(e).__name__)
            raise

    def require_healthy(self):
        if self.errors:
            raise RuntimeError('Experiment callback failed: '+','.join(self.errors))


def capture(a):
    plan = q.load_plan(q.BASE/'staging/quant_yolo_e92_r1')
    rows = []; used = 0; models = base = original = None
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = old.setup(a)
            base = models.models['layout']; original = base.run; active = {}
            def run(names, feeds):
                nonlocal used
                input_metadata(feeds)
                out = original(names, feeds)
                used += sum(v.nbytes for v in list(feeds.values()) + out)
                # Layout's third int32 [300,200,200] output is 48 MB per call.
                # Retain it for complete output diagnostics; never write tensors.
                if len(rows) >= 6 or used > 512*1024**2:
                    raise RuntimeError('Layout RAM capture budget exceeded')
                rows.append(dict(active, feeds={k: v.copy() for k, v in feeds.items()},
                                 reference={k: v.copy() for k, v in zip(output_names(base, names), out)}))
                return out
            guarded = GuardedCall(run); base.run = guarded
            for image in plan['calibration'][:3]:
                for kind in (3, 1):
                    active.update(image_sha256=image['sha256'], image_type=kind)
                    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                        out = pipeline.process(str(Path(plan['images'])/image['file']), image_type=kind, run_ner=True)
                    guarded.require_healthy()
                    if not out['success'] or not models.healthy:
                        raise RuntimeError('Original offline business capture failed')
            execution = models.stats()
    finally:
        if original is not None: base.run = original
        old.close_models(models)
    if len(rows) != 6: raise ValueError('Expected six original layout calls')
    return rows, cfg, execution


def tensor_range(value):
    if value.dtype != np.float32 or not value.size or not np.isfinite(value).all():
        raise ValueError('Expected nonempty finite float32')
    return dict(shape=list(value.shape), elements=value.size,
                min=float(value.min()), max=float(value.max()),
                abs_max=float(np.abs(value).max()),
                abs_gt_fp16_max=int(np.count_nonzero(np.abs(value) > 65504)))


def ancestor_indices(nodes, outputs, roots):
    """Minimal closed ancestor set; reject missing values and duplicate producers."""
    by = {}
    for i, node in enumerate(nodes):
        for value in node.output:
            if value in by: raise ValueError('Duplicate producer')
            by[value] = i
    selected = set(); visiting = set()
    def visit(value):
        if not value or value in roots: return
        if value not in by: raise ValueError('Missing producer')
        i = by[value]
        if i in visiting: raise ValueError('Graph cycle')
        if i in selected: return
        visiting.add(i)
        for parent in nodes[i].input: visit(parent)
        visiting.remove(i); selected.add(i)
    for value in outputs: visit(value)
    return sorted(selected)


def ranges(a):
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper
    target = a.stage/'ranges.json'; old.fresh(target)
    report = dict(provenance(), status='running', rows=[], nodes=[])
    try:
        model = onnx.load(str(SOURCE)); nodes = {n.name: n for n in model.graph.node}
        weights = {v.name: v for v in model.graph.initializer}; names = []
        if len(model.graph.node) != 1377: raise ValueError('Graph changed')
        for name in TARGETS:
            n = nodes[name]
            if n.op_type != 'Conv' or list(n.output) != [OUTPUTS[name]]:
                raise ValueError('Target differs')
            for value in (n.input[0], n.output[0]):
                if value not in names: names.append(value)
            report['nodes'].append(dict(name=name, input=n.input[0], output=n.output[0],
                parameter_ranges={k: tensor_range(numpy_helper.to_array(weights[k])) for k in n.input[1:]}))
        roots = set(weights) | {v.name for v in model.graph.input}
        indices = ancestor_indices(model.graph.node, names, roots)
        retained = [model.graph.node[i] for i in indices]
        if any(n.domain or n.op_type == 'AutoOCRGridSample' for n in retained):
            raise ValueError('Unexpected custom operator in convolution prefix')
        needed = {v for n in retained for v in n.input}
        inputs = [v for v in model.graph.input if v.name in needed]
        if [v.name for v in inputs] != ['image']: raise ValueError('Prefix input differs')
        graph = onnx.helper.make_graph(retained, 'layout_hotspot_prefix', inputs,
            [onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, [1, 256, 100, 100]) for n in names],
            [v for v in model.graph.initializer if v.name in needed])
        model.graph.CopyFrom(graph); onnx.checker.check_model(model)
        data = model.SerializeToString()
        report.update(prefix_nodes=len(retained), prefix_sha256=hashlib.sha256(data).hexdigest(),
                      probe_outputs=names, onnx_version=onnx.__version__, ort_version=ort.__version__)
        del model, weights, nodes, graph, retained
        rows, cfg, execution = capture(a); gc.collect()
        options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(data, sess_options=options, providers=['CPUExecutionProvider']); del data
        for row in rows:
            out = session.run(names, {'image': row['feeds']['image']})
            report['rows'].append(dict(describe(row), ranges={k: tensor_range(v) for k, v in zip(names, out)}))
            print('Range case', len(report['rows']), flush=True)
        report.update(status='fp32_prefix_ranges_measured_not_fp16_validated', providers=session.get_providers(),
            original_execution=execution,
            note='Original converted FP32 graph ancestors only; 3 existing development images x ordinary/enhanced. CPU optimizer disabled, added intermediate outputs, RAM only. No final-output CPU comparison; not TRT accumulator safety or FP16 accuracy proof.')
        require_ranges(report)
    except Exception as e:
        report.update(status='failed', error=type(e).__name__); raise
    finally: q.save(target, report)


def require_ranges(report):
    if (report['status'] != 'fp32_prefix_ranges_measured_not_fp16_validated'
            or report['source_sha256'] != SOURCE_SHA or len(report['rows']) != 6
            or [r['name'] for r in report['nodes']] != list(TARGETS)):
        raise ValueError('Incomplete/mismatched prefix ranges')
    for n in report['nodes']:
        if len(n['parameter_ranges']) != 2: raise ValueError('Missing weight/bias ranges')
        values = list(n['parameter_ranges'].values()) + [r['ranges'][key]
                  for r in report['rows'] for key in (n['input'], n['output'])]
        if any(v['abs_gt_fp16_max'] or not np.isfinite(v['abs_max']) or v['abs_max'] > 32752 for v in values):
            raise ValueError('Insufficient measured FP16 range headroom')


def rewire(net, trt):
    layers = [net.get_layer(i) for i in range(net.num_layers)]; targets = []
    # Validate every target before mutating any boundary.
    for name in TARGETS:
        found = [v for v in layers if v.name == name]
        if len(found) != 1 or str(found[0].type).split('.')[-1] != 'CONVOLUTION':
            raise ValueError('Unique convolution missing')
        conv = found[0]; source = conv.get_output(0)
        consumers = [(v, j) for v in layers for j in range(v.num_inputs)
                     if v.get_input(j) is not None and v.get_input(j).name == source.name]
        if source.name != OUTPUTS[name] or sorted((v.name, j) for v, j in consumers) != CONSUMERS[name]:
            raise ValueError('Original target consumers differ')
        targets.append((name, source, consumers))
    result = []
    for name, source, consumers in targets:
        restore = net.add_identity(source)
        if restore is None: raise RuntimeError('Cannot create FP32 boundary')
        restore.name = name.replace('.', '') + '_restore_fp32'; restore.set_output_type(0, trt.float32)
        former = source.name; source.name = former + '__fp16'
        out = restore.get_output(0); out.name = former
        for layer, j in consumers: layer.set_input(j, out)
        result.append(dict(name=name, consumers=[dict(name=v.name, input_index=j) for v, j in consumers]))
    return result


def layer_policy(name, kind, outputs):
    if name in TARGETS:
        if kind != 'CONVOLUTION' or outputs != [0]: raise ValueError('Half target contract differs')
        return 'half_compute_half_output'
    if outputs and kind not in COMPUTE | STRUCTURAL: raise ValueError('Unhandled float layer: ' + kind)
    return 'fp32' if outputs else 'unchanged_integer_or_shape'


def build(a):
    import tensorrt as trt
    import onnx
    target = a.stage/'build.json'; engine = a.stage/'layout_hotspot.engine'
    for p in (target, engine, a.stage/'timing.cache'): old.fresh(p)
    ranges_report = json.loads((a.stage/'ranges.json').read_text()); require_ranges(ranges_report)
    now = provenance()
    if any(ranges_report[k] != v for k, v in now.items()): raise ValueError('Probe provenance changed')
    report = dict(now, status='building', targets=TARGETS, constraints=[], accepted_for_release=False,
                  ranges_sha256=q.digest(a.stage/'ranges.json'))
    start = None
    try:
        if trt.__version__ != TRT_VERSION: raise ValueError('Exact TRT version required')
        handle = ctypes.CDLL(str(PLUGIN), mode=ctypes.RTLD_GLOBAL)
        logger = trt.Logger(trt.Logger.INFO); trt.init_libnvinfer_plugins(logger, '')
        builder = trt.Builder(logger); net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(net, logger)
        if not parser.parse(SOURCE.read_bytes()): raise RuntimeError('Original converted graph parse failed')
        # get_layer returns base ILayer on this local TRT8.2 Python binding.
        # Validate plugin identity from the hash-pinned ONNX, not an unavailable
        # IPluginV2Layer.plugin attribute on the returned base object.
        graph = onnx.load(str(SOURCE))
        grid_names = {n.name for n in graph.graph.node if n.op_type == 'AutoOCRGridSample' and n.domain == 'autoocr'}
        if len(grid_names) != 18: raise ValueError('Original plugin nodes differ')
        del graph
        shapes = {net.get_input(i).name: list(net.get_input(i).shape) for i in range(net.num_inputs)}
        if shapes != SHAPES or net.num_outputs != 3: raise ValueError('Layout I/O differs')
        report['parsed_layers'] = net.num_layers; report['rewired'] = rewire(net, trt)
        plugins = 0
        for i in range(net.num_layers):
            layer = net.get_layer(i); kind = str(layer.type).split('.')[-1]
            indices = [j for j in range(layer.num_outputs) if layer.get_output(j) is not None
                       and layer.get_output(j).dtype == trt.float32 and not layer.get_output(j).is_shape_tensor]
            policy = layer_policy(layer.name, kind, indices)
            if kind == 'PLUGIN_V2':
                if layer.name not in grid_names: raise ValueError('Unknown plugin')
                plugins += 1
            if indices:
                half = layer.name in TARGETS
                if kind in COMPUTE: layer.precision = trt.float16 if half else trt.float32
                for j in indices: layer.set_output_type(j, trt.float16 if half else trt.float32)
            report['constraints'].append(dict(name=layer.name, kind=kind, output_indices=indices, policy=policy))
        if plugins != 18: raise ValueError('Expected 18 unchanged GridSample plugins')
        config = builder.create_builder_config(); config.max_workspace_size = 512*1024**2
        # Same default timing repetitions as the original builder, no blind tactic override.
        if config.avg_timing_iterations != 1: raise ValueError('Timing default changed')
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        config.set_flag(trt.BuilderFlag.FP16); config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        config.clear_flag(trt.BuilderFlag.TF32); config.clear_flag(trt.BuilderFlag.INT8)
        profile = builder.create_optimization_profile()
        for i in range(net.num_inputs):
            item = net.get_input(i); profile.set_shape(item.name, item.shape, item.shape, item.shape)
            if [list(v) for v in profile.get_shape(item.name)] != [SHAPES[item.name]]*3: raise ValueError('Profile differs')
            item.dtype = trt.float32; item.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        for i in range(net.num_outputs): net.get_output(i).allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        config.add_optimization_profile(profile); config.set_timing_cache(config.create_timing_cache(b''), False)
        report.update(trt_version=trt.__version__, profile=SHAPES, workspace_bytes=config.max_workspace_size,
            avg_timing_iterations=config.avg_timing_iterations, flags=int(config.flags),
            tactic_sources=config.get_tactic_sources(), plugin_count=plugins, network_layers=net.num_layers,
            gate='6 same-input finite/repeat/actual-format diagnostics then 26 strict business cases. Only afterwards 2 reverse-order rounds, 6 inputs x 10 timed cycles, 3 warmup cycles; >=5% mean improvement both, p95 <=1.05x original, extra warm NvMap <=256MiB. Not automatic release.',
            note='Only Conv.91/92/93 half compute and outputs, explicit identity restoring both FP32 SiLU branches per convolution. Other float computation/output constrained FP32, shape/integer unchanged. Original float-only GridSample plugin. OBEY, no relaxed retry, no INT8/pruning/input/threshold changes.')
        start = time.perf_counter(); blob = builder.build_serialized_network(net, config)
        if blob is None:
            report['status'] = 'no_engine_returned'
        else:
            with engine.open('xb') as f: f.write(bytes(blob))
            with (a.stage/'timing.cache').open('xb') as f: f.write(bytes(config.get_timing_cache().serialize()))
            report.update(status='built_not_validated', engine_sha256=q.digest(engine), engine_bytes=engine.stat().st_size)
    except Exception as e:
        report.update(status='failed', error=type(e).__name__); raise
    finally:
        if start is not None: report['build_s'] = time.perf_counter()-start
        q.save(target, report)
    print(json.dumps({k: report.get(k) for k in ('status', 'build_s', 'engine_bytes')}), flush=True)


def arguments(modes):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=modes)
    p.add_argument('--stage', required=True, type=Path); a = p.parse_args(); a.stage = a.stage.resolve()
    if platform.machine() != 'aarch64' or not a.stage.is_relative_to(q.BASE/'staging') or not a.stage.is_dir():
        p.error('Existing isolated board stage required')
    sys.dont_write_bytecode = True; sys.path.insert(0, str(q.RELEASE/'edge_deploy/jetson'))
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE), AUTOOCR_RUNTIME_DIR=str(a.stage/'runtime'), AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL)
    return a


if __name__ == '__main__':
    a = arguments(('ranges', 'build')); {'ranges': ranges, 'build': build}[a.mode](a)
