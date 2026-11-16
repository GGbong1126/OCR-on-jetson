"""E104 bounded detection hotspot experiment, no release or input changes.

Original business tensors only; range probes are in-memory ORT FP32 diagnostics,
not a reproduction of TensorRT fusion/accumulation. No text/tensors in reports.
"""
import argparse
import contextlib
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

SOURCE = q.RELEASE / 'pth/onnx_rec/ocr_det.onnx'
SOURCE_SHA = 'b4413116303d0b3fcf6266dfb1ac5ce6b8e0b6f8955fbf9b9a2f08e402f810b2'
DEPLOYMENT_SHA = '2ced56d7d35e429199d51282eaaf961d1408bf2bc0f2c5784fcfb2b0fecba6d8'
PROFILE = [(1,3,32,32), (1,3,960,704), (1,3,960,960)]
TARGETS = ('Conv.87',)
PROBE_NODES = ('Conv.86','Conv.87','Conv.91','Conv.139','BatchNormalization.84','Conv.140')
COMPUTE = {'CONVOLUTION','DECONVOLUTION','ACTIVATION','POOLING','SCALE',
           'ELEMENTWISE','UNARY','REDUCE','SOFTMAX','MATRIX_MULTIPLY','FULLY_CONNECTED'}
REORDER = {'CONSTANT','SHUFFLE','IDENTITY','CONCATENATION','RESIZE','SLICE','GATHER','SELECT'}


def sha_tensor(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def provenance():
    if (q.digest(SOURCE) != SOURCE_SHA or
            q.digest(q.RELEASE/'edge_deploy/jetson/deployment.json') != DEPLOYMENT_SHA):
        raise ValueError('Original model or deployment differs')
    return dict(script_sha256=q.digest(__file__), source_sha256=SOURCE_SHA,
                deployment_sha256=DEPLOYMENT_SHA, helper_sha256=q.digest(old.__file__),
                plan_sha256=q.digest(q.BASE/'staging/quant_yolo_e92_r1/plan.json'))


def tensor_range(x):
    if x.dtype != np.float32 or not x.size or not np.isfinite(x).all():
        raise ValueError('Expected finite FP32 range probe')
    return dict(shape=list(x.shape), elements=x.size, min=float(x.min()), max=float(x.max()),
                abs_max=float(np.abs(x).max()), abs_gt_fp16_max=int(np.sum(np.abs(x)>65504)))


def capture(a, image_count):
    """Capture original det tensor and output, bounded RAM only, both paths."""
    plan = q.load_plan(q.BASE/'staging/quant_yolo_e92_r1')
    feeds = []
    models = base = original = None
    try:
        with open(os.devnull,'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server,pipeline,models,cfg = old.setup(a)
            base = models.models['det'].gpu
            original = base.run
            active = {}
            def run(names, values):
                out = original(names,values)
                if len(feeds) >= 2*image_count or sum(v['x'].nbytes+v['reference'].nbytes for v in feeds) + values['x'].nbytes + out[0].nbytes > 128*1024**2:
                    raise RuntimeError('RAM capture budget exceeded')
                feeds.append(dict(active, x=values['x'].copy(), reference=out[0].copy()))
                return out
            base.run = run
            for image in plan['calibration'][:image_count]:
                for kind in (3,1):
                    active.update(image_sha256=image['sha256'], image_type=kind)
                    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                        out = pipeline.process(str(Path(plan['images'])/image['file']),image_type=kind,run_ner=True)
                    if not out['success'] or not models.healthy:
                        raise RuntimeError('Original business failed')
            execution = models.stats()
    finally:
        if original is not None:
            base.run = original
        old.close_models(models)
    if len(feeds) != 2*image_count:
        raise ValueError('Incomplete original capture')
    return feeds,cfg,execution


def describe(row):
    return dict(image_sha256=row['image_sha256'], image_type=row['image_type'],
                input_shape=list(row['x'].shape), input_sha256=sha_tensor(row['x']),
                reference_sha256=sha_tensor(row['reference']))


def ranges(a):
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper
    path = a.stage/'ranges.json'; old.fresh(path)
    report = dict(provenance(), status='running', rows=[])
    try:
        model=onnx.load(str(SOURCE))
        weights={t.name:numpy_helper.to_array(t) for t in model.graph.initializer}
        nodes={n.name:n for n in model.graph.node}
        if len(model.graph.node)!=595 or not all(n in nodes for n in PROBE_NODES):
            raise ValueError('Unexpected graph topology')
        names=[]; report['nodes']=[]
        for name in PROBE_NODES:
            n=nodes[name]
            if n.op_type not in ('Conv','BatchNormalization'):
                raise ValueError('Unexpected probe operator')
            for value in (n.input[0],n.output[0]):
                if value not in names: names.append(value)
            report['nodes'].append(dict(name=name,op=n.op_type,input=n.input[0],output=n.output[0],
                parameter_ranges={k:tensor_range(weights[k]) for k in n.input[1:] if k in weights}))
        # Keep final output as a baseline sanity check. Added outputs can inhibit
        # fusion; these are original-graph FP32 ranges, not TRT kernel ranges.
        final=model.graph.output[0].name
        model.graph.output.extend(onnx.helper.make_tensor_value_info(n,onnx.TensorProto.FLOAT,None) for n in names)
        feeds,cfg,execution=capture(a,3)
        gc.collect()
        options=ort.SessionOptions(); options.intra_op_num_threads=2; options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session=ort.InferenceSession(model.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
        report.update(providers=session.get_providers(), original_execution=execution,
                      note='3 existing development images x ordinary/enhanced; original CPU FP32 graph with added outputs and optimizations disabled. Finite/safe range is not proof of FP16 accuracy or TRT internal accumulation safety. No persisted tensors/model copy.')
        del model,weights,nodes
        for row in feeds:
            outputs=session.run([final]+names,{'x':row['x']})
            if outputs[0].shape!=row['reference'].shape:
                raise ValueError('Final probability shape differs')
            ranges_={n:tensor_range(v) for n,v in zip(names,outputs[1:])}
            report['rows'].append(dict(describe(row),ranges=ranges_,
                final_output=tensor_range(outputs[0]),
                final_max_abs_error=float(np.max(np.abs(outputs[0]-row['reference']))),
                final_allclose_diagnostic=bool(np.allclose(outputs[0],row['reference'],rtol=1e-3,atol=1e-4))))
            print('Range case',len(report['rows']),flush=True)
        report['status']='fp32_ranges_measured_not_fp16_validated'
    except Exception as e:
        report.update(status='failed',error=type(e).__name__)
        raise
    finally:
        q.save(path,report)


def require_ranges(report):
    if (report['status']!='fp32_ranges_measured_not_fp16_validated' or len(report['rows'])!=6
            or report['source_sha256']!=SOURCE_SHA):
        raise ValueError('Missing complete source-matched range check')
    nodes={r['name']:r for r in report['nodes']}
    for name in TARGETS:
        n=nodes[name]
        if n['op']!='Conv' or not n['parameter_ranges']:
            raise ValueError('Not an inspected convolution')
        values=list(n['parameter_ranges'].values()) + [r['ranges'][key] for r in report['rows'] for key in (n['input'],n['output'])]
        if any(v['abs_gt_fp16_max'] or not np.isfinite(v['abs_max']) or v['abs_max']>32752 for v in values):
            raise ValueError('Hotspot lacks measured FP16 range headroom')


def layer_policy(name, kind, floating_outputs):
    if name in TARGETS:
        if kind!='CONVOLUTION' or floating_outputs!=[0]:
            raise ValueError('Selected hotspot contract differs')
        return 'fp16_compute_fp32_output'
    if floating_outputs and kind not in COMPUTE|REORDER:
        raise ValueError('Unhandled floating layer: '+kind)
    return 'fp32' if floating_outputs else 'unchanged_nonfloat_or_shape'


def build(a):
    import tensorrt as trt
    path=a.stage/'build.json'; engine=a.stage/'det_hotspot.engine'
    old.fresh(path); old.fresh(engine); old.fresh(a.stage/'timing.cache')
    ranges_report=json.loads((a.stage/'ranges.json').read_text()); require_ranges(ranges_report)
    report=dict(provenance(),status='building',targets=TARGETS,ranges_sha256=q.digest(a.stage/'ranges.json'),constraints=[])
    if not trt.__version__.startswith('8.2.1'):
        raise ValueError('Expected TRT8.2.1')
    logger=trt.Logger(trt.Logger.INFO); builder=trt.Builder(logger)
    network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser=trt.OnnxParser(network,logger)
    if not parser.parse(SOURCE.read_bytes()):
        raise RuntimeError('; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if network.num_inputs!=1 or network.get_input(0).name!='x' or network.num_outputs!=1:
        raise ValueError('Unexpected I/O')
    found=[]
    for i in range(network.num_layers):
        layer=network.get_layer(i); kind=str(layer.type).split('.')[-1]
        outputs=[j for j in range(layer.num_outputs) if layer.get_output(j) is not None
                 and layer.get_output(j).dtype==trt.float32 and not layer.get_output(j).is_shape_tensor]
        policy=layer_policy(layer.name,kind,outputs)
        if layer.name in TARGETS: found.append(layer.name)
        if not outputs: continue
        if kind in COMPUTE:
            layer.precision=trt.float16 if layer.name in TARGETS else trt.float32
        for j in outputs:
            layer.set_output_type(j,trt.float32)
            if layer.get_output_type(j)!=trt.float32:
                raise RuntimeError('FP32 output constraint not retained')
        report['constraints'].append(dict(index=i,name=layer.name,kind=kind,policy=policy,output_indices=outputs))
    if sorted(found)!=sorted(TARGETS): raise ValueError('Hotspot layer missing or duplicated')
    config=builder.create_builder_config(); config.max_workspace_size=256*1024**2
    config.avg_timing_iterations=8; config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    config.set_flag(trt.BuilderFlag.FP16); config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    config.clear_flag(trt.BuilderFlag.INT8); config.clear_flag(trt.BuilderFlag.TF32)
    profile=builder.create_optimization_profile(); profile.set_shape('x',*PROFILE)
    if [tuple(v) for v in profile.get_shape('x')]!=PROFILE: raise ValueError('Profile differs')
    config.add_optimization_profile(profile);config.set_timing_cache(config.create_timing_cache(b''),False)
    for tensor in (network.get_input(0),network.get_output(0)):
        tensor.dtype=trt.float32; tensor.allowed_formats=1<<int(trt.TensorFormat.LINEAR)
    report.update(profile=PROFILE,workspace_bytes=config.max_workspace_size,avg_timing_iterations=8,
                  trt_version=trt.__version__,flags=int(config.flags),tactic_sources=config.get_tactic_sources(),
                  parsed_layers=network.num_layers,
                  note='Only Conv.87 requested FP16 compute, float outputs and all other float compute FP32; integer/shape unchanged. OBEY, no auto relaxation. Actual Inspector and real business required. No INT8, pruning, resize or threshold change.')
    start=time.perf_counter()
    try:
        blob=builder.build_serialized_network(network,config)
        if blob is None: raise RuntimeError('Strict mixed engine build failed')
        with engine.open('xb') as f:f.write(bytes(blob))
        with (a.stage/'timing.cache').open('xb') as f:f.write(bytes(config.get_timing_cache().serialize()))
        report.update(status='built_not_validated',engine_sha256=q.digest(engine),engine_bytes=engine.stat().st_size)
    except Exception as e:
        report.update(status='failed',error=type(e).__name__)
        raise
    finally:
        report['build_s']=time.perf_counter()-start;q.save(path,report)
    print(json.dumps({k:report[k] for k in ('status','engine_bytes','build_s')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('ranges','build'))
    p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args();a.stage=a.stage.resolve()
    if platform.machine()!='aarch64' or not a.stage.is_relative_to(q.BASE/'staging') or not a.stage.is_dir():
        p.error('Existing isolated Jetson staging required')
    sys.dont_write_bytecode=True;sys.path.insert(0,str(q.RELEASE/'edge_deploy/jetson'))
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE),AUTOOCR_RUNTIME_DIR=str(a.stage/'runtime'),AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL)
    {'ranges':ranges,'build':build}[a.mode](a)


if __name__=='__main__':main()
