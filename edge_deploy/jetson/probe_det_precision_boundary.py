"""E105 isolated Conv.87 precision boundary, never a replacement detector.

Original weights and real FP32 feature tensors remain on the board/in RAM.
This tests kernel/format feasibility, not downstream OCR accuracy or speed.
"""
import argparse
import gc
import json
import logging
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
import det_hotspot_mixed as previous
import optimize_rec_fp16 as old
import quantize_yolo_ptq as q

VARIANTS=('fp32','fp16_direct_fp32','fp16_via_cast')
PROFILE=[(1,256,8,8),(1,256,240,176),(1,256,240,240)]
PRIOR=q.BASE/'staging/det_hotspot_e104_r1'
RANGES_SHA='ee46841c8ad172f31aa430b7746d7b8c1f5d700dae72244b83ce2ba7c8092737'


def policy(variant):
    if variant not in VARIANTS:raise ValueError('Unknown precision policy')
    return dict(conv_compute='float' if variant=='fp32' else 'half',
                conv_output='half' if variant=='fp16_via_cast' else 'float',
                identity_output='float',external_io='float')


def provenance():
    original=previous.provenance()
    if q.digest(PRIOR/'ranges.json')!=RANGES_SHA:raise ValueError('Prior ranges changed')
    previous.require_ranges(json.loads((PRIOR/'ranges.json').read_text()))
    return dict(script_sha256=q.digest(__file__),helpers_sha256={Path(m.__file__).name:q.digest(m.__file__) for m in (previous,old,q)},
                source_sha256=original['source_sha256'],deployment_sha256=original['deployment_sha256'],
                plan_sha256=original['plan_sha256'],ranges_sha256=RANGES_SHA)


def node_contract(node,weights,attrs):
    if (node.name!='Conv.87' or node.op_type!='Conv' or len(node.input)!=2 or len(node.output)!=1
            or node.input[0]!='p2o.pd_op.add.4.0' or node.output[0]!='p2o.pd_op.conv2d.60.0'
            or weights.dtype!=np.float32 or weights.shape!=(64,256,9,9)
            or not np.isfinite(weights).all()
            or attrs.get('kernel_shape')!=[9,9] or attrs.get('strides')!=[1,1]
            or attrs.get('pads')!=[4,4,4,4] or attrs.get('dilations',[1,1])!=[1,1]
            or attrs.get('group',1)!=1 or attrs.get('auto_pad',b'NOTSET')!=b'NOTSET'):
        raise ValueError('Unexpected source convolution contract')


def get_weights():
    import onnx
    from onnx import numpy_helper,helper
    model=onnx.load(str(previous.SOURCE))
    matches=[n for n in model.graph.node if n.name=='Conv.87']
    if len(matches)!=1:raise ValueError('Unique source hotspot missing')
    node=matches[0]
    weights=np.ascontiguousarray(numpy_helper.to_array(next(v for v in model.graph.initializer if v.name==node.input[1])))
    node_contract(node,weights,{a.name:helper.get_attribute_value(a) for a in node.attribute})
    return weights


def add_hotspot(network,x,weights,trt):
    # Empty NumPy arrays can retain a non-null data pointer: invalid bias in 8.2.
    conv=network.add_convolution_nd(x,64,(9,9),weights,trt.Weights(trt.float32))
    if conv is None:raise RuntimeError('Cannot create bias-free hotspot convolution')
    conv.name='Conv.87';conv.stride_nd=(1,1);conv.padding_nd=(4,4)
    return conv


def build(a):
    import tensorrt as trt
    path=a.stage/('build_'+a.variant+'.json');engine=a.stage/(a.variant+'.engine')
    old.fresh(path);old.fresh(engine)
    report=dict(provenance(),status='building',variant=a.variant,policy=policy(a.variant),profile=PROFILE,
                workspace_bytes=256*1024**2,avg_timing_iterations=8,
                note='Single original Conv.87 plus FP32 identity boundary; external FP32 LINEAR I/O. Not full-model topology/profile validation. No input data during build, no INT8/TF32, no precision relaxation.')
    try:
        if not trt.__version__.startswith('8.2.1'):raise ValueError('TRT8.2.1 required')
        weights=get_weights()
        logger=trt.Logger(trt.Logger.INFO);builder=trt.Builder(logger)
        network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        x=network.add_input('x',trt.float32,(-1,256,-1,-1))
        conv=add_hotspot(network,x,weights,trt)
        types={'float':trt.float32,'half':trt.float16};p=policy(a.variant)
        conv.precision=types[p['conv_compute']];conv.set_output_type(0,types[p['conv_output']])
        restore=network.add_identity(conv.get_output(0));restore.name='restore_fp32'
        restore.set_output_type(0,trt.float32)
        y=restore.get_output(0);y.name='y';network.mark_output(y)
        for tensor in (x,y):tensor.dtype=trt.float32;tensor.allowed_formats=1<<int(trt.TensorFormat.LINEAR)
        config=builder.create_builder_config();config.max_workspace_size=report['workspace_bytes']
        config.avg_timing_iterations=8;config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        config.clear_flag(trt.BuilderFlag.TF32);config.clear_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        if a.variant!='fp32':config.set_flag(trt.BuilderFlag.FP16)
        profile=builder.create_optimization_profile();profile.set_shape('x',*PROFILE)
        if [tuple(v) for v in profile.get_shape('x')]!=PROFILE:raise ValueError('Profile mismatch')
        config.add_optimization_profile(profile);config.set_timing_cache(config.create_timing_cache(b''),False)
        report.update(trt_version=trt.__version__,flags=int(config.flags),tactic_sources=config.get_tactic_sources(),
                      weight_sha256=previous.sha_tensor(weights),weight_range=previous.tensor_range(weights))
        start=time.perf_counter();blob=builder.build_serialized_network(network,config)
        report['build_s']=time.perf_counter()-start
        if blob is None:
            report.update(status='no_engine_returned',reason='Read builder log; cannot classify every failure as unsupported precision.')
        else:
            with engine.open('xb') as f:f.write(bytes(blob))
            report.update(status='built_not_executed',engine_sha256=q.digest(engine),engine_bytes=engine.stat().st_size)
    except Exception as e:
        report.update(status='failed',error=type(e).__name__);raise
    finally:q.save(path,report)
    print(json.dumps({k:report[k] for k in ('variant','status','build_s')}),flush=True)


def inspect(info,variant):
    rows=info.get('Layers',[])
    selected=[r for r in rows if isinstance(r,dict) and re.search(r'(?<![\w.])Conv\.87(?![\w.])',r.get('Name',''))
              and (r.get('ParameterType')=='Convolution' or r.get('LayerType')=='Convolution')]
    if len(selected)!=1:raise ValueError('Unique executed convolution missing')
    conv=selected[0];p=policy(variant)
    def dtype(tensors,kind):
        names={'half':r'\b(?:half|fp16)\b','float':r'\b(?:float|fp32)\b'}
        return bool(tensors) and all(re.search(names[kind],r.get('Format/Datatype','').lower()) for r in tensors)
    if not dtype(conv.get('Inputs',[]),p['conv_compute']) or not dtype(conv.get('Outputs',[]),p['conv_output']):
        raise ValueError('Executed hotspot formats do not match boundary')
    float_outputs=[r for r in rows if any(t.get('Name')=='y' for t in r.get('Outputs',[]))]
    if not float_outputs or not all(dtype(r['Outputs'],'float') for r in float_outputs):
        raise ValueError('FP32 external output boundary not evidenced')
    return dict(layer_count=len(rows),convolution=conv,
                note='Actual Inspector input/output formats and tactic, not internal accumulator precision.')


def compare(ref,out):
    if (ref.dtype!=np.float32 or out.dtype!=np.float32 or ref.shape!=out.shape
            or ref.ndim!=4 or ref.shape[0]!=1 or ref.shape[1]!=64
            or not np.isfinite(ref).all() or not np.isfinite(out).all()):
        raise ValueError('Invalid finite FP32 convolution outputs')
    delta=np.abs(ref-out)
    return dict(mae=float(delta.mean()),max_abs_error=float(delta.max()),
                reference_abs_max=float(np.abs(ref).max()),candidate_abs_max=float(np.abs(out).max()),
                allclose_diagnostic=bool(np.allclose(ref,out,rtol=1e-3,atol=1e-4)))


def load_builds(stage):
    meta=provenance();reports={}
    for variant in VARIANTS:
        r=json.loads((stage/('build_'+variant+'.json')).read_text())
        if any(r[k]!=v for k,v in meta.items()) or r['policy']!=policy(variant):raise ValueError('Build provenance/policy differs')
        if r['status']=='built_not_executed':
            if q.digest(stage/(variant+'.engine'))!=r['engine_sha256']:raise ValueError('Engine changed')
            reports[variant]=r
        elif variant=='fp16_direct_fp32' and r['status']=='no_engine_returned':
            log=(stage/('build_'+variant+'.log')).read_text()
            if ('No implementation of layer Conv.87 obeys the requested constraints.' not in log
                    or 'kOBEY_PRECISION_CONSTRAINTS was set' not in log):raise ValueError('Unclassified build failure')
        else:raise ValueError('Required control or cast engine unavailable')
    return reports


def check(a):
    import onnx
    import onnxruntime as ort
    from trt82_backend import Session
    target=a.stage/'check.json';old.fresh(target)
    builds=load_builds(a.stage)
    report=dict(provenance(),status='running',rows=[],synthetic=[],build_sha256={v:q.digest(a.stage/('build_'+v+'.json')) for v in VARIANTS},
                note='6 original FP32 CPU feature tensors; repeat/output-finite and formats check. Synthetic min/max shapes are compatibility tests. No downstream OCR, GT, benchmark or precision/accuracy acceptance.')
    sessions={}
    try:
        raw,cfg,execution=previous.capture(a,3)
        before=json.loads((PRIOR/'ranges.json').read_text())['rows']
        if [previous.describe(r) for r in raw]!=[{k:r[k] for k in previous.describe(raw[0])} for r in before]:
            raise ValueError('Original detector input/output changed')
        model=onnx.load(str(previous.SOURCE));node=next(n for n in model.graph.node if n.name=='Conv.87')
        names=[node.input[0],node.output[0]]
        model.graph.output.extend(onnx.helper.make_tensor_value_info(n,onnx.TensorProto.FLOAT,None) for n in names)
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        cpu=ort.InferenceSession(model.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
        del model;gc.collect()
        for variant,r in builds.items():
            session=Session(a.stage/(variant+'.engine'),r['engine_sha256'],'isolated_'+variant)
            sessions[variant]=session
        for index,row in enumerate(raw):
            x,y=cpu.run(names,{'x':row['x']})
            # Added probes can affect optimizations; require same original ranges.
            if previous.tensor_range(x)!=before[index]['ranges'][names[0]] or previous.tensor_range(y)!=before[index]['ranges'][names[1]]:
                raise ValueError('Original feature ranges changed')
            pair=dict(previous.describe(row),feature_shape=list(x.shape),feature_sha256=previous.sha_tensor(x),cpu_output_sha256=previous.sha_tensor(y),variants={})
            outputs={}
            for variant,native in sessions.items():
                out=native.run(None,{'x':x})[0];again=native.run(None,{'x':x})[0]
                if not np.array_equal(out,again):raise ValueError('Repeat output changed')
                pair['variants'][variant]=dict(compare(y,out),repeat_exact=True,output_sha256=previous.sha_tensor(out))
                outputs[variant]=out
            if not pair['variants']['fp32']['allclose_diagnostic']:raise ValueError('FP32 micro-control differs from source')
            pair['mixed_vs_fp32']=compare(outputs['fp32'],outputs['fp16_via_cast'])
            report['rows'].append(pair);print('Feature case',index+1,flush=True)
            del x,y,outputs,out,again
        del cpu,raw;gc.collect()
        for shape in (PROFILE[0],PROFILE[1],PROFILE[2]):
            x=np.random.default_rng(105).uniform(-1,1,shape).astype(np.float32)
            outputs={k:n.run(None,{'x':x})[0] for k,n in sessions.items()}
            report['synthetic'].append(dict(shape=shape,input_sha256=previous.sha_tensor(x),mixed_vs_fp32=compare(outputs['fp32'],outputs['fp16_via_cast'])))
        report['inspection']={}
        for variant,native in sessions.items():
            inspector=native.engine.create_engine_inspector();inspector.execution_context=native.context
            info=json.loads(inspector.get_engine_information(native.trt.LayerInformationFormat.JSON))
            q.save(a.stage/('layers_'+variant+'.json'),info)
            report['inspection'][variant]=inspect(info,variant)
        report.update(status='boundary_executed_not_full_model_validated',original_execution=execution,
                      execution={k:dict(v.stats(),precision_policy=policy(k)) for k,v in sessions.items()})
        for value in report['execution'].values():value.pop('precision',None)
    except Exception as e:
        report.update(status='failed',error=type(e).__name__);raise
    finally:
        for native in sessions.values():native.close()
        q.save(target,report)
    print(json.dumps({'status':report['status'],'real_cases':len(report['rows']),'synthetic_cases':len(report['synthetic'])}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('build','check'))
    p.add_argument('--stage',required=True,type=Path);p.add_argument('--variant',choices=VARIANTS,default='fp32')
    a=p.parse_args();a.stage=a.stage.resolve()
    if platform.machine()!='aarch64' or not a.stage.is_relative_to(q.BASE/'staging') or not a.stage.is_dir():p.error('Isolated Jetson staging required')
    sys.dont_write_bytecode=True;sys.path.insert(0,str(q.RELEASE/'edge_deploy/jetson'))
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE),AUTOOCR_RUNTIME_DIR=str(a.stage/'runtime'),AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL)
    {'build':build,'check':check}[a.mode](a)


if __name__=='__main__':main()
