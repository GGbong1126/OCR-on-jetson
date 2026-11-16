"""E105 follow-up diagnostics after FP32 allclose gate; not acceptance.

Preserves failed check and immutable engines/build script. Original CPU prefix,
isolated CPU convolution and sampled FP64 sums distinguish source wiring from
finite-precision differences. Never prints/persists tensors or recognized text.
"""
import argparse
import gc
import json
import logging
import os
from pathlib import Path
import platform
import sys

import numpy as np
import probe_det_precision_boundary as probe

q=probe.q
previous=probe.previous


def error_metrics(reference,candidate):
    result=probe.compare(reference,candidate)
    delta=np.abs(reference.astype(np.float64)-candidate.astype(np.float64))
    scale=1e-4+1e-3*np.abs(reference.astype(np.float64))
    result.update(elements=delta.size,outside_allclose_count=int((delta>scale).sum()),
                  p99_abs_error=float(np.percentile(delta,99)),
                  max_scaled_error=float(np.max(delta/scale)))
    return result


def fp64_spots(x,weights,cpu,gpu):
    if x.shape[0]!=1 or x.shape[1]!=256 or weights.shape!=(64,256,9,9):raise ValueError('Unexpected shape')
    h,w=x.shape[2:];padded=np.pad(x,((0,0),(0,0),(4,4),(4,4)))
    rng=np.random.default_rng(105)
    points=[(int(rng.integers(64)),int(rng.integers(h)),int(rng.integers(w))) for _ in range(32)]
    delta=np.abs(cpu-gpu)
    points.append(tuple(int(v) for v in np.unravel_index(np.argmax(delta),cpu.shape)[1:]))
    scaled=delta/(1e-4+1e-3*np.abs(cpu))
    points.append(tuple(int(v) for v in np.unravel_index(np.argmax(scaled),cpu.shape)[1:]))
    errors={'cpu':[],'gpu':[]};magnitudes=[]
    for c,i,j in points:
        expected=np.sum(padded[0,:,i:i+9,j:j+9].astype(np.float64)*weights[c].astype(np.float64),dtype=np.float64)
        magnitudes.append(abs(float(expected)))
        errors['cpu'].append(abs(float(cpu[0,c,i,j])-expected))
        errors['gpu'].append(abs(float(gpu[0,c,i,j])-expected))
    return dict(sample_count=len(points),cpu_max_abs_error=float(max(errors['cpu'])),gpu_max_abs_error=float(max(errors['gpu'])),
                reference_abs_max=float(max(magnitudes)),
                note='32 fixed-seed positions plus worst absolute/scaled CPU-vs-GPU positions. FP64 cross-correlation of original FP32 input/weights, not exhaustive proof; no values/coordinates saved.')


def run(a):
    import onnx
    from onnx import helper
    import onnxruntime as ort
    from trt82_backend import Session
    builds=probe.load_builds(a.stage)
    failed=json.loads((a.stage/'check.json').read_text())
    log=(a.stage/'check.log').read_text()
    if failed['status']!='failed' or 'FP32 micro-control differs from source' not in log:raise ValueError('Expected preserved failed gate')
    target=a.stage/'diagnostic.json';probe.old.fresh(target)
    report=dict(probe.provenance(),diagnostic_script_sha256=q.digest(__file__),failed_check_sha256=q.digest(a.stage/'check.json'),
                status='running',rows=[],synthetic=[],acceptance_passed=False,
                note='Supplemental diagnostic, NOT a relaxed rerun of the failed FP32 allclose gate. Same six original CPU features, no OCR/GT/benchmark or release acceptance.')
    sessions={}
    try:
        raw,cfg,execution=previous.capture(a,3)
        before=json.loads((probe.PRIOR/'ranges.json').read_text())['rows']
        if [previous.describe(r) for r in raw]!=[{k:r[k] for k in previous.describe(raw[0])} for r in before]:raise ValueError('Original input/output drift')
        model=onnx.load(str(previous.SOURCE));node=next(n for n in model.graph.node if n.name=='Conv.87')
        names=[node.input[0],node.output[0]]
        weights=probe.get_weights()
        single=helper.make_model(helper.make_graph([node],'isolated_original_conv87',
            [helper.make_tensor_value_info(names[0],onnx.TensorProto.FLOAT,[1,256,'h','w'])],
            [helper.make_tensor_value_info(names[1],onnx.TensorProto.FLOAT,[1,64,'h','w'])],
            [next(t for t in model.graph.initializer if t.name==node.input[1])]),opset_imports=list(model.opset_import))
        single.ir_version=model.ir_version;onnx.checker.check_model(single)
        model.graph.output.extend(helper.make_tensor_value_info(n,onnx.TensorProto.FLOAT,None) for n in names)
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        cpu=ort.InferenceSession(model.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
        isolated=ort.InferenceSession(single.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
        del model,single;gc.collect()
        for variant,r in builds.items():sessions[variant]=Session(a.stage/(variant+'.engine'),r['engine_sha256'],'isolated_'+variant)
        for index,row in enumerate(raw):
            x,y=cpu.run(names,{'x':row['x']});control=isolated.run(None,{names[0]:x})[0]
            if not np.array_equal(control,y):raise ValueError('Isolated CPU differs from original CPU convolution')
            if previous.tensor_range(x)!=before[index]['ranges'][names[0]] or previous.tensor_range(y)!=before[index]['ranges'][names[1]]:raise ValueError('Feature ranges differ')
            pair=dict(previous.describe(row),feature_shape=list(x.shape),feature_sha256=previous.sha_tensor(x),cpu_output_sha256=previous.sha_tensor(y),isolated_cpu_exact=True,variants={})
            outputs={}
            for variant,native in sessions.items():
                out=native.run(None,{'x':x})[0];again=native.run(None,{'x':x})[0]
                if not np.array_equal(out,again):raise ValueError('Repeat changed')
                pair['variants'][variant]=dict(error_metrics(y,out),repeat_exact=True,output_sha256=previous.sha_tensor(out));outputs[variant]=out
            pair['mixed_vs_fp32']=error_metrics(outputs['fp32'],outputs['fp16_via_cast'])
            pair['fp64_spots']=fp64_spots(x,weights,y,outputs['fp32'])
            report['rows'].append(pair);print('Diagnostic case',index+1,flush=True)
            del x,y,control,outputs,out,again
        del cpu,isolated,raw;gc.collect()
        for shape in (probe.PROFILE[0],probe.PROFILE[1],probe.PROFILE[2]):
            x=np.random.default_rng(105).uniform(-1,1,shape).astype(np.float32)
            out={k:n.run(None,{'x':x})[0] for k,n in sessions.items()}
            report['synthetic'].append(dict(shape=shape,input_sha256=previous.sha_tensor(x),mixed_vs_fp32=error_metrics(out['fp32'],out['fp16_via_cast'])))
        report['inspection']={}
        for variant,native in sessions.items():
            inspector=native.engine.create_engine_inspector();inspector.execution_context=native.context
            info=json.loads(inspector.get_engine_information(native.trt.LayerInformationFormat.JSON))
            q.save(a.stage/('diagnostic_layers_'+variant+'.json'),info)
            report['inspection'][variant]=probe.inspect(info,variant)
        report.update(status='diagnosed_boundary_executed_not_accuracy_accepted',original_execution=execution,
                      execution={k:dict(v.stats(),precision_policy=probe.policy(k)) for k,v in sessions.items()})
        for value in report['execution'].values():value.pop('precision',None)
    except Exception as e:
        report.update(status='failed',error=type(e).__name__);raise
    finally:
        for native in sessions.values():native.close()
        q.save(target,report)
    print(json.dumps({'status':report['status'],'cases':len(report['rows']),'acceptance_passed':False}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);a=p.parse_args();a.stage=a.stage.resolve()
    if platform.machine()!='aarch64' or not a.stage.is_relative_to(q.BASE/'staging') or not a.stage.is_dir():p.error('Jetson staging required')
    sys.path.insert(0,str(q.RELEASE/'edge_deploy/jetson'));sys.dont_write_bytecode=True
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE),AUTOOCR_RUNTIME_DIR=str(a.stage/'runtime'),AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL);run(a)


if __name__=='__main__':main()
