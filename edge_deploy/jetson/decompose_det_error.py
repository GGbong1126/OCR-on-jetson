"""E106 distinguish FP16 operand/output rounding from selected kernel residual.

Uses the six E105 real features and immutable micro-engines. All full tensors and
temporary ONNX graphs stay in RAM on Jetson. Not a quality/performance gate.
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
import diagnose_det_boundary as diagnostic

q=probe.q
previous=probe.previous
PRIOR=q.BASE/'staging/det_boundary_e105_r2'


def round_half(x):
    if x.dtype!=np.float32 or not x.size or not np.isfinite(x).all() or np.max(np.abs(x))>65504:
        raise ValueError('Finite FP32 values within FP16 range required, no clipping')
    result=x.astype(np.float16).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('Nonfinite FP16 round trip')
    return result


def stats(x):
    d=np.asarray(x,dtype=np.float64)
    if not d.size or not np.isfinite(d).all():raise ValueError('Finite nonempty error required')
    absolute=np.abs(d)
    return dict(elements=d.size,mae=float(absolute.mean()),rmse=float(np.sqrt(np.mean(d*d))),
                max_abs_error=float(absolute.max()),p99_abs_error=float(np.percentile(absolute,99)))


def decompose(original,both,rounded_output,gpu):
    arrays=[np.asarray(v,dtype=np.float64) for v in (original,both,rounded_output,gpu)]
    if any(v.shape!=arrays[0].shape for v in arrays):raise ValueError('Shapes differ')
    y,b,rounded,g=arrays
    operand=b-y;output=rounded-b;residual=g-rounded;total=g-y
    result={k:stats(v) for k,v in (('operand_rounding',operand),('output_rounding',output),
                                 ('implementation_residual',residual),('total_gpu_vs_cpu',total))}
    result['signed_closure_max_abs']=float(np.max(np.abs(total-operand-output-residual)))
    if result['signed_closure_max_abs']>1e-9:raise ValueError('Signed decomposition does not close')
    result['note']='Signed arrays add; MAE/RMSE/max magnitudes do NOT add and are not accuracy percentages. Residual includes implementation arithmetic/algorithm differences and CPU reference rounding; not proof of accumulator dtype.'
    return result


def prior_reports():
    builds=probe.load_builds(PRIOR)
    r=json.loads((PRIOR/'diagnostic.json').read_text())
    summary=json.loads((PRIOR/'summary.json').read_text())
    if (r['status']!='diagnosed_boundary_executed_not_accuracy_accepted' or len(r['rows'])!=6
            or summary['diagnostic_sha256']!=q.digest(PRIOR/'diagnostic.json')
            or r['diagnostic_script_sha256']!=q.digest(diagnostic.__file__)
            or any(r[k]!=v for k,v in probe.provenance().items())):raise ValueError('Prior provenance differs')
    return builds,r


def run(a):
    import onnx
    from onnx import helper,numpy_helper
    import onnxruntime as ort
    from trt82_backend import Session
    target=a.stage/'decomposition.json';probe.old.fresh(target)
    builds,prior=prior_reports()
    report=dict(script_sha256=q.digest(__file__),helpers_sha256={Path(m.__file__).name:q.digest(m.__file__) for m in (probe,diagnostic,previous,probe.old,q)},
                source_sha256=previous.SOURCE_SHA,deployment_sha256=previous.DEPLOYMENT_SHA,
                prior_report_sha256=q.digest(PRIOR/'diagnostic.json'),build_sha256={v:q.digest(PRIOR/('build_'+v+'.json')) for v in builds},
                status='running',rows=[],acceptance_passed=False,
                note='6 unchanged original CPU features. ORT CPU FP32 single-node convolutions, 2 threads, sequential, graph optimizations disabled. Operand/output FP16 round-trip diagnostic, no clipping. Original TensorRT micro-engines reused. No new engine, final OCR/GT or timing benchmark.')
    sessions={}
    try:
        raw,cfg,execution=previous.capture(a,3)
        keys=previous.describe(raw[0]).keys()
        if [previous.describe(v) for v in raw]!=[{k:v[k] for k in keys} for v in prior['rows']]:raise ValueError('Original detector capture changed')
        model=onnx.load(str(previous.SOURCE));node=next(n for n in model.graph.node if n.name=='Conv.87')
        names=[node.input[0],node.output[0]];weights=probe.get_weights();rounded_weights=round_half(weights)
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        def cpu_session(weight):
            single=helper.make_model(helper.make_graph([node],'isolated_conv87',
                [helper.make_tensor_value_info(names[0],onnx.TensorProto.FLOAT,[1,256,'h','w'])],
                [helper.make_tensor_value_info(names[1],onnx.TensorProto.FLOAT,[1,64,'h','w'])],
                [numpy_helper.from_array(weight,name=node.input[1])]),opset_imports=list(model.opset_import))
            single.ir_version=model.ir_version;onnx.checker.check_model(single)
            return ort.InferenceSession(single.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
        cpu_original=cpu_session(weights);cpu_rounded=cpu_session(rounded_weights)
        model.graph.output.extend(helper.make_tensor_value_info(n,onnx.TensorProto.FLOAT,None) for n in names)
        prefix=ort.InferenceSession(model.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
        del model;gc.collect()
        report.update(weight_sha256=previous.sha_tensor(weights),rounded_weight_sha256=previous.sha_tensor(rounded_weights),
                      weight_rounding_error=stats(rounded_weights.astype(np.float64)-weights))
        for variant,b in builds.items():sessions[variant]=Session(PRIOR/(variant+'.engine'),b['engine_sha256'],'decompose_'+variant)
        for i,row in enumerate(raw):
            x,y=prefix.run(names,{'x':row['x']});old=prior['rows'][i]
            if previous.sha_tensor(x)!=old['feature_sha256'] or previous.sha_tensor(y)!=old['cpu_output_sha256']:raise ValueError('Feature/reference drift')
            baseline=cpu_original.run(None,{names[0]:x})[0]
            if not np.array_equal(baseline,y):raise ValueError('Original CPU operator drift')
            xr=round_half(x)
            x_only=cpu_original.run(None,{names[0]:xr})[0]
            w_only=cpu_rounded.run(None,{names[0]:x})[0]
            both=cpu_rounded.run(None,{names[0]:xr})[0]
            rounded_out=round_half(both)
            outputs={}
            for variant,native in sessions.items():
                out=native.run(None,{'x':x})[0]
                if previous.sha_tensor(out)!=old['variants'][variant]['output_sha256']:raise ValueError('Micro-engine output drift')
                outputs[variant]=out
            pair=dict(previous.describe(row),feature_sha256=old['feature_sha256'],feature_shape=list(x.shape),
                original_cpu_exact=True,previous_gpu_outputs_exact=True,input_rounding_error=stats(xr.astype(np.float64)-x),
                input_only=diagnostic.error_metrics(y,x_only),weight_only=diagnostic.error_metrics(y,w_only),
                both_operands=diagnostic.error_metrics(y,both),both_and_output=diagnostic.error_metrics(y,rounded_out),
                gpu_vs_rounded_reference=diagnostic.error_metrics(rounded_out,outputs['fp16_via_cast']),
                decomposition=decompose(y,both,rounded_out,outputs['fp16_via_cast']),
                rounded_fp64_spots=diagnostic.fp64_spots(xr,rounded_weights,both,outputs['fp16_via_cast']),
                output_sha256={k:previous.sha_tensor(v) for k,v in dict(input_only=x_only,weight_only=w_only,both=both,rounded_output=rounded_out,**outputs).items()})
            report['rows'].append(pair);print('Decomposed case',i+1,flush=True)
            del x,y,xr,baseline,x_only,w_only,both,rounded_out,outputs,out
        report.update(status='decomposed_not_accuracy_validated',original_execution=execution,
                      execution={k:dict(v.stats(),precision_policy=probe.policy(k)) for k,v in sessions.items()})
        for v in report['execution'].values():v.pop('precision',None)
    except Exception as e:
        report.update(status='failed',error=type(e).__name__);raise
    finally:
        for native in sessions.values():native.close()
        q.save(target,report)
    print(json.dumps({'status':report['status'],'cases':len(report['rows']),'acceptance_passed':False}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);a=p.parse_args();a.stage=a.stage.resolve()
    if platform.machine()!='aarch64' or not a.stage.is_relative_to(q.BASE/'staging') or not a.stage.is_dir():p.error('New isolated Jetson staging required')
    sys.path.insert(0,str(q.RELEASE/'edge_deploy/jetson'));sys.dont_write_bytecode=True
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE),AUTOOCR_RUNTIME_DIR=str(a.stage/'runtime'),AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL);run(a)


if __name__=='__main__':main()
