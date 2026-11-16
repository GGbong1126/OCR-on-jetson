"""E106 same-feature alternate implementation check, not business acceptance."""
import argparse
import gc
import json
import logging
import os
from pathlib import Path
import platform
import sys

import numpy as np
import decompose_det_error as d
import probe_det_alternative as alternate

probe=d.probe
q=d.q
previous=d.previous


def run(a):
    import onnx
    from onnx import helper,numpy_helper
    import onnxruntime as ort
    from trt82_backend import Session
    target=a.stage/'alternative_check.json';probe.old.fresh(target)
    build=json.loads((a.stage/'alternative_build.json').read_text())
    baseline=json.loads((a.stage/'decomposition.json').read_text())
    if (build['status']!='built_not_executed' or build['script_sha256']!=q.digest(alternate.__file__)
            or build['decomposition_script_sha256']!=q.digest(d.__file__)
            or build['provenance']!=probe.provenance()
            or q.digest(a.stage/'alternative.engine')!=build['engine_sha256']
            or baseline['status']!='decomposed_not_accuracy_validated'
            or baseline['script_sha256']!=q.digest(d.__file__)):raise ValueError('Source/build/decomposition differs')
    builds,prior=d.prior_reports()
    report=dict(script_sha256=q.digest(__file__),build_sha256=q.digest(a.stage/'alternative_build.json'),
                baseline_sha256=q.digest(a.stage/'decomposition.json'),status='running',rows=[],synthetic=[],
                accepted_for_release=False,full_model_built=False,business_pairs=0,benchmark_calls=0)
    sessions={}
    try:
        raw,cfg,execution=previous.capture(a,3)
        keys=previous.describe(raw[0]).keys()
        if [previous.describe(v) for v in raw]!=[{k:v[k] for k in keys} for v in prior['rows']]:raise ValueError('Capture drift')
        model=onnx.load(str(previous.SOURCE));node=next(n for n in model.graph.node if n.name=='Conv.87')
        names=[node.input[0],node.output[0]];weights=d.round_half(probe.get_weights())
        single=helper.make_model(helper.make_graph([node],'rounded_conv87',
            [helper.make_tensor_value_info(names[0],onnx.TensorProto.FLOAT,[1,256,'h','w'])],
            [helper.make_tensor_value_info(names[1],onnx.TensorProto.FLOAT,[1,64,'h','w'])],
            [numpy_helper.from_array(weights,name=node.input[1])]),opset_imports=list(model.opset_import))
        single.ir_version=model.ir_version;onnx.checker.check_model(single)
        model.graph.output.extend(helper.make_tensor_value_info(n,onnx.TensorProto.FLOAT,None) for n in names)
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        cpu=ort.InferenceSession(model.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
        rounded=ort.InferenceSession(single.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
        del model,single;gc.collect()
        sessions['reference']=Session(d.PRIOR/'fp32.engine',builds['fp32']['engine_sha256'],'e106_reference')
        sessions['alternative']=Session(a.stage/'alternative.engine',build['engine_sha256'],'e106_alternative')
        for i,row in enumerate(raw):
            x,y=cpu.run(names,{'x':row['x']});old=prior['rows'][i];b=baseline['rows'][i]
            if previous.sha_tensor(x)!=old['feature_sha256'] or previous.sha_tensor(y)!=old['cpu_output_sha256']:raise ValueError('Feature/reference drift')
            both=rounded.run(None,{names[0]:d.round_half(x)})[0];round_y=d.round_half(both)
            if previous.sha_tensor(both)!=b['output_sha256']['both'] or previous.sha_tensor(round_y)!=b['output_sha256']['rounded_output']:raise ValueError('Rounded reference drift')
            ref=sessions['reference'].run(None,{'x':x})[0]
            if previous.sha_tensor(ref)!=old['variants']['fp32']['output_sha256']:raise ValueError('Reference engine drift')
            out=sessions['alternative'].run(None,{'x':x})[0];repeat=sessions['alternative'].run(None,{'x':x})[0]
            if not np.array_equal(out,repeat):raise ValueError('Alternative not repeatable')
            report['rows'].append(dict(previous.describe(row),feature_sha256=previous.sha_tensor(x),
                output_sha256=previous.sha_tensor(out),repeat_exact=True,
                vs_fp32=d.diagnostic.error_metrics(ref,out),decomposition=d.decompose(y,both,round_y,out)))
            print('Alternative case',i+1,flush=True)
        del cpu,rounded,raw,x,y,both,round_y,ref,out,repeat;gc.collect()
        for shape in probe.PROFILE:
            x=np.random.default_rng(105).uniform(-1,1,shape).astype(np.float32)
            out={k:n.run(None,{'x':x})[0] for k,n in sessions.items()}
            report['synthetic'].append(dict(shape=list(shape),input_sha256=previous.sha_tensor(x),
                vs_fp32=d.diagnostic.error_metrics(out['reference'],out['alternative'])))
        native=sessions['alternative'];inspector=native.engine.create_engine_inspector();inspector.execution_context=native.context
        info=json.loads(inspector.get_engine_information(native.trt.LayerInformationFormat.JSON))
        q.save(a.stage/'alternative_layers.json',info)
        report['inspection']=probe.inspect(info,'fp16_via_cast')
        selected=next(v for v in build['reported'] if 'Conv.87' in v['name'])['algorithm']
        if alternate.unsigned(int(report['inspection']['convolution']['TacticValue'],16))!=alternate.unsigned(selected['tactic']):raise ValueError('Inspector tactic differs from selector report')
        report.update(status='alternate_executed_not_business_validated',original_execution=execution,
            execution={k:v.stats() for k,v in sessions.items()},note='6 original real features plus synthetic min/opt/max. Numeric diagnostics only; no OCR/GT or timing benchmark. No release changes.')
        for key,value in report['execution'].items():
            value.pop('precision',None);value['precision_policy']=probe.policy('fp32' if key=='reference' else 'fp16_via_cast')
    except Exception as e:
        report.update(status='failed',error=type(e).__name__);raise
    finally:
        for native in sessions.values():native.close()
        q.save(target,report)
    print(json.dumps(dict(status=report['status'],cases=len(report['rows']),accepted_for_release=False)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args();a.stage=a.stage.resolve()
    if platform.machine()!='aarch64' or not a.stage.is_relative_to(q.BASE/'staging') or not a.stage.is_dir():p.error('Isolated Jetson stage required')
    sys.path.insert(0,str(q.RELEASE/'edge_deploy/jetson'));sys.dont_write_bytecode=True
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE),AUTOOCR_RUNTIME_DIR=str(a.stage/'runtime_alternative'),AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL);run(a)


if __name__=='__main__':main()
