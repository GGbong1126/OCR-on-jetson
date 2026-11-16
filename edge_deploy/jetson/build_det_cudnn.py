"""E107 full detector: only Conv.87 half, restore both consumers to float.

Replays the E106 implementation/tactic only on the same TRT version and device.
No relaxed precision constraints, release writes, HTTP, or model export.
"""
import argparse
import json
from pathlib import Path
import platform
import re
import sys
import time

import det_hotspot_mixed as original
import probe_det_alternative as micro

q=original.q
old=original.old
PRIOR=q.BASE/'staging/det_error_e106_r1'


def signature(row):
    return (row['implementation'],micro.unsigned(row['tactic']),
            [(r['dtype'],r['format']) for r in row['io']])


def choose(rows,wanted):
    matches=[r['index'] for r in rows if signature(r)==signature(wanted)]
    if len(matches)!=1:raise ValueError('One previously tested implementation/tactic/IO required')
    return matches


def provenance():
    p=original.provenance()
    b=json.loads((PRIOR/'alternative_build.json').read_text())
    c=json.loads((PRIOR/'alternative_check.json').read_text())
    if (b['status']!='built_not_executed' or c['status']!='alternate_executed_not_business_validated'
            or c['build_sha256']!=q.digest(PRIOR/'alternative_build.json')
            or b['script_sha256']!=q.digest(micro.__file__)
            or b['provenance']!=micro.probe.provenance()
            or q.digest(PRIOR/'alternative.engine')!=b['engine_sha256']):raise ValueError('E106 evidence changed')
    chosen=[v['algorithm'] for v in b['reported'] if v['name']=='Conv.87']
    if len(chosen)!=1 or len(c['rows'])!=6 or c['execution']['alternative']['errors']:raise ValueError('Incomplete prior evidence')
    if micro.unsigned(int(c['inspection']['convolution']['TacticValue'],16))!=micro.unsigned(chosen[0]['tactic']):raise ValueError('Prior executed tactic differs')
    return dict(script_sha256=q.digest(__file__),source_sha256=p['source_sha256'],deployment_sha256=p['deployment_sha256'],
                plan_sha256=p['plan_sha256'],original_helper_sha256=q.digest(original.__file__),
                micro_helper_sha256=q.digest(micro.__file__),micro_build_sha256=q.digest(PRIOR/'alternative_build.json'),
                micro_check_sha256=q.digest(PRIOR/'alternative_check.json'),wanted=chosen[0],trt_version=b['trt_version'])


def rewire(network,trt):
    layers=[network.get_layer(i) for i in range(network.num_layers)]
    found=[l for l in layers if l.name=='Conv.87']
    if len(found)!=1 or str(found[0].type).split('.')[-1]!='CONVOLUTION':raise ValueError('Unique original hotspot missing')
    conv=found[0];source=conv.get_output(0)
    if source.name!='p2o.pd_op.conv2d.60.0':raise ValueError('Unexpected hotspot tensor')
    consumers=[(l,j) for l in layers for j in range(l.num_inputs)
               if l.get_input(j) is not None and l.get_input(j).name==source.name]
    if sorted((l.name,j) for l,j in consumers)!=[('Conv.88',0),('Conv.91',0)]:raise ValueError('Consumer contract changed')
    restore=network.add_identity(source)
    if restore is None:raise RuntimeError('Could not add FP32 boundary')
    restore.name='Conv87_restore_fp32';restore.set_output_type(0,trt.float32)
    former=source.name;source.name=former+'__fp16'
    y=restore.get_output(0);y.name=former
    for layer,j in consumers:layer.set_input(j,y)
    for layer,j in consumers:
        if layer.get_input(j).name!=former:raise ValueError('Consumer not rewired')
    return [dict(name=l.name,input_index=j) for l,j in consumers]


def build(a):
    import tensorrt as trt
    target=a.stage/'build.json';engine=a.stage/'det_cudnn.engine'
    old.fresh(target);old.fresh(engine)
    report=dict(provenance(),status='building',constraints=[],selections=[],reported=[],accepted_for_release=False)
    started=None
    try:
        if trt.__version__!=report['trt_version']:raise ValueError('Exact TRT version differs')
        class Selector(trt.IAlgorithmSelector):
            def __init__(self):trt.IAlgorithmSelector.__init__(self)
            def select_algorithms(self,context,choices):
                if re.search(r'(?<![\w.])Conv\.87(?![\w.])',context.name):
                    rows=[micro.algorithm(context,c,i) for i,c in enumerate(choices)]
                    entry=dict(name=context.name,choices=rows);report['selections'].append(entry)
                    try:
                        if context.name!='Conv.87':raise ValueError('Unexpected fused hotspot context')
                        result=choose(rows,report['wanted']);entry['selected_indices']=result
                        return result
                    except ValueError as e:
                        entry.update(selection_error=str(e),selected_indices=[]);return []
                return list(range(len(choices)))
            def report_algorithms(self,contexts,choices):
                report['reported'].extend(dict(name=c.name,algorithm=micro.algorithm(c,v,0)) for c,v in zip(contexts,choices))
        logger=trt.Logger(trt.Logger.INFO);builder=trt.Builder(logger)
        net=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser=trt.OnnxParser(net,logger)
        if not parser.parse(original.SOURCE.read_bytes()):raise RuntimeError('Original ONNX parse failed')
        if net.num_inputs!=1 or net.get_input(0).name!='x' or net.num_outputs!=1:raise ValueError('I/O differs')
        report['parsed_layers']=net.num_layers;report['rewired_consumers']=rewire(net,trt)
        for i in range(net.num_layers):
            layer=net.get_layer(i);kind=str(layer.type).split('.')[-1]
            outputs=[j for j in range(layer.num_outputs) if layer.get_output(j) is not None
                     and layer.get_output(j).dtype==trt.float32 and not layer.get_output(j).is_shape_tensor]
            original.layer_policy(layer.name,kind,outputs)
            if not outputs:continue
            half=layer.name=='Conv.87'
            if kind in original.COMPUTE:layer.precision=trt.float16 if half else trt.float32
            for j in outputs:layer.set_output_type(j,trt.float16 if half else trt.float32)
            report['constraints'].append(dict(name=layer.name,kind=kind,output_indices=outputs,policy='half_compute_half_output' if half else 'fp32'))
        config=builder.create_builder_config();config.max_workspace_size=256*1024**2
        config.avg_timing_iterations=8;config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        config.clear_flag(trt.BuilderFlag.TF32);config.clear_flag(trt.BuilderFlag.INT8)
        for flag in ('FP16','OBEY_PRECISION_CONSTRAINTS','REJECT_EMPTY_ALGORITHMS','DISABLE_TIMING_CACHE'):config.set_flag(getattr(trt.BuilderFlag,flag))
        profile=builder.create_optimization_profile();profile.set_shape('x',*original.PROFILE)
        if [tuple(v) for v in profile.get_shape('x')]!=original.PROFILE:raise ValueError('Profile mismatch')
        config.add_optimization_profile(profile);selector=Selector();config.algorithm_selector=selector
        for t in (net.get_input(0),net.get_output(0)):t.dtype=trt.float32;t.allowed_formats=1<<int(trt.TensorFormat.LINEAR)
        report.update(profile=original.PROFILE,workspace_bytes=config.max_workspace_size,avg_timing_iterations=8,
            flags=int(config.flags),tactic_sources=config.get_tactic_sources(),network_layers=net.num_layers,
            note='Only Conv.87 half compute/output. Both Conv.88/91 consume explicit restored FP32. Remaining float compute/output FP32. Exact E106 tactic/implementation/IO replay on same runtime; no fallback if unavailable. Full-model Inspector and business required. No INT8/pruning/input changes.')
        started=time.perf_counter();blob=builder.build_serialized_network(net,config)
        report['build_s']=time.perf_counter()-started
        if blob is None:report['status']='no_engine_returned'
        else:
            actual=[r['algorithm'] for r in report['reported'] if r['name']=='Conv.87']
            if len(actual)!=1 or signature(actual[0])!=signature(report['wanted']):raise ValueError('Actual build choice mismatch')
            with engine.open('xb') as f:f.write(bytes(blob))
            report.update(status='built_not_validated',engine_sha256=q.digest(engine),engine_bytes=engine.stat().st_size)
    except Exception as e:
        report.update(status='failed',error=type(e).__name__);raise
    finally:
        if started is not None:report.setdefault('build_s',time.perf_counter()-started)
        q.save(target,report)
    print(json.dumps({k:report.get(k) for k in ('status','build_s','engine_bytes')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,type=Path);a=p.parse_args();a.stage=a.stage.resolve()
    if platform.machine()!='aarch64' or not a.stage.is_relative_to(q.BASE/'staging') or not a.stage.is_dir():p.error('Isolated Jetson stage required')
    sys.dont_write_bytecode=True;build(a)
