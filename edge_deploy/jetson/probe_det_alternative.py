"""E106 one alternate Conv.87 implementation, isolated and never released.

Exclude the implementation identified by the prior executed tactic. Select the
fastest measured available *different implementation*, not a guessed precision
from a tactic name. Empty/ambiguous selection fails closed.
"""
import argparse
import json
from pathlib import Path
import platform
import re
import time

import probe_det_precision_boundary as probe
import decompose_det_error as decomposition

q=probe.q


def unsigned(value):
    return int(value)&((1<<64)-1)


def choose(rows,prior_tactic):
    matching={r['implementation'] for r in rows if unsigned(r['tactic'])==unsigned(prior_tactic)}
    if len(matching)!=1:raise ValueError('Prior tactic does not identify one implementation')
    old=next(iter(matching))
    eligible=[r for r in rows if r['implementation']!=old and r['timing_ms']>=0]
    if not eligible:raise ValueError('No different implementation available')
    selected=min(eligible,key=lambda r:(r['timing_ms'],r['workspace_bytes'],r['implementation'],r['tactic'],r['index']))
    return old,selected['index']


def algorithm(context,choice,index):
    return dict(index=index,implementation=int(choice.algorithm_variant.implementation),
                tactic=int(choice.algorithm_variant.tactic),timing_ms=float(choice.timing_msec),
                workspace_bytes=int(choice.workspace_size),
                io=[dict(dtype=str(choice.get_algorithm_io_info(i).dtype),
                         format=str(choice.get_algorithm_io_info(i).tensor_format),
                         strides=list(choice.get_algorithm_io_info(i).strides))
                    for i in range(context.num_inputs+context.num_outputs)])


def build(a):
    import tensorrt as trt
    path=a.stage/'alternative_build.json';engine=a.stage/'alternative.engine'
    probe.old.fresh(path);probe.old.fresh(engine)
    builds,prior=decomposition.prior_reports()
    conv=prior['inspection']['fp16_via_cast']['convolution']
    prior_tactic=int(conv['TacticValue'],16)
    report=dict(status='building',script_sha256=q.digest(__file__),
                decomposition_script_sha256=q.digest(decomposition.__file__),
                provenance=probe.provenance(),prior_sha256=q.digest(decomposition.PRIOR/'diagnostic.json'),
                prior_tactic=prior_tactic,profile=probe.PROFILE,policy=probe.policy('fp16_via_cast'),
                workspace_bytes=256*1024**2,avg_timing_iterations=8,selections=[],reported=[],
                note='Single alternate implementation experiment. Selector timing is builder data, NOT an inference benchmark. No assumed accumulator dtype, no full-model/quality/release acceptance.')
    try:
        if not trt.__version__.startswith('8.2.1'):raise ValueError('TRT8.2.1 required')
        class Selector(trt.IAlgorithmSelector):
            def __init__(self):trt.IAlgorithmSelector.__init__(self)
            def select_algorithms(self,context,choices):
                rows=[algorithm(context,c,i) for i,c in enumerate(choices)]
                entry=dict(name=context.name,choices=rows)
                report['selections'].append(entry)
                if re.search(r'(?<![\w.])Conv\.87(?![\w.])',context.name):
                    try:
                        old,index=choose(rows,prior_tactic)
                        entry.update(excluded_implementation=old,selected_indices=[index])
                        return [index]
                    except ValueError as e:
                        entry.update(selection_error=str(e),selected_indices=[])
                        return []
                entry['selected_indices']=list(range(len(choices)))
                return entry['selected_indices']
            def report_algorithms(self,contexts,choices):
                report['reported'].extend(dict(name=c.name,algorithm=algorithm(c,v,0)) for c,v in zip(contexts,choices))
        weights=probe.get_weights()
        logger=trt.Logger(trt.Logger.INFO);builder=trt.Builder(logger)
        network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        x=network.add_input('x',trt.float32,(-1,256,-1,-1))
        layer=probe.add_hotspot(network,x,weights,trt)
        layer.precision=trt.float16;layer.set_output_type(0,trt.float16)
        identity=network.add_identity(layer.get_output(0));identity.name='restore_fp32'
        identity.set_output_type(0,trt.float32)
        y=identity.get_output(0);y.name='y';network.mark_output(y)
        for tensor in (x,y):tensor.dtype=trt.float32;tensor.allowed_formats=1<<int(trt.TensorFormat.LINEAR)
        config=builder.create_builder_config();config.max_workspace_size=report['workspace_bytes']
        config.avg_timing_iterations=8;config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        config.clear_flag(trt.BuilderFlag.TF32);config.clear_flag(trt.BuilderFlag.INT8)
        for flag in ('FP16','OBEY_PRECISION_CONSTRAINTS','REJECT_EMPTY_ALGORITHMS','DISABLE_TIMING_CACHE'):
            config.set_flag(getattr(trt.BuilderFlag,flag))
        profile=builder.create_optimization_profile();profile.set_shape('x',*probe.PROFILE)
        if [tuple(v) for v in profile.get_shape('x')]!=probe.PROFILE:raise ValueError('Profile differs')
        config.add_optimization_profile(profile)
        selector=Selector();config.algorithm_selector=selector
        report.update(trt_version=trt.__version__,flags=int(config.flags),tactic_sources=config.get_tactic_sources())
        start=time.perf_counter();blob=builder.build_serialized_network(network,config)
        report['build_s']=time.perf_counter()-start
        selected=[r for r in report['selections'] if 'excluded_implementation' in r]
        executed=[r for r in report['reported'] if re.search(r'(?<![\w.])Conv\.87(?![\w.])',r['name'])]
        if blob is None:report['status']='no_engine_returned'
        else:
            if len(selected)!=1 or len(executed)!=1:raise ValueError('Unique selection/report missing')
            choice=selected[0]['choices'][selected[0]['selected_indices'][0]]
            actual=executed[0]['algorithm']
            if (actual['implementation'],actual['tactic'],actual['io'])!=(choice['implementation'],choice['tactic'],choice['io']):
                raise ValueError('Reported choice differs')
            if actual['implementation']==selected[0]['excluded_implementation']:raise ValueError('Excluded implementation returned')
            with engine.open('xb') as f:f.write(bytes(blob))
            report.update(status='built_not_executed',engine_sha256=q.digest(engine),engine_bytes=engine.stat().st_size)
    except Exception as e:
        report.update(status='failed',error=type(e).__name__);raise
    finally:q.save(path,report)
    print(json.dumps({k:report.get(k) for k in ('status','build_s','engine_bytes')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args();a.stage=a.stage.resolve()
    if platform.machine()!='aarch64' or not a.stage.is_relative_to(q.BASE/'staging') or not a.stage.is_dir():p.error('Isolated Jetson stage required')
    build(a)


if __name__=='__main__':main()
