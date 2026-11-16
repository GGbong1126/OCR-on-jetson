"""Gated raw-det ABBA timing; not full OCR, HTTP, queues or GPU-kernel timing."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import det_batch2 as b


def cycle(single, candidate, data, groups):
    outputs = {}
    for indices in groups:
        if candidate is None:
            for index in indices:
                outputs[index] = single.run(None, {'x': data[index][1]})[0].copy()
        else:
            x = np.concatenate([data[index][1] for index in indices])
            y = b.route(single, candidate, x)
            for slot, index in enumerate(indices):
                outputs[index] = y[slot:slot+1].copy()
    if set(outputs) != set(range(len(data))):
        raise ValueError('Timing cycle lost/duplicated image mapping')
    return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--variant',choices=('single','routed'),required=True)
    p.add_argument('--round',choices=('r1','r2'),required=True)
    a=p.parse_args();stage=a.stage.resolve()
    if not stage.is_relative_to(b.q.BASE/'staging'):p.error('Isolated project stage required')
    state=subprocess.check_output(['systemctl','show','autoocr-jetson-dev.service','--property=ActiveState'],text=True)
    if state.strip()!='ActiveState=inactive':raise RuntimeError('Keep original service stopped')
    plan=b.load_plan(stage)
    b.cv2.setNumThreads(plan['cv2_threads'])
    b.require(stage,'check.json','gpu_det_boxes_consistent_not_full_business_validated')
    built=b.require(stage,'build.json','built_not_validated')
    target=stage/('benchmark_'+a.variant+'_'+a.round+'.json')
    if target.exists():raise FileExistsError(str(target))
    from trt82_backend import Session
    spec=json.loads((b.q.RELEASE/'edge_deploy/jetson/deployment.json').read_text())['engines']['det']
    data=b.inputs(plan,b.legacy_method())
    report=dict(status='running',variant=a.variant,round=a.round,
                script_sha256=b.q.digest(__file__),plan_sha256=b.q.digest(stage/'plan.json'),
                opencv_threads=b.cv2.getNumThreads(),numpy_version=np.__version__,
                gate_sha256=b.q.digest(stage/'check.json'),build_sha256=b.q.digest(stage/'build.json'),
                verification_cycles=1,warmup_cycles=plan['warmup'],measured_cycles=plan['repeat'],images_per_cycle=len(data),
                groups=plan['groups'],input_sha256=[b.tensor_sha(x) for _,x in data],
                memory_before=b.q.memory(),cycle_ms=[],
                note='Session.run + grouping/validation/demux copies, entire 13-image raw-det cycle; '
                     'excludes decode/pre/post and hashing. Separate processes, both sessions only for routed tails. '
                     'No clock changes, end-to-end OCR/HTTP speedup, GT or peak-memory claim. '
                     'Ten measured cycles; tail percentiles are exploratory.')
    single=candidate=None
    try:
        single=Session(b.q.RELEASE/spec['path'],spec['sha256'],'det_n1_benchmark')
        if a.variant=='routed':candidate=Session(stage/'det_batch2.engine',built['engine_sha256'],'det_n2_benchmark')
        call=lambda:cycle(single,candidate,data,plan['groups'])
        expected={i:b.tensor_sha(y) for i,y in call().items()}
        for _ in range(plan['warmup']):
            if {i:b.tensor_sha(y) for i,y in call().items()}!=expected:raise ValueError('Warmup drift')
        report['memory_after_warmup']=b.q.memory()
        for _ in range(plan['repeat']):
            start=time.perf_counter();out=call();elapsed=(time.perf_counter()-start)*1000
            if {i:b.tensor_sha(y) for i,y in out.items()}!=expected:raise ValueError('Measured output drift')
            report['cycle_ms'].append(elapsed)
        times=np.asarray(report['cycle_ms'])
        if not np.isfinite(times).all() or (times<=0).any():raise ValueError('Invalid timing')
        report.update(status='raw_det_benchmark_completed',mean_cycle_ms=float(times.mean()),
                      p50_cycle_ms=float(np.median(times)),images_per_second=float(len(data)*1000/times.mean()),
                      single=single.stats(),candidate=candidate.stats() if candidate else None,
                      memory_after=b.q.memory(),output_sha256=expected)
    except Exception as exc:
        report.update(status='failed',error=type(exc).__name__);raise
    finally:
        cleanup_errors=[]
        for session in (candidate,single):
            if session is not None:
                try:session.close()
                except Exception as exc:cleanup_errors.append(type(exc).__name__)
        if cleanup_errors:report.update(status='failed',cleanup_errors=cleanup_errors)
        gc.collect();report['memory_after_close']=b.q.memory()
        report['fingerprints_unchanged']=b.fingerprints()==plan['fingerprints']
        b.q.save(target,report)
    if report['status']!='raw_det_benchmark_completed':raise RuntimeError('Benchmark cleanup failed')
    print(json.dumps({k:report[k] for k in ('status','variant','round','mean_cycle_ms','images_per_second')}),flush=True)


if __name__=='__main__':main()
