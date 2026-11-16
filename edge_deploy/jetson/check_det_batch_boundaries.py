"""Synthetic min/opt/max execution and safe out-of-profile rejection checks."""
import argparse
import json
from pathlib import Path

import numpy as np
import det_batch2 as b


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args();stage=a.stage.resolve()
    if not stage.is_relative_to(b.q.BASE/'staging'):p.error('Isolated project stage required')
    plan=b.load_plan(stage);built=b.require(stage,'build.json','built_not_validated')
    target=stage/'boundaries.json'
    if target.exists():raise FileExistsError(str(target))
    from trt82_backend import Session
    cfg=json.loads((b.q.RELEASE/'edge_deploy/jetson/deployment.json').read_text())['engines']['det']
    report=dict(status='running',script_sha256=b.q.digest(__file__),
                plan_sha256=b.q.digest(stage/'plan.json'),build_sha256=b.q.digest(stage/'build.json'),
                memory_before=b.q.memory(),rows=[],
                note='Synthetic zeros/ones, min/opt/max; finite/shape/repeat/swap checks. '
                     'Numeric N1 comparison diagnostic, not real-image/text/GT or benchmark. '
                     'Invalid profiles queried only, not executed or treated as GPU errors.')
    single=batched=None
    try:
        single=Session(b.q.RELEASE/cfg['path'],cfg['sha256'],'det_n1_boundary')
        batched=Session(stage/'det_batch2.engine',built['engine_sha256'],'det_n2_boundary')
        if batched.profiles!={'x':[list(v) for v in b.PROFILE]}:raise ValueError('Actual engine profile differs')
        for shape in b.PROFILE:
            x=np.zeros(shape,np.float32);x[1]=1
            y=b.route(single,batched,x);repeat=b.route(single,batched,x)
            swap=b.route(single,batched,x[::-1].copy())[::-1].copy()
            reference=np.concatenate([single.run(None,{'x':x[i:i+1]})[0] for i in range(2)])
            order=b.compare(y,swap)
            if not np.array_equal(y,repeat) or not order['allclose']:raise ValueError('Boundary slot/repeat failure')
            report['rows'].append(dict(shape=list(shape),output_shape=list(y.shape),repeat_exact=True,
                                       input_sha256=b.tensor_sha(x),output_sha256=b.tensor_sha(y),
                                       swap=order,numeric=b.compare(reference,y)))
        rejects=[]
        for shape in ((1,3,32,32),(3,3,32,32),(2,3,16,32),(2,3,992,32)):
            accepted=batched.in_profile({'x':np.empty(shape,np.float32)})
            if accepted:raise ValueError('Unexpected out-of-profile acceptance')
            rejects.append(dict(shape=list(shape),accepted=False,executed=False))
        if single.errors or batched.errors:raise RuntimeError('Unexpected GPU errors')
        report.update(status='det_profile_boundaries_executed',rejections=rejects,
                      single=single.stats(),batched=batched.stats(),memory_after=b.q.memory())
    except Exception as exc:
        report.update(status='failed',error=type(exc).__name__);raise
    finally:
        errors=[]
        for session in (batched,single):
            if session is not None:
                try:session.close()
                except Exception as exc:errors.append(type(exc).__name__)
        if errors:report.update(status='failed',cleanup_errors=errors)
        report['fingerprints_unchanged']=b.fingerprints()==plan['fingerprints']
        b.q.save(target,report)
    if report['status']!='det_profile_boundaries_executed':raise RuntimeError('Cleanup failed')
    print(json.dumps(dict(status=report['status'],boundaries=len(report['rows']),rejections=len(rejects))),flush=True)


if __name__=='__main__':main()
