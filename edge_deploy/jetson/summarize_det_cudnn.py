"""E107 evidence summary, retaining failed gates and no automatic promotion."""
import argparse
import json
from pathlib import Path
import build_det_cudnn as build
import evaluate_det_cudnn as evaluation
import summarize_det_hotspot as prior_summary

q=build.q


def summarize(stage):
    b=json.loads((stage/'build.json').read_text())
    if any(b[k]!=v for k,v in build.provenance().items()):raise ValueError('Build provenance changed')
    result=dict(status='build_not_accepted',build_status=b['status'],build_s=b.get('build_s'),
                accepted_for_release=False,deployment_changed=False,business_pairs=0,benchmark_calls=0)
    if b['status']=='built_not_validated':
        if q.digest(stage/'det_cudnn.engine')!=b['engine_sha256']:raise ValueError('Engine changed')
        with evaluation.adapter():
            checked=evaluation.legacy.require(stage,'check.json','finite_same_inputs_compared_not_business_accepted')
            quality=json.loads((stage/'evaluation.json').read_text())
            meta=evaluation.metadata(stage)
            if any(quality[k]!=v for k,v in meta.items()):raise ValueError('Quality provenance changed')
        if len(checked['synthetic'])!=3:raise ValueError('Shape check incomplete')
        layers=json.loads((stage/'layers.json').read_text())
        if checked['inspection']!=evaluation.inspect(layers,b['wanted']):raise ValueError('Inspector summary differs')
        paths=[stage/('benchmark_%s_%s.json'%(v,r)) for v in ('reference','mixed') for r in ('r1','r2')]
        measured=[json.loads(p.read_text()) for p in paths if p.exists()]
        result.update(prior_summary.summarize(checked,quality,measured))
        result.update(numeric_cases=len(checked['rows']),shape_cases=len(checked['synthetic']),
                      numeric_allclose_pass_count=sum(v['allclose_diagnostic'] for v in checked['rows']),
                      probability_mae_range=[min(r['mae'] for r in checked['rows']),max(r['mae'] for r in checked['rows'])],
                      probability_max_abs_error=max(r['max_abs_error'] for r in checked['rows']))
    elif b['status'] in ('failed','no_engine_returned'):
        if any((stage/n).exists() for n in ('det_cudnn.engine','check.json','evaluation.json')):raise ValueError('Unexpected downstream artifacts after build failure')
        result.update(selection_failures=[r.get('selection_error') for r in b['selections'] if r.get('selection_error')],
                      note='Build failure is not a numerical/business/performance result. No precision relaxation or release change.')
    else:raise ValueError('Incomplete build')
    names=['build.json','build.log','ranges.json','check.json','layers.json','evaluation.json']
    names += ['benchmark_%s_%s.json'%(v,r) for v in ('reference','mixed') for r in ('r1','r2')]
    result.update(script_sha256=q.digest(__file__),reports_sha256={n:q.digest(stage/n) for n in names if (stage/n).exists()})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,type=Path);a=p.parse_args()
    result=summarize(a.stage);q.save(a.stage/'summary.json',result);print(json.dumps(result,indent=2))
