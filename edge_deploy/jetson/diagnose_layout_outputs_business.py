"""Untimed E111 business diagnostic after raw numeric failure, not acceptance.

Preserves check.json/status and refuses benchmark authorization. Investigates
whether the raw detection-table comparison predicts actual downstream changes.
"""
import contextlib
import json
import os
from pathlib import Path

import evaluate_layout_outputs as e


def require_diagnostic(stage):
    checked=e.require(stage,'check.json','numeric_not_accepted')
    prior=json.loads((stage/'output_diagnostic.json').read_text())
    if (prior['status']!='diagnosed_not_acceptance' or len(prior['rows'])!=6
            or prior['execution']['calls']!=6 or prior['execution']['errors']
            or checked['execution']['calls']!=12 or checked['execution']['errors']
            or any(prior[k]!=v for k,v in e.metadata(stage).items())):
        raise ValueError('Incomplete diagnostic lineage')
    for row in checked['rows']:
        if not row['repeat_exact'] or not all(v['all_finite'] for v in row['raw'].values()):
            raise ValueError('Unstable/nonfinite candidate')
    return prior


def run(a):
    require_diagnostic(a.stage)
    target=a.stage/'business_diagnostic.json';e.old.fresh(target)
    report=dict(e.metadata(a.stage),status='running',rows=[],
        business_diagnostic_script_sha256=e.q.digest(__file__),
        output_diagnostic_sha256=e.q.digest(a.stage/'output_diagnostic.json'),
        accepted_for_release=False,benchmark_authorized=False,
        original_numeric_gate='numeric_not_accepted',
        note='Untimed downstream diagnostic, not a replacement/relaxation of original failed raw-output gate. No benchmark, GT, release or normalization. 13 existing images x both paths; original six other models/rules.')
    plan=e.q.load_plan(e.q.BASE/'staging/quant_yolo_e92_r1')
    models=trial=base=original=None
    try:
        with open(os.devnull,'w') as quiet:
            with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                server,pipeline,models,cfg=e.old.setup(a)
            base=models.models['layout'];original=base.run;trial=e.candidate(a.stage)
            active='reference';hashes=None
            def call(names,feeds):
                if names not in (None,[]) or trial.errors:raise ValueError('Invalid request/unhealthy candidate')
                hashes[active].append(e.source.input_metadata(feeds))
                return original(names,feeds) if active=='reference' else e.retained_outputs(trial,feeds)
            guarded=e.source.GuardedCall(call);base.run=guarded
            for image in plan['calibration']+plan['holdout']:
                for kind in (3,1):
                    values={};counts={};hashes={'reference':[],'candidate':[]}
                    order=('reference','candidate') if len(report['rows'])%2==0 else ('candidate','reference')
                    for active in order:
                        before=models.stats()
                        with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                            result=pipeline.process(str(Path(plan['images'])/image['file']),image_type=kind,run_ner=True)
                        guarded.require_healthy();after=models.stats()
                        if not result['success'] or not models.healthy or trial.errors:raise RuntimeError('Business/backend failure')
                        counts[active]={r:after[r]['cpu_fallback']['calls']-before[r]['cpu_fallback']['calls'] for r in ('det','rec')}
                        values[active]=result['data']
                    if len(hashes['reference'])!=1 or hashes['reference']!=hashes['candidate']:
                        raise ValueError('Candidate bypass or input drift')
                    ref,out=values['reference'],values['candidate'];row=e.q.strict_compare(ref,out)
                    row.update(image_sha256=image['sha256'],image_type=kind,order=order,
                        same_original_input=True,inputs=hashes['reference'][0],cpu_fallback_calls=counts,
                        fields=e.q.field_diagnostics(ref,out,server._layout_special_text),
                        number_symbol_sequence_equal=e.shared.protected_equal(ref['FullText'],out['FullText']),
                        strict_objects_equal=row['passed'])
                    row['business_consistent']=e.shared.business_passed(row)
                    del row['passed']
                    report['rows'].append(row)
                    print('Business diagnostic',len(report['rows']),row['business_consistent'],flush=True)
            report.update(status='diagnostic_completed_not_accepted',
                business_consistent_count=sum(r['business_consistent'] for r in report['rows']))
    except Exception as err:
        report.update(status='failed',error=type(err).__name__);raise
    finally:
        if original is not None:base.run=original
        if trial is not None:report['candidate_execution']=trial.stats();trial.close()
        if models is not None:report['original_execution']=models.stats()
        e.old.close_models(models);e.q.save(target,report)
    print(json.dumps(dict(status=report['status'],business_consistent_count=report['business_consistent_count'],benchmark_authorized=False)),flush=True)


if __name__=='__main__':
    a=e.source.arguments(('diagnose',));run(a)
