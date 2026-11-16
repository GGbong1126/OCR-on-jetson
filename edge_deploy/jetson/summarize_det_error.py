"""Summarize E106 immutable numerical evidence without accepting a release."""
import argparse
import json
from pathlib import Path
import decompose_det_error as d
import probe_det_alternative as alternate
import evaluate_det_alternative as evaluate


def ranges(rows,key):
    return {metric:[min(r[key][metric] for r in rows),max(r[key][metric] for r in rows)]
            for metric in ('mae','max_abs_error')}


def summarize(stage):
    q=d.q
    load=lambda name:json.loads((stage/name).read_text())
    base=load('decomposition.json');build=load('alternative_build.json');check=load('alternative_check.json')
    if (base['status']!='decomposed_not_accuracy_validated' or base['script_sha256']!=q.digest(d.__file__)
            or build['status']!='built_not_executed' or build['script_sha256']!=q.digest(alternate.__file__)
            or check['status']!='alternate_executed_not_business_validated' or check['script_sha256']!=q.digest(evaluate.__file__)
            or check['build_sha256']!=q.digest(stage/'alternative_build.json')
            or check['baseline_sha256']!=q.digest(stage/'decomposition.json')
            or q.digest(stage/'alternative.engine')!=build['engine_sha256']):raise ValueError('Evidence drift or incomplete check')
    if len(base['rows'])!=6 or len(check['rows'])!=6 or len(check['synthetic'])!=3:raise ValueError('Incomplete inputs')
    for b,c in zip(base['rows'],check['rows']):
        for key in ('image_sha256','image_type','input_sha256','reference_sha256','feature_sha256'):
            if b[key]!=c[key]:raise ValueError('Source mismatch')
        if not b['original_cpu_exact'] or not b['previous_gpu_outputs_exact'] or not c['repeat_exact']:raise ValueError('Reference drift')
        if b['decomposition']['signed_closure_max_abs']!=0 or c['decomposition']['signed_closure_max_abs']!=0:raise ValueError('Closure mismatch')
    if any(v['errors'] for v in check['execution'].values()):raise ValueError('Execution error')
    return dict(status='decomposed_alternate_evaluated_not_business_accepted',
        accepted_for_release=False,full_detector_built=False,business_pairs=0,benchmark_calls=0,
        source_cases=6,synthetic_cases=3,
        original_implementation={key:ranges([v['decomposition'] for v in base['rows']],key)
            for key in ('operand_rounding','output_rounding','implementation_residual','total_gpu_vs_cpu')},
        alternative_vs_gpu_fp32=ranges(check['rows'],'vs_fp32'),
        alternative_implementation_residual=ranges([v['decomposition'] for v in check['rows']],'implementation_residual'),
        alternative_allclose_pass_count=sum(v['vs_fp32']['allclose_diagnostic'] for v in check['rows']),
        alternative_outside_allclose_count=sum(v['vs_fp32']['outside_allclose_count'] for v in check['rows']),
        elements=sum(v['vs_fp32']['elements'] for v in check['rows']),
        decomposition_sha256=q.digest(stage/'decomposition.json'),alternative_check_sha256=q.digest(stage/'alternative_check.json'),
        alternative_build_sha256=q.digest(stage/'alternative_build.json'),
        fp64_rounded_operand_sample_count=sum(v['rounded_fp64_spots']['sample_count'] for v in base['rows']),
        fp64_rounded_operand_cpu_max_error=max(v['rounded_fp64_spots']['cpu_max_abs_error'] for v in base['rows']),
        fp64_rounded_operand_original_gpu_max_error=max(v['rounded_fp64_spots']['gpu_max_abs_error'] for v in base['rows']),
        note='Residual is not proof of accumulator dtype. Magnitudes do not sum. FP64 spots use half-rounded operands represented as FP32 (legacy helper note says original FP32). Builder timing is not inference performance. No full-model OCR/GT or release claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,type=Path);a=p.parse_args()
    result=summarize(a.stage);d.q.save(a.stage/'summary.json',result)
    print(json.dumps(result,indent=2))
