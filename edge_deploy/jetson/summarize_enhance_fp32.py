"""E108 strict evidence gates. No automatic release promotion."""
import argparse
import json
from pathlib import Path
import numpy as np
import evaluate_enhance_fp32 as e

META=('script_sha256','build_script_sha256','build_sha256','engine_sha256','baseline_engine_sha256',
      'onnx_sha256','deployment_sha256','plan_sha256','helpers_sha256','diagnostic_helper_sha256')


def summarize(check, quality, reports):
    if (check['status']!='finite_original_tiles_not_business_accepted' or len(check['rows'])!=27
            or check['execution']['errors'] or check['execution']['calls']!=54
            or not all(r['repeat_exact'] for r in check['rows'])):
        raise ValueError('Incomplete numeric gate')
    rows=quality['rows']
    if (quality['status'] not in ('all_cases_consistent_not_gt','differences_not_accepted')
            or len(rows)!=14 or sum(r['image_type']==3 for r in rows)!=1
            or len({r['image_sha256'] for r in rows if r['image_type']==1})!=13
            or not all(r['same_original_inputs'] for r in rows)
            or any(quality[k]!=check[k] for k in META) or quality['candidate_execution']['errors']):
        raise ValueError('Incomplete or mismatched business gate')
    for r in rows:
        if r['passed']!=e.legacy.business_passed(r['strict_equal'],r['field_diagnostics']):
            raise ValueError('Stored business result differs')
        if (r['image_type']==3 and r['tile_calls']!=0) or (r['image_type']==1 and r['tile_calls']<=0):
            raise ValueError('Missing execution or invalid negative control')
        if r['tile_calls']!=len(r['tile_input_sha256']): raise ValueError('Tile evidence incomplete')
    if (quality['status']=='all_cases_consistent_not_gt')!=e.legacy.quality_ok(rows):
        raise ValueError('Stored quality status differs')
    fallback={v:{role:sum(r['cpu_fallback_calls'][v][role] for r in rows) for role in ('det','rec')}
              for v in ('reference','candidate')}
    result=dict(status='quality_or_fallback_not_accepted',accepted_for_release=False,deployment_changed=False,
        business_cases=14,enhanced_cases=13,passed=sum(r['passed'] for r in rows),
        differences={k:sum(not r['strict_equal'][k] for r in rows) for k in ('FullText','Text','MarkdownBodyLines','NER','raw_line_texts','special')},
        number_symbol_changed=sum(not r['full_text_diagnostics']['number_symbol_sequence_equal'] for r in rows),
        special_text_changes={k:sum(not r['field_diagnostics']['api_special_text_equal'][k] for r in rows) for k in ('title','category_no','page_no')},
        geometric_changes=sum(not r['boxes_exactly_equal'] for r in rows),cpu_fallback_calls=fallback,
        benchmark_calls=0,rounds=[],note='Existing images regression, not GT. No text normalization. Snapshot memory is not peak; no whole-page or HTTP speed claim.')
    if not e.quality_ready(rows):
        if reports: raise ValueError('Rejected quality must not be benchmarked')
        return result
    if len(reports)!=4 or {(r['variant'],r['round']) for r in reports}!={(v,r) for v in ('reference','candidate') for r in ('r1','r2')}:
        raise ValueError('Four unique timing reports required')
    for r in reports:
        if (any(r[k]!=check[k] for k in META) or r['status']!='measured' or not r['pre_post_outputs_exact']
                or r['warmup']!=3 or r['repeat']!=20 or r['execution']['calls']!=216 or r['execution']['errors']
                or r['inputs']!=[e.describe(v) for v in check['rows'][:9]]):
            raise ValueError('Benchmark contract/provenance differs')
        if [(v['cycle'],v['index']) for v in r['rows']]!=[(c,i) for c in range(20) for i in range(9)]:
            raise ValueError('Missing/reordered samples')
        values=[v['wall_ms'] for v in r['rows']]
        if not np.isfinite(values).all() or min(values)<=0: raise ValueError('Invalid times')
        if any(not np.isclose(r['timing'][k],v,rtol=1e-9,atol=1e-9) for k,v in e.common.summary(values).items()):
            raise ValueError('Stored statistics differ')
    lookup={(r['variant'],r['round']):r for r in reports}
    for rnd in ('r1','r2'):
        ref,out=(lookup[v,rnd] for v in ('reference','candidate'))
        delta=lambda r:(r['after_warmup']['memory']['NvMapMemUsed_kib']-r['before']['memory']['NvMapMemUsed_kib'])/1024
        rm,cm=delta(ref),delta(out)
        mean=100*(1-out['timing']['mean_ms']/ref['timing']['mean_ms'])
        p95=100*(out['timing']['p95_ms']/ref['timing']['p95_ms']-1)
        result['rounds'].append(dict(round=rnd,reference=ref['timing'],candidate=out['timing'],
            mean_reduction_percent=mean,p95_increase_percent=p95,reference_nvmap_delta_mib=rm,
            candidate_nvmap_delta_mib=cm,extra_nvmap_mib=cm-rm,
            performance_gate=mean>=5 and p95<=5,memory_gate=min(rm,cm)>=0 and cm-rm<=256))
    result.update(benchmark_calls=720,no_extra_cpu_fallback=True,
        status='eligible_for_full_page_performance_test' if all(r['performance_gate'] and r['memory_gate'] for r in result['rounds']) else 'performance_or_resource_not_accepted')
    result['note']+=' 180 timed samples/configuration; tails exploratory. Engineering screen, not statistical significance or production acceptance.'
    return result


def from_stage(stage):
    b=json.loads((stage/'build.json').read_text())
    if any(b[k]!=v for k,v in e.build.provenance().items()): raise ValueError('Build source changed')
    if b['status']=='failed':
        if any((stage/n).exists() for n in ('enhance_fp32.engine','check.json','evaluation.json')):
            raise ValueError('Unexpected downstream artifact')
        result=dict(status='build_not_accepted',accepted_for_release=False,deployment_changed=False,benchmark_calls=0)
    else:
        if e.q.digest(stage/'enhance_fp32.engine')!=b['engine_sha256']: raise ValueError('Engine changed')
        checked=e.require(stage,'check.json','finite_original_tiles_not_business_accepted')
        quality=json.loads((stage/'evaluation.json').read_text())
        if any(quality[k]!=v for k,v in e.metadata(stage).items()): raise ValueError('Evaluation source changed')
        if checked['inspection']!=e.inspect(json.loads((stage/'layers.json').read_text())): raise ValueError('Inspection differs')
        paths=[stage/('benchmark_%s_%s.json'%(v,r)) for v in ('reference','candidate') for r in ('r1','r2')]
        result=summarize(checked,quality,[json.loads(p.read_text()) for p in paths if p.exists()])
        result.update(numeric_tiles=27,allclose_passed=sum(r['allclose_diagnostic'] for r in checked['rows']),
                      max_abs_error=max(r['max_abs_error'] for r in checked['rows']))
    names=['build.json','build.log','check.json','layers.json','evaluation.json']
    names+=['benchmark_%s_%s.json'%(v,r) for v in ('reference','candidate') for r in ('r1','r2')]
    result.update(build_s=b['build_s'],script_sha256=e.q.digest(__file__),
                  reports_sha256={n:e.q.digest(stage/n) for n in names if (stage/n).exists()})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--stage',required=True,type=Path); a=p.parse_args()
    result=from_stage(a.stage); e.q.save(a.stage/'summary.json',result); print(json.dumps(result,indent=2))
