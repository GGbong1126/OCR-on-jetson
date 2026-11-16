"""Recompute E110 quality/timing decisions from technical reports only."""
import argparse
import json
from pathlib import Path

import evaluate_layout_hotspot as e

META=('script_sha256','build_script_sha256','build_sha256','source_sha256','plugin_sha256',
      'deployment_sha256','plan_sha256','business_helper_sha256','engine_sha256')


def summarize(check, quality, benchmarks):
    if (check['status']!='finite_same_inputs_compared_not_business_accepted' or len(check['rows'])!=6
            or check['execution']['errors'] or check['execution']['calls']!=12
            or not all(r['repeat_exact'] and all(v['all_finite'] for v in r['raw'].values()) for r in check['rows'])):
        raise ValueError('Invalid numeric/format gate')
    if (any(quality[k]!=check[k] for k in META) or len(quality['rows'])!=26
            or quality['candidate_execution']['errors'] or quality['candidate_execution']['calls']!=26):
        raise ValueError('Incomplete paired business report')
    cases=[(r['image_sha256'],r['image_type']) for r in quality['rows']]
    if len(set(cases))!=26 or {k for _,k in cases}!={1,3} or any(sum(k==v for _,k in cases)!=13 for v in (1,3)):
        raise ValueError('Expected 13 images x 2 paths')
    images={h for h,_ in cases}
    if len(images)!=13 or any((h,1) not in cases or (h,3) not in cases for h in images): raise ValueError('Unpaired image set')
    passed=[e.business_passed(r) for r in quality['rows']]
    if passed!=[r['passed'] for r in quality['rows']]: raise ValueError('Stored quality decision differs')
    expected='all_26_business_consistent_not_gt' if all(passed) else 'differences_not_accepted'
    if quality['status']!=expected: raise ValueError('Stored quality status differs')
    result=dict(accepted_for_release=False,numeric_cases=6,business_cases=26,business_passed=sum(passed),
        full_text_changed=sum(not r['strict_equal']['FullText'] for r in quality['rows']),
        number_symbol_changed=sum(not r['number_symbol_sequence_equal'] for r in quality['rows']),
        ner_changed=sum(not r['strict_equal']['NER'] for r in quality['rows']),
        grouping_changed=sum(not r['strict_equal']['Text'] for r in quality['rows']),
        boxes_exactly_equal=sum(r['boxes_exactly_equal'] for r in quality['rows']),
        special_text_changed={k:sum(not r['fields']['api_special_text_equal'][k] for r in quality['rows']) for k in ('title','category_no','page_no')},
        failed_cases=[dict(image_sha256=r['image_sha256'],image_type=r['image_type']) for r in quality['rows'] if not r['passed']],
        benchmark_calls=0,rounds=[],
        note='Original baseline consistency, not GT accuracy. Same-GPU timings only if complete strict quality passed; no auto deployment.')
    if not all(passed):
        if benchmarks: raise ValueError('Performance not authorized by failed quality gate')
        result['status']='quality_not_accepted'; return result
    if not benchmarks:
        result['status']='quality_passed_performance_pending'; return result
    by={(r['variant'],r['round']):r for r in benchmarks}
    if len(benchmarks)!=4 or set(by)!={(v,r) for v in ('reference','candidate') for r in ('r1','r2')}:
        raise ValueError('Four independent timing processes required')
    gate=[]
    for r in benchmarks:
        if (any(r[k]!=check[k] for k in META) or r['status']!='measured' or r['warmup']!=3 or r['repeat']!=10
                or r['execution']['errors'] or r['execution']['calls']!=84 or not r['pre_post_outputs_exact']):
            raise ValueError('Invalid timing report')
        if [(v['cycle'],v['index']) for v in r['rows']]!=[(c,i) for c in range(10) for i in range(6)]:
            raise ValueError('Timing sample sequence differs')
        if r['inputs']!=[{k:v[k] for k in ('image_sha256','image_type','inputs','reference_sha256')} for v in check['rows']]:
            raise ValueError('Timing input identity differs')
        if e.old.summary([v['wall_ms'] for v in r['rows']])!=r['timing']: raise ValueError('Timing summary differs')
    for rnd in ('r1','r2'):
        ref,out=by['reference',rnd],by['candidate',rnd]
        def memory(row):
            return (row['after_warmup']['memory']['NvMapMemUsed_kib']-row['before']['memory']['NvMapMemUsed_kib'])/1024
        rm,cm=memory(ref),memory(out)
        if min(rm,cm)<0: raise ValueError('Invalid memory baseline')
        improvement=(1-out['timing']['mean_ms']/ref['timing']['mean_ms'])*100
        p95=out['timing']['p95_ms']/ref['timing']['p95_ms']; extra=cm-rm
        ok=improvement>=5 and p95<=1.05 and extra<=256; gate.append(ok)
        result['rounds'].append(dict(round=rnd,reference=ref['timing'],candidate=out['timing'],mean_improvement_percent=improvement,
            p95_ratio=p95,reference_warm_nvmap_mib=rm,candidate_warm_nvmap_mib=cm,extra_warm_nvmap_mib=extra,passed=ok))
    result.update(benchmark_calls=240,status='eligible_for_full_page_performance_test' if all(gate) else 'performance_or_resource_not_accepted')
    return result


def from_stage(stage):
    read=lambda name:json.loads((stage/name).read_text())
    e.build.require_ranges(read('ranges.json'))
    b=read('build.json')
    if b.get('ranges_sha256')!=e.q.digest(stage/'ranges.json'):
        raise ValueError('Range provenance differs')
    if b['status']!='built_not_validated':
        if b['status'] not in ('failed','no_engine_returned'): raise ValueError('Build still incomplete')
        return dict(status='build_not_accepted',build_status=b['status'],accepted_for_release=False,benchmark_calls=0)
    check=read('check.json')
    if (check['build_sha256']!=e.q.digest(stage/'build.json') or check['engine_sha256']!=b['engine_sha256']
            or check['build_script_sha256']!=b['script_sha256'] or check['source_sha256']!=b['source_sha256']
            or check['plugin_sha256']!=b['plugin_sha256']):
        raise ValueError('Check/build provenance differs')
    if check['status']!='finite_same_inputs_compared_not_business_accepted':
        import diagnose_layout_hotspot as d
        diagnostic=d.from_stage(stage)
        if diagnostic!=read('precision_diagnostic.json'): raise ValueError('Postmortem differs')
        return dict(status='precision_boundary_not_accepted',check_status=check['status'],accepted_for_release=False,
            finite_repeat_cases=diagnostic['finite_repeat_cases'],unexpected_half_compute_count=len(diagnostic['unexpected_half_compute']),
            coordinate_half_cast_count=len(diagnostic['coordinate_half_casts']),business_cases=0,benchmark_calls=0,
            reports_sha256={n:e.q.digest(stage/n) for n in ('ranges.json','build.json','check.json','layers.json','precision_diagnostic.json')})
    inspection=e.inspect(read('layers.json'))
    if inspection!=check['inspection']: raise ValueError('Inspector summary differs')
    benchmarks=[read('benchmark_%s_%s.json'%(v,r)) for v in ('reference','candidate') for r in ('r1','r2') if (stage/('benchmark_%s_%s.json'%(v,r))).exists()]
    result=summarize(check,read('evaluation.json'),benchmarks)
    names=('ranges.json','build.json','check.json','layers.json','evaluation.json')+tuple('benchmark_%s_%s.json'%(r['variant'],r['round']) for r in benchmarks)
    result['reports_sha256']={n:e.q.digest(stage/n) for n in names}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,type=Path)
    a=p.parse_args(); result=from_stage(a.stage);e.q.save(a.stage/'summary.json',result);print(json.dumps(result,indent=2))
