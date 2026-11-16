"""Recompute E111 retained-output/business/performance gates; never deploy."""
import argparse
import json
from pathlib import Path

import evaluate_layout_outputs as e

META=('evaluator_sha256','build_sha256','prepare_sha256','engine_sha256','helper_sha256','provenance')


def summarize_business_diagnostic(report, meta, diagnostic_sha):
    if (report['status']!='diagnostic_completed_not_accepted' or len(report['rows'])!=26
            or report['candidate_execution']['calls']!=26 or report['candidate_execution']['errors']
            or report['accepted_for_release'] or report['benchmark_authorized']
            or report['original_numeric_gate']!='numeric_not_accepted'
            or any(report[k]!=v for k,v in meta.items())
            or report['output_diagnostic_sha256']!=diagnostic_sha):
        raise ValueError('Invalid business diagnostic lineage')
    cases={(r['image_sha256'],r['image_type']) for r in report['rows']}
    images={h for h,_ in cases}
    if len(images)!=13 or cases!={(h,k) for h in images for k in (1,3)}:raise ValueError('Diagnostic cases differ')
    consistent=[e.shared.business_passed(r) for r in report['rows']]
    if consistent!=[r['business_consistent'] for r in report['rows']] or sum(consistent)!=report['business_consistent_count']:
        raise ValueError('Diagnostic decision differs')
    return dict(business_diagnostic_cases=26,business_diagnostic_consistent=sum(consistent),
        diagnostic_full_text_changed=sum(not r['strict_equal']['FullText'] for r in report['rows']),
        diagnostic_ner_changed=sum(not r['strict_equal']['NER'] for r in report['rows']),
        diagnostic_number_symbol_changed=sum(not r['number_symbol_sequence_equal'] for r in report['rows']),
        note='Original numeric gate still failed; additional untimed business diagnostic is not acceptance or benchmark authorization.')


def summarize(check, quality, benchmarks):
    if (check['status']!='retained_outputs_checked' or len(check['rows'])!=6
            or check['execution']['calls']!=12 or check['execution']['errors']
            or not all(e.numeric_passed(r) and r['passed'] for r in check['rows'])):
        raise ValueError('Invalid numeric gate')
    if (any(quality[k]!=check[k] for k in META) or len(quality['rows'])!=26
            or quality['candidate_execution']['calls']!=26 or quality['candidate_execution']['errors']):
        raise ValueError('Incomplete business report')
    cases=[(r['image_sha256'],r['image_type']) for r in quality['rows']]
    images={h for h,_ in cases}
    if len(images)!=13 or len(cases)!=len(set(cases)) or set(cases)!={(h,k) for h in images for k in (1,3)}:
        raise ValueError('Expected 13 images with both paths')
    passed=[e.shared.business_passed(r) for r in quality['rows']]
    if passed!=[r['passed'] for r in quality['rows']]: raise ValueError('Stored decision differs')
    expected='all_26_business_consistent_not_gt' if all(passed) else 'differences_not_accepted'
    if quality['status']!=expected: raise ValueError('Business status differs')
    result=dict(accepted_for_release=False,numeric_cases=6,business_cases=26,business_passed=sum(passed),
        full_text_changed=sum(not r['strict_equal']['FullText'] for r in quality['rows']),
        number_symbol_changed=sum(not r['number_symbol_sequence_equal'] for r in quality['rows']),
        ner_changed=sum(not r['strict_equal']['NER'] for r in quality['rows']),
        boxes_exactly_equal=sum(r['boxes_exactly_equal'] for r in quality['rows']),benchmark_calls=0,rounds=[],
        note='Baseline consistency, not GT. FP32 graph/output-transfer optimization, not quantization or trained pruning. No automatic release.')
    if not all(passed):
        if benchmarks: raise ValueError('Failed quality cannot authorize timing')
        return dict(result,status='quality_not_accepted')
    if not benchmarks: return dict(result,status='quality_passed_performance_pending')
    by={(r['variant'],r['round']):r for r in benchmarks}
    if len(benchmarks)!=4 or set(by)!={(v,r) for v in ('reference','candidate') for r in ('r1','r2')}:
        raise ValueError('Four independent timing processes required')
    for r in benchmarks:
        if (any(r[k]!=check[k] for k in META) or r['status']!='measured' or r['warmup']!=3 or r['repeat']!=10
                or r['execution']['errors'] or r['execution']['calls']!=84 or not r['pre_post_outputs_exact']):
            raise ValueError('Invalid timing report')
        if [(v['cycle'],v['index']) for v in r['rows']]!=[(c,i) for c in range(10) for i in range(6)]:
            raise ValueError('Changed timing sequence')
        if r['inputs']!=[{k:v[k] for k in ('image_sha256','image_type','inputs','reference_sha256')} for v in check['rows']]:
            raise ValueError('Changed timing input identity')
        if e.old.summary([v['wall_ms'] for v in r['rows']])!=r['timing']: raise ValueError('Invalid timing summary')
    for rnd in ('r1','r2'):
        ref,out=by['reference',rnd],by['candidate',rnd]
        def mem(row): return (row['after_warmup']['memory']['NvMapMemUsed_kib']-row['before']['memory']['NvMapMemUsed_kib'])/1024
        rm,cm=mem(ref),mem(out)
        if min(rm,cm)<0: raise ValueError('Invalid memory baseline')
        improvement=(1-out['timing']['mean_ms']/ref['timing']['mean_ms'])*100
        p95=out['timing']['p95_ms']/ref['timing']['p95_ms']; extra=cm-rm
        result['rounds'].append(dict(round=rnd,reference=ref['timing'],candidate=out['timing'],mean_improvement_percent=improvement,
            p95_ratio=p95,reference_warm_nvmap_mib=rm,candidate_warm_nvmap_mib=cm,extra_warm_nvmap_mib=extra,
            passed=improvement>=5 and p95<=1.05 and extra<=256))
    return dict(result,benchmark_calls=240,status='eligible_for_full_page_performance_test' if all(r['passed'] for r in result['rounds']) else 'performance_or_resource_not_accepted')


def from_stage(stage):
    read=lambda name:json.loads((stage/name).read_text())
    built=read('build.json')
    if built['status']!='built_not_validated':
        if built['status']!='failed': raise ValueError('Build incomplete')
        return dict(status='build_not_accepted',accepted_for_release=False,benchmark_calls=0)
    meta=e.metadata(stage); check=read('check.json')
    if any(check[k]!=v for k,v in meta.items()): raise ValueError('Check provenance differs')
    if check['status']!='retained_outputs_checked':
        if check['status'] not in ('failed','numeric_not_accepted'): raise ValueError('Check incomplete')
        result=dict(status='retained_outputs_not_accepted',accepted_for_release=False,business_cases=0,benchmark_calls=0,
            finite_cases=len(check['rows']),numeric_passed=sum(e.numeric_passed(r) for r in check['rows']),
            reports_sha256={n:e.q.digest(stage/n) for n in ('prepare.json','build.json','check.json')})
        if (stage/'business_diagnostic.json').exists():
            report=read('business_diagnostic.json')
            result.update(summarize_business_diagnostic(report,meta,e.q.digest(stage/'output_diagnostic.json')))
            result['reports_sha256'].update({n:e.q.digest(stage/n) for n in ('output_diagnostic.json','business_diagnostic.json')})
        return result
    if e.inspect(read('layers.json'))!=check['inspection']: raise ValueError('Inspection differs')
    benchmarks=[read('benchmark_%s_%s.json'%(v,r)) for v in ('reference','candidate') for r in ('r1','r2') if (stage/('benchmark_%s_%s.json'%(v,r))).exists()]
    result=summarize(check,read('evaluation.json'),benchmarks)
    names=('prepare.json','build.json','check.json','layers.json','evaluation.json')+tuple('benchmark_%s_%s.json'%(r['variant'],r['round']) for r in benchmarks)
    result['reports_sha256']={n:e.q.digest(stage/n) for n in names}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--stage',required=True,type=Path)
    a=p.parse_args(); result=from_stage(a.stage.resolve()); e.q.save(a.stage/'summary.json',result)
    print(json.dumps(result,indent=2))
