"""E112 auditable policy/performance decisions. No promotion or HTTP startup."""
import argparse
from pathlib import Path
import layout_semantic_trial as m

META=('prior_metadata','prior_archive_sha256','prior_reports_sha256','script_sha256','diagnostic_helper_sha256','plan_sha256')


def summarize(plan,quality,benchmarks,expected_inputs):
    if plan['rules']!=m.RULES or quality['old_raw_gate']!='numeric_not_accepted':raise ValueError('Policy/history differs')
    if (len(quality['rows'])!=26 or quality['candidate_execution']['calls']!=26 or quality['candidate_execution']['errors']
            or any(quality[k]!=plan[k] for k in META if k!='plan_sha256')
            or [{k:r[k] for k in ('image_sha256','image_type')} for r in quality['rows']]!=plan['image_cases']):
        raise ValueError('Incomplete quality/provenance')
    cases={(r['image_sha256'],r['image_type']) for r in quality['rows']};images={h for h,_ in cases}
    if len(images)!=13 or cases!={(h,k) for h in images for k in (1,3)}:raise ValueError('Image coverage differs')
    passed=[m.row_passed(r) for r in quality['rows']]
    expected='semantic_business_consistent_not_gt' if all(passed) else 'quality_not_accepted'
    if quality['status']!=expected or passed!=[r['passed'] for r in quality['rows']]:raise ValueError('Stored quality decision differs')
    result=dict(accepted_for_release=False,old_raw_gate='numeric_not_accepted',semantic_cases=26,
        semantic_passed=sum(passed),benchmark_calls=0,rounds=[],
        numeric_allclose_cases=sum(all(v['allclose_diagnostic'] for v in r['raw_numeric_diagnostic'].values()) for r in quality['rows']),
        business_consistent=sum(m.e.shared.business_passed(r) for r in quality['rows']),
        fallback_totals={v:{role:sum(r['cpu_fallback_calls'][v][role] for r in quality['rows']) for role in ('det','rec')} for v in ('reference','candidate')},
        max_effective_coordinate_delta=max((r['effective_match']['matched_max_coordinate_delta'] or 0) for r in quality['rows']),
        note='E112 business/effective-box acceptance only on existing development images, not GT or independent generalization. E111 raw-table failure preserved. FP32 graph/output-copy optimization, not quantization. No automatic release.')
    if not all(passed):
        if benchmarks:raise ValueError('Quality failure cannot authorize benchmark')
        return dict(result,status='quality_not_accepted')
    if not benchmarks:return dict(result,status='semantic_quality_passed_performance_pending')
    order=[('reference','r1'),('candidate','r1'),('candidate','r2'),('reference','r2')]
    by={(r['variant'],r['round']):r for r in benchmarks}
    if len(benchmarks)!=4 or set(by)!=set(order):raise ValueError('Four timing processes required')
    for row in benchmarks:
        if (any(row[k]!=quality[k] for k in META) or row['status']!='measured'
                or row['warmup']!=3 or row['repeat']!=10 or row['execution']['calls']!=84 or row['execution']['errors']
                or not row['pre_post_outputs_exact'] or row['inputs']!=expected_inputs
                or row['started_ns']>=row['ended_ns']):raise ValueError('Invalid benchmark provenance or execution')
        if [(r['cycle'],r['index']) for r in row['rows']]!=[(c,i) for c in range(10) for i in range(6)]:raise ValueError('Sample order changed')
        if m.old.summary([r['wall_ms'] for r in row['rows']])!=row['timing'] or row['timing']['mean_ms']<=0:raise ValueError('Timing summary differs')
    if any(by[a]['ended_ns']>by[b]['started_ns'] for a,b in zip(order,order[1:])):raise ValueError('Process order/overlap differs')
    for rnd in ('r1','r2'):
        ref,out=by['reference',rnd],by['candidate',rnd]
        def mem(r):return (r['after_warmup']['memory']['NvMapMemUsed_kib']-r['before']['memory']['NvMapMemUsed_kib'])/1024
        rm,cm=mem(ref),mem(out)
        if min(rm,cm)<0:raise ValueError('Negative memory increment')
        gain=(1-out['timing']['mean_ms']/ref['timing']['mean_ms'])*100
        tail=out['timing']['p95_ms']/ref['timing']['p95_ms'];extra=cm-rm
        result['rounds'].append(dict(round=rnd,reference=ref['timing'],candidate=out['timing'],
            mean_improvement_percent=gain,p95_ratio=tail,extra_warm_nvmap_mib=extra,
            reference_warm_nvmap_mib=rm,candidate_warm_nvmap_mib=cm,
            passed=gain>=m.RULES['mean_improvement_min_percent'] and tail<=m.RULES['p95_ratio_max'] and extra<=m.RULES['extra_warm_nvmap_max_mib']))
    return dict(result,benchmark_calls=240,status='eligible_for_full_page_performance_test' if all(r['passed'] for r in result['rounds']) else 'performance_or_resource_not_accepted')


def from_stage(stage):
    meta=m.metadata(stage);plan=m.read(stage/'plan.json');quality=m.read(stage/'quality.json')
    if any(quality[k]!=v for k,v in meta.items()):raise ValueError('Current provenance differs')
    if m.e.inspect(m.read(stage/'layers.json'))!=quality['inspection']:raise ValueError('Inspector differs')
    checked=m.read(m.PRIOR/'check.json')
    expected_inputs=[{k:r[k] for k in ('image_sha256','image_type','inputs','reference_sha256')} for r in checked['rows']]
    names=['plan.json','quality.json','layers.json'];benchmarks=[]
    for variant,rnd in (('reference','r1'),('candidate','r1'),('candidate','r2'),('reference','r2')):
        name='benchmark_%s_%s.json'%(variant,rnd)
        if (stage/name).exists():
            row=m.read(stage/name)
            if row['quality_sha256']!=m.q.digest(stage/'quality.json'):raise ValueError('Timing used different quality report')
            names.append(name);benchmarks.append(row)
    result=summarize(plan,quality,benchmarks,expected_inputs)
    result['reports_sha256']={n:m.q.digest(stage/n) for n in names}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args();result=from_stage(a.stage.resolve());m.q.save(a.stage/'summary.json',result)
    print(m.json.dumps(result,indent=2))
