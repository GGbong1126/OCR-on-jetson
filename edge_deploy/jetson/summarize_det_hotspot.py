"""Audit E104 quality first; compare timing only for a complete accepted trial."""
import argparse
import json
from pathlib import Path

import numpy as np
import evaluate_det_hotspot as evaluation
import det_hotspot_mixed as build
import optimize_rec_fp16 as old
import quantize_yolo_ptq as q

META=('script_sha256','build_sha256','engine_sha256','source_sha256','deployment_sha256','plan_sha256')


def summarize_failed_build(record, ranges, log, artifacts):
    """Classify this constrained build failure, never an accuracy/perf result."""
    build.require_ranges(ranges)
    if (record['status']!='failed' or record['source_sha256']!=build.SOURCE_SHA
            or record['deployment_sha256']!=build.DEPLOYMENT_SHA
            or record['targets']!=list(build.TARGETS)
            or record['profile']!=[list(v) for v in build.PROFILE]
            or record['workspace_bytes']!=256*1024**2 or record['flags']!=1025
            or not np.isfinite(record['build_s']) or record['build_s']<=0):
        raise ValueError('Unexpected failed-build policy')
    target=[r for r in record['constraints'] if r['policy']!='fp32']
    if (len(target)!=1 or target[0]['name']!='Conv.87'
            or target[0]['kind']!='CONVOLUTION'
            or target[0]['policy']!='fp16_compute_fp32_output'):
        raise ValueError('Unexpected precision boundary')
    forbidden={'det_hotspot.engine','timing.cache','check.json','layers.json','evaluation.json'}
    if forbidden.intersection(artifacts) or any(n.startswith('benchmark_') for n in artifacts):
        raise ValueError('Downstream artifacts exist despite build failure')
    failure='No implementation of layer Conv.87 obeys the requested constraints.'
    if failure not in log or 'kOBEY_PRECISION_CONSTRAINTS was set' not in log:
        raise ValueError('Specific precision-constraint failure not evidenced')
    return dict(status='build_constraints_not_satisfied',build_s=record['build_s'],
        target='Conv.87',requested_compute='fp16',requested_output='fp32',
        workspace_bytes=record['workspace_bytes'],engine_created=False,
        range_cases=6,candidate_inference_calls=0,business_pairs=0,benchmark_calls=0,
        deployment_changed=False,
        note='No conforming implementation for this build configuration. Not proof that FP16 is unsupported at every workspace/shape/output precision. No candidate accuracy or speed result; no constraint relaxation. Original FP32 probes are diagnostic, not proof of TensorRT internal ranges.')


def summarize(check,quality,reports):
    if (check['status']!='finite_same_inputs_compared_not_business_accepted' or len(check['rows'])!=6
            or check['execution']['errors'] or not all(r['repeat_exact'] for r in check['rows'])):
        raise ValueError('Incomplete numeric/precision execution gate')
    if (quality['status'] not in ('all_26_business_consistent_not_gt','differences_not_accepted')
            or len(quality['rows'])!=26
            or len({(r['image_sha256'],r['image_type']) for r in quality['rows']})!=26
            or any(quality[k]!=check[k] for k in META) or quality['candidate_execution']['errors']):
        raise ValueError('Incomplete or mismatched business experiment')
    for row in quality['rows']:
        if row['passed']!=evaluation.business_passed(row):raise ValueError('Stored quality gate disagrees')
    passed=sum(r['passed'] for r in quality['rows'])
    if (quality['status']=='all_26_business_consistent_not_gt') != (passed==26):
        raise ValueError('Stored status disagrees with rows')
    fields=('FullText','Text','MarkdownBodyLines','NER','raw_line_texts','special')
    fallback={v:{role:sum(r['cpu_fallback_calls'][v][role] for r in quality['rows'])
                    for role in ('det','rec')} for v in ('reference','candidate')}
    result=dict(status='quality_not_accepted',business_pairs=26,passed=passed,
        differences={k:sum(not r['strict_equal'][k] for r in quality['rows']) for k in fields},
        number_symbol_changed=sum(not r['number_symbol_sequence_equal'] for r in quality['rows']),
        special_text_changes={k:sum(not r['fields']['api_special_text_equal'][k] for r in quality['rows'])
                              for k in ('title','category_no','page_no')},
        geometric_changes=sum(not r['boxes_exactly_equal'] for r in quality['rows']),
        cpu_fallback_calls=fallback,benchmark_calls=0,rounds=[],deployment_changed=False,
        note='Regression consistency, not GT accuracy. Existing 13 images, not an independent test set. No text normalization. Profiler/range probes do not establish internal accumulator precision.')
    if passed!=26:
        if reports:raise ValueError('Rejected candidate must not be benchmarked')
        return result
    expected={(v,r) for v in ('reference','mixed') for r in ('r1','r2')}
    if len(reports)!=4 or {(r['variant'],r['round']) for r in reports}!=expected:
        raise ValueError('Four unique measured reports required')
    for report in reports:
        if (report['status']!='measured' or report['execution']['errors'] or not report['pre_post_outputs_exact']
                or any(report[k]!=check[k] for k in META) or report['warmup']!=3 or report['repeat']!=10):
            raise ValueError('Timing provenance/health differs')
        meta=[{k:r[k] for k in ('image_sha256','image_type','input_shape','input_sha256','reference_sha256')} for r in check['rows']]
        if report['inputs']!=meta:raise ValueError('Timing inputs differ')
        rows=report['rows']
        if [(r['cycle'],r['index']) for r in rows]!=[(c,i) for c in range(10) for i in range(6)]:
            raise ValueError('Incomplete or reordered samples')
        times=[r['wall_ms'] for r in rows]
        if not np.isfinite(times).all() or min(times)<=0:raise ValueError('Invalid timing')
        if any(not np.isclose(report['timing'][k],v,rtol=1e-9,atol=1e-9) for k,v in old.summary(times).items()):
            raise ValueError('Stored timing summary differs')
    pairs={(r['variant'],r['round']):r for r in reports}
    for round_ in ('r1','r2'):
        ref,cand=(pairs[v,round_] for v in ('reference','mixed'))
        def mem_delta(r):
            return (r['after_warmup']['memory']['NvMapMemUsed_kib']-r['before']['memory']['NvMapMemUsed_kib'])/1024
        rm,cm=mem_delta(ref),mem_delta(cand)
        mean=100*(1-cand['timing']['mean_ms']/ref['timing']['mean_ms'])
        p95=100*(cand['timing']['p95_ms']/ref['timing']['p95_ms']-1)
        result['rounds'].append(dict(round=round_,reference=ref['timing'],mixed=cand['timing'],
            mean_reduction_percent=mean,p95_increase_percent=p95,reference_nvmap_delta_mib=rm,mixed_nvmap_delta_mib=cm,
            extra_nvmap_mib=cm-rm,performance_gate=mean>=5 and p95<=5,memory_gate=min(rm,cm)>=0 and cm-rm<=256))
    result['benchmark_calls']=240
    result['no_extra_cpu_fallback']=all(fallback['candidate'][k]<=fallback['reference'][k] for k in ('det','rec'))
    passed=all(r['performance_gate'] and r['memory_gate'] for r in result['rounds']) and result['no_extra_cpu_fallback']
    result['status']='eligible_for_full_page_performance_test' if passed else 'performance_or_resource_not_accepted'
    result['note']+=' 60 samples/configuration: tails exploratory. Predeclared engineering screen, not statistical significance, full-page speed or release acceptance; memory snapshots not peaks.'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);a=p.parse_args()
    old.fresh(a.stage/'summary.json')
    record=json.loads((a.stage/'build.json').read_text())
    if record['status']=='failed':
        if (record['ranges_sha256']!=q.digest(a.stage/'ranges.json')
                or record['script_sha256']!=q.digest(a.stage/'det_hotspot_mixed.py')):
            raise ValueError('Failed-build evidence changed')
        result=summarize_failed_build(record,json.loads((a.stage/'ranges.json').read_text()),
            (a.stage/'build.log').read_text(),[p.name for p in a.stage.iterdir()])
        names=['ranges.json','build.json','build.log']
    else:
        check=json.loads((a.stage/'check.json').read_text());quality=json.loads((a.stage/'evaluation.json').read_text())
        paths=[a.stage/('benchmark_%s_%s.json'%(v,r)) for v in ('reference','mixed') for r in ('r1','r2')]
        reports=[json.loads(p.read_text()) for p in paths if p.exists()]
        result=summarize(check,quality,reports)
        names=['ranges.json','build.json','check.json','evaluation.json','layers.json']+[p.name for p in paths if p.exists()]
    result['reports_sha256']={n:q.digest(a.stage/n) for n in names}
    result['script_sha256']=q.digest(__file__)
    q.save(a.stage/'summary.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
