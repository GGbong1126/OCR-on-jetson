"""Validate E103 short-profile experiment against predeclared engineering gates."""
import argparse
import json
from pathlib import Path

import numpy as np
import optimize_rec_fp16 as old
import quantize_yolo_ptq as q
from rec_short_fp32 import business_passed


def summarize(check, evaluation, reports):
    if (check['status'] != 'short_ctc_consistent_not_gt' or not check['rows']
            or not all(r['token_sequence_equal'] for r in check['rows'])
            or check['execution']['errors']):
        raise ValueError('Same-crop quality gate failed')
    if (evaluation['status'] != 'all_26_business_consistent_not_gt'
            or len(evaluation['rows']) != 26
            or len({(r['image_sha256'], r['image_type']) for r in evaluation['rows']}) != 26
            or any(not r['passed'] or not business_passed(r) for r in evaluation['rows'])
            or evaluation['short_execution']['errors'] or evaluation['broad_execution']['errors']):
        raise ValueError('Full business quality gate failed')
    expected = {(v, r) for v in ('reference', 'routed') for r in ('r1', 'r2')}
    if len(reports) != 4 or {(r['variant'], r['round']) for r in reports} != expected:
        raise ValueError('Four unique reports required')
    keys = ('script_sha256', 'build_sha256', 'engine_sha256', 'deployment_sha256')
    for report in [evaluation] + reports:
        if any(report[k] != check[k] for k in keys):
            raise ValueError('Candidate provenance differs')
    base = reports[0]
    for r in reports:
        if (r['status'] != 'measured' or not r['pre_post_tokens_equal'] or r['execution']['errors']
                or any(r[k] != base[k] for k in ('image_sha256', 'input_shapes', 'input_sha256', 'warmup', 'repeat'))):
            raise ValueError('Failed, drifted or different workload')
        n = len(r['input_shapes'])
        if (n != len(r['input_sha256']) or n != 50 or r['warmup'] != 3 or r['repeat'] != 20
                or [(v['cycle'], v['index']) for v in r['rows']] != [(c, i) for c in range(20) for i in range(n)]):
            raise ValueError('Incomplete or reordered benchmark')
        times = [v['wall_ms'] for v in r['rows']]
        if not np.isfinite(times).all() or min(times) <= 0:
            raise ValueError('Invalid timings')
        computed = old.summary(times)
        if any(not np.isclose(r['timing'][k], v, rtol=1e-9, atol=1e-9) for k, v in computed.items()):
            raise ValueError('Stored statistics differ from samples')
        for point in ('before', 'after_warmup', 'after'):
            mem = r[point]['memory']['NvMapMemUsed_kib']
            if not isinstance(mem, int) or mem < 0:
                raise ValueError('Missing or invalid memory snapshot')
    pairs = {(r['variant'], r['round']): r for r in reports}
    rounds = []
    for round_ in ('r1', 'r2'):
        ref, cand = (pairs[v, round_] for v in ('reference', 'routed'))
        speed = 100 * (1 - cand['timing']['mean_ms'] / ref['timing']['mean_ms'])
        tail = 100 * (cand['timing']['p95_ms'] / ref['timing']['p95_ms'] - 1)
        def memory_delta(r):
            return (r['after_warmup']['memory']['NvMapMemUsed_kib'] - r['before']['memory']['NvMapMemUsed_kib']) / 1024
        rm, cm = memory_delta(ref), memory_delta(cand)
        row = dict(round=round_, reference=ref['timing'], routed=cand['timing'],
                   mean_reduction_percent=speed, p95_increase_percent=tail,
                   reference_nvmap_delta_mib=rm, routed_nvmap_delta_mib=cm,
                   extra_nvmap_mib=cm-rm,
                   performance_gate=speed >= 5 and tail <= 5,
                   memory_gate=0 <= cm-rm <= 384 and min(rm, cm) >= 0)
        rounds.append(row)
    passed = all(r['performance_gate'] and r['memory_gate'] for r in rounds)
    return dict(status='eligible_for_full_page_performance_test' if passed else 'not_selected_for_release',
                same_crop_count=len(check['rows']), business_pairs=26, measured_calls=4000,
                rounds=rounds, deployment_changed=False,
                note='Predeclared engineering screen: both means improve >=5%, p95 <=5% worse, extra prewarmed NvMap <=384MiB. Snapshots not peaks. Single original-image workload, no GT, no statistical significance or full-page speed claim. Quality consistency is not accuracy.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, required=True)
    a = p.parse_args()
    names = ['check.json', 'evaluation.json'] + ['benchmark_%s_%s.json' % (v,r)
             for v in ('reference','routed') for r in ('r1','r2')]
    values = [json.loads((a.stage/name).read_text()) for name in names]
    result = summarize(values[0], values[1], values[2:])
    result['reports_sha256'] = {name:q.digest(a.stage/name) for name in names}
    result['summary_script_sha256'] = q.digest(__file__)
    q.save(a.stage/'summary.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
