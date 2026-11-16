"""Validate four E102 reports and summarize diagnostics without document text."""
import argparse
import json
import math
from pathlib import Path

import quantize_yolo_ptq as q


def summarize(reports):
    expected = {(v, r) for v in ('reference', 'skip_same') for r in ('r1', 'r2')}
    if len(reports) != 4 or {(r['variant'], r['round']) for r in reports} != expected:
        raise ValueError('Four unique paired reports required')
    base = reports[0]
    keys = ('input_shapes', 'input_sha256', 'baseline_output_sha256',
            'image_sha256', 'engine_sha256', 'deployment_sha256',
            'backend_sha256', 'script_sha256', 'warmup_cycles', 'repeat_cycles')
    for report in reports:
        if (report['status'] != 'diagnosed_exact_same_engine_not_gt'
                or report['execution']['errors'] or any(report[k] != base[k] for k in keys)):
            raise ValueError('Report failed, backend error, or provenance mismatch')
        n, cycles = len(report['input_shapes']), report['repeat_cycles']
        groups = report['patterns']
        if len(groups) != 5 or {g['name'] for g in groups} != {
                'repeat_a', 'repeat_b', 'alternate_ab', 'alternate_ba', 'original_sequence'}:
            raise ValueError('Incomplete diagnostic patterns')
        for group in groups:
            if len(group['rows']) != cycles * (n if group['name'] == 'original_sequence' else 2):
                raise ValueError('Incomplete diagnostic calls')
            for row in group['rows']:
                if (not row['output_exact'] or not 0 <= row['index'] < n
                        or not math.isfinite(row['call_ms']) or row['call_ms'] <= 0):
                    raise ValueError('Invalid diagnostic call')
        rows = report['bare_rows']
        if (len(rows) != cycles * n or
                [(v['cycle'], v['index']) for v in rows] != [(c, i) for c in range(cycles) for i in range(n)]):
            raise ValueError('Incomplete or reordered bare timing')
        if any(not math.isfinite(v['wall_ms']) or v['wall_ms'] <= 0 for v in rows):
            raise ValueError('Invalid timing')
    results = []
    by_key = {(r['variant'], r['round']): r for r in reports}
    for round_ in ('r1', 'r2'):
        ref, cand = (by_key[v, round_] for v in ('reference', 'skip_same'))
        mean = lambda r: sum(v['wall_ms'] for v in r['bare_rows']) / len(r['bare_rows'])
        rm, cm = mean(ref), mean(cand)
        results.append({'round': round_, 'reference_mean_ms': rm, 'skip_same_mean_ms': cm,
                        'latency_reduction_percent': 100 * (1 - cm / rm)})
    comparisons = []
    for report in reports:
        groups = {r['name']: r for r in report['patterns']}
        static = (groups['repeat_a']['means_ms']['call_ms'] + groups['repeat_b']['means_ms']['call_ms']) / 2
        alternate = (groups['alternate_ab']['means_ms']['call_ms'] + groups['alternate_ba']['means_ms']['call_ms']) / 2
        seq = groups['original_sequence']
        comparisons.append({'variant': report['variant'], 'round': report['round'],
            'matched_mix_repeated_call_ms': static, 'alternating_call_ms': alternate,
            'alternating_extra_ms_diagnostic': alternate - static,
            'sequence_shape_changes': seq['shape_changes'],
            'sequence_shape_api_calls': sum(v['shape_api_calls'] for v in seq['rows']),
            'sequence_means_ms': seq['means_ms'],
            'measured_malloc_count': sum(g['malloc_count'] for g in groups.values()),
            'measured_free_count': sum(v['free_count'] for g in groups.values() for v in g['rows'])})
    return {'status': 'four_reports_validated_not_deployment_acceptance',
            'source_script_sha256': base['script_sha256'], 'engine_sha256': base['engine_sha256'],
            'image_sha256': base['image_sha256'], 'input_count': len(base['input_shapes']),
            'diagnostic_calls': sum(len(g['rows']) for r in reports for g in r['patterns']),
            'bare_timing_calls': sum(len(r['bare_rows']) for r in reports),
            'same_input_and_baseline_output_hashes': True,
            'rounds': results, 'patterns': comparisons,
            'profile_probes': [r['profile_probe'] for r in reports],
            'cache_simulation': base['sequence_cache_simulation'],
            'context_device_bytes': base['context_device_bytes'],
            'note': 'One image repeated, no independent accuracy set, no full candidate business/HTTP benchmark. API timings are host wall intervals including synchronous execution, not pure enqueue/kernel time. No locked clocks or statistical significance claim. No deployment changed.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, required=True)
    a = p.parse_args()
    paths = [a.stage / ('runtime_%s_%s.json' % (v, r))
             for v in ('reference', 'skip_same') for r in ('r1', 'r2')]
    result = summarize([json.loads(p.read_text()) for p in paths])
    result['reports_sha256'] = {p.name: q.digest(p) for p in paths}
    q.save(a.stage / 'summary.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
