"""E109 batch2 evidence gates; compare identical nine-tile sequences, not calls."""
import argparse
import json
from pathlib import Path

import numpy as np
import evaluate_enhance_batch2 as e

META = ('script_sha256', 'build_script_sha256', 'build_sha256', 'export_sha256',
        'engine_sha256', 'baseline_engine_sha256', 'onnx_sha256',
        'deployment_sha256', 'plan_sha256', 'helpers_sha256', 'diagnostic_helper_sha256')


def execution(stats, calls):
    if stats['errors'] != 0 or stats['calls'] != calls:
        raise ValueError('Execution count/error mismatch')


def summarize(check, quality, reports):
    if check['status'] != 'finite_original_tiles_not_business_accepted':
        raise ValueError('Completed real-input check required')
    execution(check['execution'], 36)
    execution(check['tail_execution'], 6)
    expected = [dict(indices=[offset+i for i in g], batch=len(g))
                for offset in (0, 9, 18) for g in e.batches(9)]
    pairs = check['pairs']
    if ([{k: p[k] for k in ('indices', 'batch')} for p in pairs] != expected
            or not all(p['repeat_exact'] and p['swap']['allclose'] for p in pairs)):
        raise ValueError('Pair order/repeat/swap check incomplete')
    numeric = check['rows']
    if len(numeric) != 27 or [r['index'] for r in numeric] != list(range(27)):
        raise ValueError('Missing/reordered real tiles')
    for pair in expected:
        for slot, index in enumerate(pair['indices']):
            r = numeric[index]
            if (r['batch'] != pair['batch'] or r['slot'] != slot or not r['repeat_exact']
                    or r['input_shape'] != [1, 3, 512, 512]
                    or not np.isfinite([r['mae'], r['max_abs_error']]).all()
                    or min(r['mae'], r['max_abs_error']) < 0):
                raise ValueError('Invalid tile check')
    if any(quality[k] != check[k] for k in META):
        raise ValueError('Business evidence provenance changed')
    rows = quality['rows']
    if (quality['status'] not in ('all_cases_consistent_not_gt', 'differences_not_accepted')
            or len(rows) != 14 or sum(r['image_type'] == 3 for r in rows) != 1
            or len({r['image_sha256'] for r in rows if r['image_type'] == 1}) != 13):
        raise ValueError('Incomplete business cases')
    for r in rows:
        hashes = r['tile_input_sha256']
        e.verify_page(r['image_type'], hashes, hashes, r['candidate_batch_sizes'])
        if (not r['same_original_inputs'] or r['tile_calls'] != len(hashes)
                or r['batch2_calls'] != r['candidate_batch_sizes'].count(2)
                or r['original_tail_calls'] != r['candidate_batch_sizes'].count(1)
                or r['passed'] != e.legacy.business_passed(r['strict_equal'], r['field_diagnostics'])):
            raise ValueError('Business route/result mismatch')
    execution(quality['candidate_execution'], sum(r['batch2_calls'] for r in rows))
    if (quality['status'] == 'all_cases_consistent_not_gt') != e.legacy.quality_ok(rows):
        raise ValueError('Stored quality status differs')
    result = dict(status='quality_or_fallback_not_accepted', accepted_for_release=False,
        deployment_changed=False, numeric_tiles=27,
        allclose_passed=sum(r['allclose_diagnostic'] for r in numeric),
        max_abs_error=max(r['max_abs_error'] for r in numeric),
        business_cases=14, passed=sum(r['passed'] for r in rows),
        business_batch2_calls=sum(r['batch2_calls'] for r in rows),
        business_original_tail_calls=sum(r['original_tail_calls'] for r in rows),
        differences={k: sum(not r['strict_equal'][k] for r in rows)
                     for k in ('FullText', 'Text', 'MarkdownBodyLines', 'NER', 'raw_line_texts', 'special')},
        number_symbol_changed=sum(not r['full_text_diagnostics']['number_symbol_sequence_equal'] for r in rows),
        special_text_changes={k: sum(not r['field_diagnostics']['api_special_text_equal'][k] for r in rows)
                              for k in ('title', 'category_no', 'page_no')},
        geometric_changes=sum(not r['boxes_exactly_equal'] for r in rows),
        cpu_fallback_calls={v: {k: sum(r['cpu_fallback_calls'][v][k] for r in rows)
                               for k in ('det', 'rec')} for v in ('reference', 'candidate')},
        benchmark_sequences=0, timed_tiles=0, timed_engine_calls=0, rounds=[],
        note='Existing-image regression, not GT. Timing unit is a complete unchanged 9-tile sequence including odd tail, packing and output copies; excludes split/merge/OCR/HTTP. NvMap snapshots are not peak. No automatic promotion.')
    if not e.previous.quality_ready(rows):
        if reports:
            raise ValueError('Rejected quality must not be benchmarked')
        return result
    if len(reports) != 4 or {(r['variant'], r['round']) for r in reports} != {
            (v, r) for v in ('reference', 'candidate') for r in ('r1', 'r2')}:
        raise ValueError('Four unique timing reports required')
    for r in reports:
        if (any(r[k] != check[k] for k in META) or r['status'] != 'measured'
                or not r['pre_post_outputs_exact'] or r['warmup'] != 3 or r['repeat'] != 20
                or r['inputs'] != [e.describe(v) for v in numeric[:9]]
                or [x['cycle'] for x in r['rows']] != list(range(20))):
            raise ValueError('Timing contract/provenance differs')
        is_ref = r['variant'] == 'reference'
        execution(r['execution'], 216 if is_ref else 24)
        if is_ref:
            if r['batch2_execution'] is not None:
                raise ValueError('Reference must not use batch2')
        else:
            execution(r['batch2_execution'], 96)
        values = [x['wall_ms'] for x in r['rows']]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Invalid timing')
        if any(not np.isclose(r['timing'][k], v, rtol=1e-9, atol=1e-9)
               for k, v in e.common.summary(values).items()):
            raise ValueError('Stored statistics differ')
    lookup = {(r['variant'], r['round']): r for r in reports}
    for rnd in ('r1', 'r2'):
        ref, out = (lookup[v, rnd] for v in ('reference', 'candidate'))
        delta = lambda r: (r['after_warmup']['memory']['NvMapMemUsed_kib'] - r['before']['memory']['NvMapMemUsed_kib']) / 1024
        rm, cm = delta(ref), delta(out)
        mean = 100 * (1 - out['timing']['mean_ms'] / ref['timing']['mean_ms'])
        p95 = 100 * (out['timing']['p95_ms'] / ref['timing']['p95_ms'] - 1)
        result['rounds'].append(dict(round=rnd, reference=ref['timing'], candidate=out['timing'],
            mean_reduction_percent=mean, p95_increase_percent=p95, reference_nvmap_delta_mib=rm,
            candidate_nvmap_delta_mib=cm, extra_nvmap_mib=cm-rm,
            performance_gate=mean >= 5 and p95 <= 5,
            memory_gate=np.isfinite([rm, cm]).all().item() and min(rm, cm) >= 0 and cm-rm <= 512))
    result.update(benchmark_sequences=80, timed_tiles=720, timed_engine_calls=560,
        status='eligible_for_full_page_performance_test' if all(r['performance_gate'] and r['memory_gate'] for r in result['rounds']) else 'performance_or_resource_not_accepted')
    result['note'] += ' Only 20 sequences/process; tail percentiles exploratory, not statistical significance.'
    return result


def from_stage(stage):
    meta = e.metadata(stage)
    if e.q.digest(stage/'enhance_batch2.engine') != meta['engine_sha256']:
        raise ValueError('Engine changed')
    check = e.require(stage, 'check.json', 'finite_original_tiles_not_business_accepted')
    quality = json.loads((stage/'evaluation.json').read_text())
    if check['inspection'] != e.previous.inspect(json.loads((stage/'layers.json').read_text())):
        raise ValueError('Inspection differs')
    names = ['export.json', 'build.json', 'build.log', 'check.json', 'layers.json', 'evaluation.json']
    bench = ['benchmark_%s_%s.json' % (v, r) for v in ('reference', 'candidate') for r in ('r1', 'r2')]
    result = summarize(check, quality, [json.loads((stage/n).read_text()) for n in bench if (stage/n).exists()])
    result.update(script_sha256=e.q.digest(__file__), reports_sha256={n: e.q.digest(stage/n) for n in names+bench if (stage/n).exists()})
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, required=True)
    stage = p.parse_args().stage.resolve()
    result = from_stage(stage)
    e.q.save(stage/'summary.json', result)
    print(json.dumps(result, indent=2))
