import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'jetson'))
import summarize_enhance_batch2 as s


def fixtures():
    meta = dict.fromkeys(s.META, 'sha')
    pairs, numeric = [], []
    for offset in (0, 9, 18):
        for group in s.e.batches(9):
            indices = [offset+i for i in group]
            pairs.append(dict(indices=indices, batch=len(group), repeat_exact=True, swap={'allclose': True}))
            for slot, i in enumerate(indices):
                numeric.append(dict(index=i, batch=len(group), slot=slot, repeat_exact=True,
                    input_shape=[1, 3, 512, 512], image_sha256=str(offset), input_sha256=str(i),
                    reference_sha256=str(i), candidate_sha256=str(i), mae=0., max_abs_error=0., allclose_diagnostic=True))
    check = dict(meta, status='finite_original_tiles_not_business_accepted', rows=numeric, pairs=pairs,
                 execution=dict(calls=36, errors=0), tail_execution=dict(calls=6, errors=0))
    rows = [dict(image_sha256=str(i), image_type=1, passed=True, same_original_inputs=True,
                 tile_calls=9, tile_input_sha256=['x']*9, candidate_batch_sizes=[2,2,2,2,1],
                 batch2_calls=4, original_tail_calls=1,
                 strict_equal=dict.fromkeys(('FullText','Text','MarkdownBodyLines','NER','raw_line_texts','special'), True),
                 field_diagnostics={'api_special_text_equal': dict.fromkeys(('title','category_no','page_no'), True)},
                 full_text_diagnostics={'number_symbol_sequence_equal': True}, boxes_exactly_equal=True,
                 cpu_fallback_calls={v: dict(det=0, rec=0) for v in ('reference','candidate')}) for i in range(13)]
    control = copy.deepcopy(rows[0])
    control.update(image_type=3, tile_calls=0, tile_input_sha256=[], candidate_batch_sizes=[], batch2_calls=0, original_tail_calls=0)
    rows.insert(0, control)
    quality = dict(meta, status='all_cases_consistent_not_gt', rows=rows, candidate_execution=dict(calls=52, errors=0))
    reports = []
    for variant in ('reference', 'candidate'):
        for rnd in ('r1', 'r2'):
            ref = variant == 'reference'
            ms = 900 if ref else 800
            reports.append(dict(meta, status='measured', variant=variant, round=rnd,
                pre_post_outputs_exact=True, warmup=3, repeat=20,
                inputs=[s.e.describe(r) for r in numeric[:9]],
                execution=dict(calls=216 if ref else 24, errors=0),
                batch2_execution=None if ref else dict(calls=96, errors=0),
                rows=[dict(cycle=i, wall_ms=ms) for i in range(20)], timing=s.e.common.summary([ms]*20),
                before={'memory': {'NvMapMemUsed_kib': 1000}},
                after_warmup={'memory': {'NvMapMemUsed_kib': 1000+1024*(300 if ref else 600)}}))
    return check, quality, reports


class Tests(unittest.TestCase):
    def test_sequence_units_not_batch_call_comparison(self):
        result = s.summarize(*fixtures())
        self.assertEqual((result['benchmark_sequences'], result['timed_tiles'], result['timed_engine_calls']), (80,720,560))
        self.assertEqual(result['status'], 'eligible_for_full_page_performance_test')
        self.assertFalse(result['accepted_for_release'])

    def test_reject_missing_reordered_padded_tiles(self):
        for kind in ('pairs','tail','slot','order','shape','nan'):
            c, q, b = fixtures()
            if kind == 'pairs': c['pairs'].reverse()
            if kind == 'tail': c['tail_execution']['calls'] = 0
            if kind == 'slot': c['rows'][0]['slot'] = 1
            if kind == 'order': c['rows'].reverse()
            if kind == 'shape': c['rows'][0]['input_shape'][0] = 2
            if kind == 'nan': c['rows'][0]['mae'] = float('nan')
            with self.assertRaises(ValueError): s.summarize(c, q, b)

    def test_quality_precedes_performance(self):
        for kind in ('text','fallback'):
            c, q, b = fixtures()
            if kind == 'text':
                q['rows'][1]['strict_equal']['FullText'] = False
                q['rows'][1]['passed'] = False
                q['status'] = 'differences_not_accepted'
            else: q['rows'][1]['cpu_fallback_calls']['candidate']['rec'] = 1
            self.assertEqual(s.summarize(c,q,[])['status'], 'quality_or_fallback_not_accepted')
            with self.assertRaises(ValueError): s.summarize(c,q,b)

    def test_business_counts_negative_control(self):
        for kind in ('tail','batch','negative','usage'):
            c, q, b = fixtures()
            if kind == 'tail': q['rows'][1]['original_tail_calls'] = 0
            if kind == 'batch': q['candidate_execution']['calls'] = 51
            if kind == 'negative': q['rows'][0]['candidate_batch_sizes'] = [1]
            if kind == 'usage': q['rows'][1]['same_original_inputs'] = False
            with self.assertRaises(ValueError): s.summarize(c,q,b)

    def test_timing_source_order_stats_and_execution(self):
        for kind in ('source','order','stats','tail','batch','missing'):
            c, q, b = fixtures()
            if kind == 'source': b[2]['export_sha256'] = 'bad'
            if kind == 'order': b[0]['rows'].reverse()
            if kind == 'stats': b[0]['timing']['mean_ms'] = 1
            if kind == 'tail': b[2]['execution']['calls'] = 0
            if kind == 'batch': b[2]['batch2_execution']['errors'] = 1
            if kind == 'missing': b.pop()
            with self.assertRaises(ValueError): s.summarize(c,q,b)

    def test_resource_or_speed_failure(self):
        for kind in ('speed','memory'):
            c, q, b = fixtures()
            if kind == 'speed':
                b[2]['rows'] = [dict(cycle=i, wall_ms=890) for i in range(20)]
                b[2]['timing'] = s.e.common.summary([890]*20)
            else: b[2]['after_warmup']['memory']['NvMapMemUsed_kib'] += 513*1024
            self.assertEqual(s.summarize(c,q,b)['status'], 'performance_or_resource_not_accepted')


if __name__ == '__main__': unittest.main()
