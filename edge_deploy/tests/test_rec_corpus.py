import copy
from pathlib import Path
import sys
import tempfile
import unittest
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import validate_rec_corpus as c
import archive_rec_corpus as a


def fixture():
    inventory = [dict(sha256='%064x' % i, width=1000 + i * 10, height=1800, bytes=100) for i in range(80)]
    excluded = [r['sha256'] for r in inventory[:13]]
    selected = c.select(inventory, set(excluded))
    cases = [dict(index=i, image_sha256=r['sha256'], image_type=kind)
             for i, (r, kind) in enumerate((r, kind) for r in selected for kind in c.RULES['paths'])]
    plan = dict(status='prepared', rules=copy.deepcopy(c.RULES), inventory=inventory,
                excluded_sha256=excluded, selected=selected, cases=cases,
                package={'serve_rec_wide.py': 'factory', 'rec_wide_adapter.py': 'adapter'})
    run = dict(status='completed', rules=copy.deepcopy(c.RULES), parent_pid=1, startup_environment={}, workers=[])
    reports = {}
    for j, name in enumerate(c.RULES['order']):
        run['workers'].append(dict(variant=name, pid=10 + j, exitcode=0))
        state = {k: dict(calls=1, errors=0, cpu_fallback=dict(calls=0, errors=0))
                 for k in ('layout', 'cell', 'det', 'rec', 'yolo', 'enhance', 'ner')}
        report = dict(status='completed_observed_not_gt', variant=name, rules=copy.deepcopy(c.RULES),
            pid=10 + j, parent_pid=1, startup_environment={}, spawn_method='spawn',
            factory_sha256='factory', adapter_sha256='adapter', all_gpu_sessions_closed=True,
            initial_execution=copy.deepcopy(state), rows=[])
        cpu = False
        for case in cases:
            width = (160, 2304, 2305)[case['index'] % 3]
            route = a.expected_route([1, 3, 48, width], name)
            call = dict(index=0, input_shape=[1, 3, 48, width], input_dtype='float32', input_sha256='input',
                        ctc_sha256='ctc', ctc_length=2, output_sha256='wide' if route == 'wide_gpu' else 'original',
                        route=route, cold_cpu=route == 'cpu' and not cpu)
            cpu = cpu or route == 'cpu'
            counts = {k: dict(calls=1, cpu_fallback_calls=0) for k in state}
            counts['rec'] = dict(calls=int(route != 'cpu'), cpu_fallback_calls=int(route == 'cpu'))
            for k, row in counts.items():
                state[k]['calls'] += row['calls']
                state[k]['cpu_fallback']['calls'] += row['cpu_fallback_calls']
            report['rows'].append(dict(case, signature={'business_sha256': str(case['index'])},
                                      counters=counts, rec_calls=[call], diagnostic_wall_s=1.))
        report.update(execution=state, coverage=c.coverage(report['rows']))
        reports[name] = report
    return plan, run, reports


class Tests(unittest.TestCase):
    def test_selection_frozen_unique_excludes_prior(self):
        plan, _, _ = fixture()
        selected = plan['selected']
        self.assertEqual(len({r['sha256'] for r in selected}), 32)
        self.assertFalse(set(r['sha256'] for r in selected) & set(plan['excluded_sha256']))
        self.assertEqual(c.select(list(reversed(plan['inventory'])), set(plan['excluded_sha256'])), selected)

    def test_case_insensitive_image_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (21, 32)).save(root / 'UPPER.JPG')
            Image.new('RGB', (23, 34)).save(root / 'lower.jpg')
            rows, paths = c.inventory(root)
            self.assertEqual(len(rows), 2)
            self.assertEqual({p.suffix for p in paths.values()}, {'.jpg', '.JPG'})
            self.assertTrue(all('file' not in r and 'path' not in r for r in rows))

    def test_duplicate_file_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('a.jpg', 'b.JPG'):
                Image.new('RGB', (21, 32)).save(root / name)
            with self.assertRaises(ValueError):
                c.inventory(root)

    def test_empty_insufficient_selection(self):
        with self.assertRaises(ValueError):
            c.select([], set())

    def test_all_routes_and_outside_union(self):
        for width, ref, candidate in ((31, 'cpu', 'cpu'), (32, 'original_gpu', 'original_gpu'),
                (1536, 'original_gpu', 'original_gpu'), (1537, 'cpu', 'wide_gpu'),
                (2304, 'cpu', 'wide_gpu'), (2305, 'cpu', 'cpu')):
            self.assertEqual(a.expected_route([1, 3, 48, width], 'reference'), ref)
            self.assertEqual(a.expected_route([1, 3, 48, width], 'candidate'), candidate)
        self.assertEqual(a.expected_route([2, 3, 48, 32], 'candidate'), 'cpu')

    def test_complete_pair_gate_and_coverage(self):
        result = a.summarize(*fixture())
        self.assertEqual(result['status'], 'expanded_corpus_consistent_not_gt')
        self.assertEqual(result['paired_cases'], 64)
        self.assertEqual(result['coverage']['candidate']['over_2304_widths'], [2305])
        self.assertEqual(result['coverage']['candidate']['near_upper_2200_2304_widths'], [2304])

    def test_missing_protocol_evidence_rejected(self):
        for mode in ('source', 'coverage', 'calls', 'route', 'init', 'pid', 'env', 'errors', 'hash', 'cleanup'):
            plan, run, reports = fixture()
            r = reports['candidate']
            if mode == 'source': plan['selected'].reverse()
            if mode == 'coverage': r['rows'].pop()
            if mode == 'calls': r['execution']['rec']['calls'] += 1
            if mode == 'route': r['rows'][1]['rec_calls'][0]['route'] = 'original_gpu'
            if mode == 'init': r['rows'][2]['rec_calls'][0]['cold_cpu'] = False
            if mode == 'pid': r['pid'] = 1
            if mode == 'env': r['startup_environment'] = {'CUDA_VISIBLE_DEVICES': '-1'}
            if mode == 'errors': r['execution']['rec']['errors'] = 1
            if mode == 'hash': r['factory_sha256'] = 'other'
            if mode == 'cleanup': r['all_gpu_sessions_closed'] = False
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                a.summarize(plan, run, reports)

    def test_quality_differences_not_silently_accepted(self):
        for mode in ('signature', 'input', 'ctc', 'unchanged_output'):
            plan, run, reports = fixture()
            row = reports['candidate']['rows'][0]
            if mode == 'signature': row['signature'] = {}
            if mode == 'input': row['rec_calls'][0]['input_sha256'] = 'changed'
            if mode == 'ctc': row['rec_calls'][0]['ctc_sha256'] = 'changed'
            if mode == 'unchanged_output': row['rec_calls'][0]['output_sha256'] = 'changed'
            with self.subTest(mode=mode):
                result = a.summarize(plan, run, reports)
                self.assertEqual(result['status'], 'expanded_corpus_not_accepted')
                self.assertEqual(result['failed_case_indices'], [0])

    def test_technical_archive_and_stage_boundary(self):
        self.assertEqual(len(a.SCRIPTS), len(set(a.SCRIPTS)))
        self.assertTrue(all(name.endswith('.py') and Path(name).name == name for name in a.SCRIPTS))
        with self.assertRaises(ValueError):
            a.archive(Path('wrong'))


if __name__ == '__main__':
    unittest.main()
