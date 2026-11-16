import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import soak_rec_corpus as s
import archive_rec_soak as a


def snapshot(t, nv=4000000, rss=3000000, free=6000000, swap=7000000):
    return dict(monotonic_s=t, process_rss_kib=rss,
                memory=dict(NvMapMemUsed_kib=nv, MemAvailable_kib=free, SwapFree_kib=swap))


def fixture():
    counts = {k: dict(calls=1, cpu_fallback_calls=0) for k in ('rec', 'det', 'layout', 'cell', 'yolo', 'enhance', 'ner')}
    counts['rec']['calls'] = 2
    cases = [dict(index=i, image_sha256='%064x' % (i // 2), image_type=(3, 1)[i % 2],
                  signature={'business_sha256': str(i)}, counters=copy.deepcopy(counts), wide_calls=1)
             for i in range(90)]
    plan = dict(status='prepared', rules=copy.deepcopy(s.RULES), cases=cases,
                package={'serve_rec_wide.py': 'factory', 'rec_wide_adapter.py': 'adapter'})
    initial = {k: dict(calls=0, errors=0, cpu_fallback=dict(calls=0, errors=0)) for k in counts}
    initial['rec'].update(calls=3, wide_execution={'calls': 3})
    execution = copy.deepcopy(initial)
    for k in counts:
        execution[k]['calls'] += counts[k]['calls'] * 180
    execution['rec']['wide_execution']['calls'] += 180
    warm, end = snapshot(100), snapshot(1001)
    worker = dict(status='completed_bounded_soak_not_http', rules=copy.deepcopy(s.RULES),
        pid=20, parent_pid=10, spawn_method='spawn', startup_environment={},
        all_gpu_sessions_closed=True, factory_sha256='factory', adapter_sha256='adapter',
        initial_execution=initial, execution=execution,
        cycles=[dict(cycle=0, wall_s=100, after_cycle=warm), dict(cycle=1, wall_s=901, after_cycle=end)],
        rows=[dict(index=i, cycle=cy, signature=cases[i]['signature'], counters=copy.deepcopy(counts),
                   wide_calls=1, diagnostic_wall_s=1) for cy in range(2) for i in s.indices(cy)],
        after_warmup=warm, postwarm_s=901, checkpoints=[snapshot(100 + 901 * i / 90) for i in range(1, 91)])
    worker['resources'] = s.resources(warm, worker['checkpoints'])
    run = dict(status='completed', rules=copy.deepcopy(s.RULES), worker_exitcode=0,
        worker_pid=20, parent_pid=10, startup_environment={},
        samples=[snapshot(t) for t in range(105, 1001, 5)], before_worker=snapshot(0), after_exit=snapshot(1010))
    return plan, run, worker


class Tests(unittest.TestCase):
    def test_cli_summary_serialization(self):
        result = a.summarize(*fixture())
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            a.print_result({'members': 47}, result)
        self.assertEqual([json.loads(line) for line in stream.getvalue().splitlines()], [{'members': 47}, result])

    def test_happy_path(self):
        result = a.summarize(*fixture())
        self.assertEqual(result['status'], 'bounded_soak_passed_not_http')
        self.assertEqual((result['total_pages'], result['postwarm_pages'], result['wide_page_calls']), (180, 90, 180))

    def test_union_frozen_45(self):
        old13 = [dict(image_sha256='%064x' % (i // 2), image_type=(3, 1)[i % 2], signature={},
                      counters={'rec': dict(calls=2, cpu_fallback_calls=1)}) for i in range(26)]
        new32 = [dict(image_sha256='%064x' % (13 + i // 2), image_type=(3, 1)[i % 2], signature={},
                     counters={'rec': dict(calls=3, cpu_fallback_calls=0)}, rec_calls=[dict(route='wide_gpu')]) for i in range(64)]
        original = copy.deepcopy(old13)
        result = s.union_cases(old13, new32)
        self.assertEqual(old13, original)
        self.assertEqual(len(result), 90)
        self.assertTrue(all(r['wide_calls'] == 1 and r['counters']['rec']['calls'] == 3 for r in result))
        for bad in (new32[:-1], [new32[1]] + new32[1:], [dict(new32[0], image_sha256=old13[0]['image_sha256'])] + new32[1:]):
            with self.assertRaises(ValueError):
                s.union_cases(old13, bad)

    def test_protocol_failures(self):
        for target, key, val in [('run', 'worker_pid', 21), ('run', 'worker_exitcode', 1),
                ('worker', 'status', 'failed'), ('worker', 'all_gpu_sessions_closed', False),
                ('worker', 'factory_sha256', 'changed'), ('worker', 'adapter_sha256', 'changed'),
                ('worker', 'spawn_method', 'fork'), ('worker', 'startup_environment', {'changed': True}),
                ('worker', 'parent_pid', 20)]:
            with self.subTest(key=key):
                plan, run, worker = fixture()
                {'run': run, 'worker': worker}[target][key] = val
                with self.assertRaises(ValueError):
                    a.summarize(plan, run, worker)

    def test_per_page_drift(self):
        for key, val in [('signature', {}), ('counters', {}), ('wide_calls', 0), ('index', 2),
                         ('diagnostic_wall_s', float('nan'))]:
            plan, run, worker = fixture()
            worker['rows'][0][key] = val
            with self.assertRaises(ValueError):
                a.summarize(plan, run, worker)
        with self.assertRaises(ValueError):
            s.check_page({'signature': {}, 'counters': {}, 'wide_calls': 0}, {}, {}, 0, False)

    def test_coverage_and_duration(self):
        for mutate in (lambda p, r, w: w['rows'].pop(), lambda p, r, w: w['checkpoints'].pop(),
                       lambda p, r, w: w.update(postwarm_s=899),
                       lambda p, r, w: w['cycles'][-1]['after_cycle'].update(monotonic_s=999),
                       lambda p, r, w: p['cases'][0].update(image_type=1),
                       lambda p, r, w: w['checkpoints'][0].update(monotonic_s=0)):
            args = fixture()
            mutate(*args)
            with self.assertRaises(ValueError):
                a.summarize(*args)

    def test_errors_and_counters(self):
        for mutate in (lambda w: w['execution']['rec'].update(errors=1),
                       lambda w: w['execution']['ner'].update(calls=1),
                       lambda w: w['execution']['rec']['wide_execution'].update(calls=3),
                       lambda w: w['initial_execution']['rec']['wide_execution'].update(calls=4)):
            plan, run, worker = fixture()
            mutate(worker)
            with self.assertRaises(ValueError):
                a.summarize(plan, run, worker)

    def test_resources_no_invented_pass(self):
        for key, value in [('NvMapMemUsed_kib', 4000000 + 65 * 1024),
                           ('MemAvailable_kib', 1024), ('SwapFree_kib', 7000000 - 17 * 1024)]:
            plan, run, worker = fixture()
            worker['checkpoints'][-1]['memory'][key] = value
            worker['resources'] = s.resources(worker['after_warmup'], worker['checkpoints'])
            self.assertEqual(a.summarize(plan, run, worker)['status'], 'bounded_soak_resource_not_accepted')
        plan, run, worker = fixture()
        worker['checkpoints'][-1]['process_rss_kib'] += 129 * 1024
        worker['resources'] = s.resources(worker['after_warmup'], worker['checkpoints'])
        self.assertFalse(a.summarize(plan, run, worker)['resources']['passed'])

    def test_periodic_gaps_and_exit_resource(self):
        plan, run, worker = fixture()
        run['samples'] = run['samples'][5:]
        with self.assertRaises(ValueError):
            a.summarize(plan, run, worker)
        plan, run, worker = fixture()
        run['after_exit']['memory']['NvMapMemUsed_kib'] += 65 * 1024
        self.assertEqual(a.summarize(plan, run, worker)['status'], 'bounded_soak_resource_not_accepted')

    def test_allowlist_and_limits(self):
        self.assertEqual(len(a.SCRIPTS), len(set(a.SCRIPTS)))
        self.assertTrue(all(Path(name).name == name and name.endswith('.py') for name in a.SCRIPTS))
        self.assertEqual(s.STAGE.name, 'rec_soak_e126_r1')
        self.assertGreaterEqual(s.RULES['minimum_soak_s'], 900)
        self.assertEqual(s.indices(1), list(reversed(s.indices(0))))


if __name__ == '__main__':
    unittest.main()
