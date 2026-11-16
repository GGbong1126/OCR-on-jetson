import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import archive_rec_wide_adapter as a

ROLES = ('det', 'rec', 'layout', 'cell', 'yolo', 'enhance', 'ner')


def snapshot(nv=1000, rss=2000):
    return dict(memory=dict(NvMapMemUsed_kib=nv, MemAvailable_kib=2 * 1024**2, SwapFree_kib=1024**2),
                process_rss_kib=rss)


def fixture():
    cases = [dict(index=i, signature={'sha': str(i)}, counters={k: dict(calls=10 if k == 'rec' else 1,
        cpu_fallback_calls=2 if k == 'rec' and i in (14, 15) else 0) for k in ROLES}) for i in range(26)]
    plan = dict(status='prepared', rules=copy.deepcopy(a.v.RULES), cases=cases,
                package={'serve_rec_wide.py': 'factory', 'rec_wide_adapter.py': 'adapter'})
    run = dict(status='completed_offline', rules=copy.deepcopy(a.v.RULES), parent_pid=1,
               startup_environment={}, workers=[])
    reports = {}
    for j, variant in enumerate(a.v.RULES['order']):
        on = variant == 'on_soak'
        run['workers'].append(dict(variant=variant, pid=j + 2, exitcode=0))
        x = dict(variant=variant, status='completed_offline_not_http', rules=copy.deepcopy(a.v.RULES),
                 pid=j + 2, parent_pid=1, spawn_method='spawn', startup_environment={},
                 overlay_loaded=on, all_gpu_sessions_closed=True, loaded_factory_sha256='factory',
                 loaded_adapter_sha256='adapter', rows=[], process_wall_s=100., checkpoints=[],
                 after_warmup=snapshot())
        running = {k: dict(calls=1, errors=0, cpu_fallback=dict(calls=0, errors=0)) for k in ROLES}
        if on:
            running['rec'].update(calls=4, original_execution=dict(calls=1, errors=0),
                                  wide_execution=dict(calls=3, errors=0), wide_profiles={'x': a.v.w.PROFILE})
        x['after_load_execution'] = copy.deepcopy(running)
        for c in range(5 if on else 1):
            for i in (range(26) if c % 2 == 0 else reversed(range(26))):
                counts = a.v.w.expected_counters(cases[i], 'candidate' if on else 'reference')
                wide = cases[i]['counters']['rec']['cpu_fallback_calls'] if on else 0
                x['rows'].append(dict(index=i, cycle=c, signature=cases[i]['signature'], counters=counts,
                                      wide_calls=wide, wall_ms=100.))
                for k, r in counts.items():
                    running[k]['calls'] += r['calls']
                    running[k]['cpu_fallback']['calls'] += r['cpu_fallback_calls']
                if on:
                    running['rec']['original_execution']['calls'] += counts['rec']['calls'] - wide
                    running['rec']['wide_execution']['calls'] += wide
                    if c:
                        x['checkpoints'].append(snapshot())
        x['page_execution'] = copy.deepcopy(running)
        if on:
            x['resources'] = a.v.resource_summary(x['after_warmup'], x['checkpoints'])
            x['boundaries'] = []
            for width in a.v.RULES['boundaries']:
                route = 'original_gpu' if width <= 1536 else 'wide_gpu' if width <= 2304 else 'cpu'
                delta = dict(original=2 if route == 'original_gpu' else 0,
                             wide=2 if route == 'wide_gpu' else 0, cpu=2 if route == 'cpu' else 0)
                x['boundaries'].append(dict(width=width, route=route, call_delta=delta,
                    passed=True, token_sequence_equal=True, allclose_diagnostic=True, repeated_exact=True))
                running['rec']['calls'] += delta['original'] + delta['wide']
                running['rec']['original_execution']['calls'] += delta['original']
                running['rec']['wide_execution']['calls'] += delta['wide']
                running['rec']['cpu_fallback']['calls'] += delta['cpu']
        x['execution'] = running
        reports[variant] = x
    return plan, run, reports


class Tests(unittest.TestCase):
    def test_complete_restart_rollback_and_soak(self):
        result = a.summarize(*fixture())
        self.assertEqual(result['status'], 'offline_adapter_validated_not_http_enabled')
        self.assertEqual(result['total_pages'], 182)
        self.assertEqual(result['workers']['on_soak']['wide_page_calls'], 20)
        self.assertFalse(result['http_validated'])

    def test_incomplete_or_drift_rejected(self):
        for mode in ('order', 'isolation', 'env', 'package', 'close', 'page', 'signature', 'calls',
                     'total', 'boundary', 'counter', 'samples', 'metrics', 'wide', 'rollback', 'error'):
            plan, run, reports = fixture()
            x = reports['on_soak']
            if mode == 'order': run['workers'].reverse()
            if mode == 'isolation': x['pid'] = 1
            if mode == 'env': x['startup_environment'] = {'CUDA_VISIBLE_DEVICES': '-1'}
            if mode == 'package': x['loaded_factory_sha256'] = 'wrong'
            if mode == 'close': x['all_gpu_sessions_closed'] = False
            if mode == 'page': x['rows'].pop()
            if mode == 'signature': x['rows'][0]['signature'] = {}
            if mode == 'calls': x['rows'][0]['counters']['rec']['calls'] += 1
            if mode == 'total': x['page_execution']['rec']['calls'] += 1
            if mode == 'boundary': x['boundaries'][0]['passed'] = False
            if mode == 'counter': x['execution']['rec']['wide_execution']['calls'] += 1
            if mode == 'samples': x['checkpoints'].pop()
            if mode == 'metrics': x['resources']['passed'] = False
            if mode == 'wide': x['rows'][0]['wide_calls'] = 1
            if mode == 'rollback': reports['off_after']['overlay_loaded'] = True
            if mode == 'error': x['execution']['rec']['errors'] = 1
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                a.summarize(plan, run, reports)

    def test_resource_gate_not_silently_passed(self):
        for mode in ('nvmap', 'rss', 'available', 'swap'):
            plan, run, reports = fixture()
            x = reports['on_soak']
            sample = x['checkpoints'][-1]
            if mode == 'nvmap': sample['memory']['NvMapMemUsed_kib'] += 65 * 1024
            if mode == 'rss': sample['process_rss_kib'] += 129 * 1024
            if mode == 'available': sample['memory']['MemAvailable_kib'] = 1000
            if mode == 'swap': sample['memory']['SwapFree_kib'] -= 17 * 1024
            x['resources'] = a.v.resource_summary(x['after_warmup'], x['checkpoints'])
            with self.subTest(mode=mode):
                self.assertEqual(a.summarize(plan, run, reports)['status'], 'offline_resource_gate_not_met')

    def test_archive_no_private_or_engine_files(self):
        self.assertEqual(len(a.SCRIPTS), len(set(a.SCRIPTS)))
        self.assertTrue(all(Path(name).name == name and name.endswith('.py') for name in a.SCRIPTS))
        self.assertTrue(all(name.endswith('.json') for name in a.REPORTS))

    def test_wrong_stage_refused(self):
        with self.assertRaises(ValueError):
            a.archive(Path('wrong'))


if __name__ == '__main__':
    unittest.main()
