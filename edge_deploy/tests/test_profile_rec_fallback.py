import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import profile_rec_fallback as d

PROFILE = {'x': [[1, 3, 48, 32], [1, 3, 48, 160], [1, 3, 48, 1536]]}
ROLES = ('det', 'rec', 'yolo', 'layout', 'cell', 'enhance', 'ner')


class Tests(unittest.TestCase):
    @staticmethod
    def fixture():
        cases, pages = [], []
        for i in range(26):
            counts = {k: dict(calls=1, cpu_fallback_calls=int(k == 'rec' and i in (14, 15))) for k in ROLES}
            cases.append(dict(index=i, image_sha256=str(i//2), image_type=(3, 1)[i % 2],
                              signature={'sha': str(i)}, counters=counts))
        for phase in d.RULES['phases']:
            for i in d.measured_indices(phase):
                row = copy.deepcopy(cases[i])
                calls = []
                for route, width in [('gpu', 160)] + ([('cpu', 2000)] if i in (14, 15) else []):
                    cold = phase == 'warmup' and i == 14 and route == 'cpu'
                    shapes = {'x': [1, 3, 48, width]}
                    call = dict(call_index=len(calls), route=route, input_shapes=shapes, input_dtypes={'x': 'float32'},
                                output_shape=[1, width//8, 18385], output_dtype='float32',
                                outside_dimensions=d.outside_dimensions(shapes, PROFILE), cold_cpu=cold,
                                route_wall_ms=20., cpu_creation_ms=5. if cold else 0.)
                    if route == 'cpu':
                        call.update(input_sha256={'x': 'input'}, output_sha256='out', ctc_sha256='ctc', ctc_length=10)
                    calls.append(call)
                row.update(phase=phase, wall_ms=100., calls=calls)
                pages.append(row)
        before = {k: dict(calls=1, errors=0, cpu_fallback=dict(calls=0, errors=0)) for k in ROLES}
        before['rec']['profiles'] = copy.deepcopy(PROFILE)
        after = copy.deepcopy(before)
        for k in ROLES:
            after[k]['calls'] += 78
        after['rec']['cpu_fallback']['calls'] = 6
        return (dict(status='prepared', rules=copy.deepcopy(d.RULES), cases=cases),
                dict(status='diagnosed', rules=copy.deepcopy(d.RULES), profiles=copy.deepcopy(PROFILE), pages=pages,
                     cpu_creations=[dict(wall_ms=5., providers=['CPUExecutionProvider'], threads=2,
                                         execution_mode='ExecutionMode.ORT_SEQUENTIAL')],
                     after_setup_execution=before, execution=after))

    def test_summary_and_nested_timing(self):
        result = d.summarize(*self.fixture())
        self.assertEqual(result['status'], 'fallback_attributed_not_optimized')
        self.assertEqual(result['cpu_shapes'][0]['timing']['count'], 4)
        self.assertAlmostEqual(result['rounds'][0]['cpu_fraction_of_all_pages'], 40/2600)
        self.assertFalse(result['accepted_for_release'])

    def test_reject_corrupted_evidence(self):
        for kind in ('coverage', 'business', 'counters', 'route', 'shape', 'output', 'hash_missing',
                     'negative', 'nested', 'cold', 'threads', 'provider', 'init_time', 'total', 'profile', 'error'):
            plan, report = self.fixture()
            row = next(v for v in report['pages'] if v['phase'] == 'r2' and v['index'] == 14)
            call = row['calls'][-1]
            if kind == 'coverage': report['pages'].pop()
            if kind == 'business': row['signature'] = {}
            if kind == 'counters': row['counters']['rec']['calls'] = 2
            if kind == 'route': call['route'] = 'gpu'
            if kind == 'shape': call['input_shapes']['x'][3] = 1000
            if kind == 'output': call['ctc_sha256'] = 'drift'
            if kind == 'hash_missing': del call['input_sha256']
            if kind == 'negative': call['route_wall_ms'] = -1.
            if kind == 'nested': call['route_wall_ms'] = 1000.
            if kind == 'cold': call['cold_cpu'] = True
            if kind == 'threads': report['cpu_creations'][0]['threads'] = 8
            if kind == 'provider': report['cpu_creations'][0]['providers'] = ['CUDAExecutionProvider']
            if kind == 'init_time': report['cpu_creations'][0]['wall_ms'] = 6.
            if kind == 'total': report['execution']['rec']['cpu_fallback']['calls'] += 1
            if kind == 'profile': report['profiles']['x'][2][3] += 1
            if kind == 'error': report['execution']['rec']['cpu_fallback']['errors'] = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                d.summarize(plan, report)

    def test_profile_dimensions_inclusive(self):
        for width in (32, 1536):
            self.assertEqual(d.outside_dimensions({'x': [1, 3, 48, width]}, PROFILE), [])
        for axis, value in ((0, 2), (1, 1), (2, 49), (3, 1537), (3, 31)):
            shape = [1, 3, 48, 160]; shape[axis] = value
            r = d.outside_dimensions({'x': shape}, PROFILE)
            self.assertEqual((r[0]['axis'], r[0]['value']), (axis, value))
        for shapes in ({'bad': [1, 3, 48, 160]}, {'x': [1, 160]}):
            with self.assertRaises(ValueError): d.outside_dimensions(shapes, PROFILE)

    def test_probe_preserves_tensors_and_output(self):
        output = np.array([[[0., 1., 0.], [0., 1., 0.], [1., 0., 0.]]], dtype=np.float32)
        received = []
        class CPU:
            def get_session_options(self):
                return SimpleNamespace(intra_op_num_threads=2, execution_mode='ExecutionMode.ORT_SEQUENTIAL')
            def get_providers(self): return ['CPUExecutionProvider']
        session = SimpleNamespace(gpu=SimpleNamespace(profiles=PROFILE), cpu=None, onnx=Path('rec.onnx'))
        def original(names, feeds):
            received.append(feeds['x'])
            if feeds['x'].shape[-1] > 1536 and session.cpu is None:
                session.cpu = probe.create_cpu(session.onnx)
            return [output]
        session.run = original
        probe = d.Probe(session, lambda path: CPU())
        small = np.zeros([1, 3, 48, 160], dtype=np.float32)
        wide = np.zeros([1, 3, 48, 2000], dtype=np.float32)
        for x in (small, wide, wide):
            self.assertIs(probe.call(None, {'x': x})[0], output)
            self.assertIs(received[-1], x)
        probe.finish_page()
        self.assertEqual([r['route'] for r in probe.rows], ['gpu', 'cpu', 'cpu'])
        self.assertEqual([r['cold_cpu'] for r in probe.rows], [False, True, False])
        self.assertEqual(probe.rows[-1]['ctc_length'], 1)
        self.assertEqual(len(probe.creations), 1)
        self.assertEqual(probe.pending, [])
        self.assertNotIn('tokens', probe.rows[-1])

    def test_runtime_error_not_retried(self):
        def run(names, feeds): raise RuntimeError('failed')
        session = SimpleNamespace(gpu=SimpleNamespace(profiles=PROFILE), cpu=None, onnx=Path('rec.onnx'), run=run)
        probe = d.Probe(session, lambda path: None)
        with self.assertRaises(RuntimeError): probe.call(None, {'x': np.zeros([1, 3, 48, 160])})
        self.assertEqual(probe.rows, [])


if __name__ == '__main__': unittest.main()
