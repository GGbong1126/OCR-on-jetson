import ast
import json
import os
from pathlib import Path
import tempfile
import unittest

from analyze import ORDER, analyze
from experiment import MODES, Recorder, delta, transform

SOURCE = Path(os.environ.get('AUTOOCR_ABLATION_SOURCE', Path(__file__).resolve().parents[1]/'batch_ocr'))/'multistream/bridge.py'


class Tests(unittest.TestCase):
    def test_each_intervention_matches_frozen_source_once_and_compiles(self):
        source = SOURCE.read_text(encoding='utf-8')
        for mode in MODES:
            result = transform(source, mode)
            ast.parse(result)
            self.assertEqual(result == source, mode in ('sync', 'full', 'blocking_submit'))

    def test_mismatched_source_rejected(self):
        for mode in ('one_inflight', 'no_fanout', 'inline_ctc'):
            with self.assertRaises(AssertionError):
                transform('', mode)

    def test_recorder_return_exception_and_snapshot_delta(self):
        r = Recorder(); before = r.snapshot()
        self.assertEqual(r.call('test', lambda: 42), 42)
        with self.assertRaises(ValueError):
            r.call('error', lambda: int('invalid'))
        after = r.snapshot(); rows = delta(before, after)
        self.assertEqual(rows['test']['calls'], 1)
        self.assertEqual(rows['error']['calls'], 1)
        self.assertGreaterEqual(rows['test']['wall_ms'], 0)
        self.assertEqual(before, {})

    def test_analysis_requires_quality_and_actual_restoration(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as name:
            folder=Path(name);arms=[]
            for i,mode in enumerate(ORDER):
                seconds=300 if mode=='sync' else 250
                arms.append(dict(seconds=seconds,mode=mode))
                p=folder/('%02d_%s.json'%(i,mode))
                r=dict(closed=True,status='full_python_ocr_workers_consistent_not_http_or_gt',resource_gate=dict(complete=True),
                       passes=[dict(full_data_equal_count=26,warmup=j==0,seconds=seconds/3) for j in range(4)],
                       rows=[dict(image_sha256='synthetic',image_type=3,candidate_sha256='synthetic',reference_rec_widths=[32],candidate_rec_widths=[32])],workers={})
                p.write_text(json.dumps(r))
                label='rec.context.execute_v2' if mode=='sync' else 'rec.context.execute_async_v2'
                p.with_suffix('.host.json').write_text(json.dumps(dict(completed=True,mode=mode,snapshot_count=2,measured={label:dict(calls=3438)})))
            control=dict(status='ablation_complete_e162_restored',restored=True,measurements_completed=True,
                         order=list(ORDER),arms=arms,restored_checks=[dict(equal=True)],restored_master_pid=12345)
            (folder/'control_r3.json').write_text(json.dumps(control))
            result=analyze(folder)
            self.assertAlmostEqual(result['throughput_gain_percent'],20)
            self.assertEqual(result['seconds_saved'],50)
            control['restored']=False
            (folder/'control_r3.json').write_text(json.dumps(control))
            with self.assertRaises(AssertionError): analyze(folder)


if __name__ == '__main__':
    unittest.main()
