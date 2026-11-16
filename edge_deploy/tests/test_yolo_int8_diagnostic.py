"""CPU tests for diagnostic aggregation; not TensorRT execution tests."""
from pathlib import Path
import contextlib
import io
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import diagnose_yolo_int8 as d


class DiagnosticTests(unittest.TestCase):
    def test_fixed_shapes_have_distinct_artifacts(self):
        self.assertNotEqual(d.artifact_name('benchmark', 'matched', 'fixed', 0, 'r1'),
                            d.artifact_name('benchmark', 'matched', 'fixed', 1, 'r1'))
        self.assertEqual(d.artifact_name('profile', 'int8', 'mixed', 0, 'r2'),
                         'profile_int8_mixed_r2.json')

    def test_timing_summary(self):
        self.assertEqual(d.summary([1, 2, 3])['mean_ms'], 2)
        self.assertEqual(d.summary([1, 2, 3])['count'], 3)

    def test_invalid_timings_rejected(self):
        for values in ([], [float('nan')], [float('inf')], [-1]):
            with self.assertRaises(ValueError):
                d.summary(values)

    def test_per_inference_not_per_callback(self):
        layers = [{'Name': 'conv', 'LayerType': 'CaskConvolution', 'Format': 'Int8'},
                  {'Name': 'Reformatting CopyNode', 'LayerType': 'Shuffle'}]
        result = d.aggregate([('conv', 2), ('conv', 4), ('Reformatting CopyNode', 1), ('unknown', 1)], layers, 2)
        self.assertEqual(result['summed_layer_ms_per_inference'], 4)
        self.assertEqual(result['category_ms_per_inference']['CaskConvolution'], 3)
        self.assertEqual(result['category_ms_per_inference']['Reformat'], .5)
        self.assertEqual(result['unmapped_callback_names'], ['unknown'])
        self.assertTrue(result['layer_rows'][0]['has_int8_format'])

    def test_profile_empty_or_invalid_rejected(self):
        for rows, count in (([], 1), ([('x', 1)], 0), ([('x', float('nan'))], 1), ([('x', -1)], 1)):
            with self.assertRaises(ValueError):
                d.aggregate(rows, [], count)

    def test_existing_report_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            d.fresh(path)
            path.touch()
            with self.assertRaises(ValueError):
                d.fresh(path)

    def pool_probe(self, drift=False):
        made = []

        class Fake:
            def __init__(self):
                self.shapes = []
                self.closed = False
                self.engine = SimpleNamespace(device_memory_size=10)
                self.number = len(made)
                made.append(self)

            def run(self, _, feeds):
                x = feeds['images']
                self.shapes.append(x.shape)
                return [x.copy() + (1 if drift and self.number else 0)]

            def close(self):
                self.closed = True

            def stats(self):
                return {'errors': 0, 'calls': len(self.shapes)}

        tensors = [{'images': np.zeros((1, 3, 2, w), np.float32)} for w in (2, 3, 2, 3, 2)]
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(stage=Path(directory), variant='original', round='r1', warmup=1, repeat=1)
            with (patch.object(d, 'provenance', return_value=({}, {})),
                  patch.object(d, 'feeds', return_value=tensors),
                  patch.object(d, 'session', side_effect=lambda _: Fake()),
                  patch.object(d, 'snapshot', return_value={}),
                  patch.object(d.q, 'memory', return_value={}),
                  patch.object(d.q, 'digest', return_value='a' * 64),
                  contextlib.redirect_stdout(io.StringIO())):
                if drift:
                    with self.assertRaises(ValueError):
                        d.pooled(args)
                else:
                    d.pooled(args)
            report = json.loads((args.stage / 'pooled_original_mixed_r1.json').read_text())
        self.assertTrue(all(s.closed for s in made))
        return made, report

    def test_pool_preserves_exact_inputs_and_closes_sessions(self):
        made, report = self.pool_probe()
        self.assertEqual(len(made), 3)
        self.assertEqual(len(set(made[0].shapes)), 2)
        self.assertTrue(all(len(set(s.shapes)) == 1 for s in made[1:]))
        self.assertEqual(len(report['rows']), 5)
        self.assertTrue(all(report['exact_equal_after']))
        self.assertEqual(report['status'], 'exact_shape_route_measured_not_deployed')

    def test_pool_rejects_changed_outputs(self):
        _, report = self.pool_probe(drift=True)
        self.assertEqual(report['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
