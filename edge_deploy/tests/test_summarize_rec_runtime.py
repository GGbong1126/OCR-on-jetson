from pathlib import Path
import copy
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import summarize_rec_runtime as m


def reports():
    result = []
    for variant in ('reference', 'skip_same'):
        for round_ in ('r1', 'r2'):
            row = {'variant': variant, 'round': round_,
                   'status': 'diagnosed_exact_same_engine_not_gt',
                   'input_shapes': [[1, 3, 48, 32], [1, 3, 48, 160]],
                   'input_sha256': ['a', 'b'], 'baseline_output_sha256': [['c'], ['d']],
                   'warmup_cycles': 1, 'repeat_cycles': 1, 'execution': {'errors': 0},
                   'profile_probe': {}, 'sequence_cache_simulation': [], 'context_device_bytes': 10}
            for key in ('image_sha256', 'engine_sha256', 'deployment_sha256', 'backend_sha256', 'script_sha256'):
                row[key] = 'frozen'
            row['bare_rows'] = [{'cycle': 0, 'index': i, 'wall_ms': 10} for i in range(2)]
            row['patterns'] = [{'name': name, 'shape_changes': 2, 'malloc_count': 0,
                                'means_ms': {'call_ms': 10},
                                'rows': [{'output_exact': True, 'index': i, 'call_ms': 10,
                                          'shape_api_calls': 1, 'free_count': 0} for i in range(2)]}
                               for name in ('repeat_a', 'repeat_b', 'alternate_ab', 'alternate_ba', 'original_sequence')]
            result.append(row)
    return result


class Tests(unittest.TestCase):
    def test_valid_report_and_missing_or_duplicate_run(self):
        data = reports()
        self.assertEqual(m.summarize(data)['bare_timing_calls'], 8)
        for bad in (data[:3], [data[0]] * 4):
            with self.assertRaises(ValueError):
                m.summarize(bad)

    def test_changed_provenance_or_backend_error(self):
        for key in ('input_sha256', 'baseline_output_sha256', 'script_sha256'):
            data = reports()
            data[1][key] = 'changed'
            with self.assertRaises(ValueError):
                m.summarize(data)
        data = reports()
        data[2]['execution']['errors'] = 1
        with self.assertRaises(ValueError):
            m.summarize(data)

    def test_invalid_output_and_incomplete_diagnostic(self):
        data = reports()
        data[0]['patterns'][0]['rows'][0]['output_exact'] = False
        with self.assertRaises(ValueError):
            m.summarize(data)
        data = reports()
        data[0]['patterns'].pop()
        with self.assertRaises(ValueError):
            m.summarize(data)

    def test_nonfinite_or_reordered_bare_calls(self):
        for kind in ('nan', 'reverse'):
            data = copy.deepcopy(reports())
            if kind == 'nan':
                data[0]['bare_rows'][0]['wall_ms'] = float('nan')
            else:
                data[0]['bare_rows'].reverse()
            with self.assertRaises(ValueError):
                m.summarize(data)


if __name__ == '__main__':
    unittest.main()
