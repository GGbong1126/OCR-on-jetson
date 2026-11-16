"""Dataset regression safety and CLI; CPU/mock only, no GPU claims."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compare_rec_dataset as tool
import compare_rec_expanded as expanded
import test_compare_rec_expanded as fixture


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.ExpandedTests()
        self.fixture.setUp()
        self.sample = expanded.make_sample(self.fixture.image, self.fixture.row, "test#0")

    def test_original_and_padding_separate_no_plaintext(self):
        row = tool.compare_sample(fixture.FakeRec(), fixture.FakeRec(), self.sample, [1536])
        result = tool.summarize([row], 1)
        self.assertEqual(result['status'], tool.SUCCESS)
        self.assertEqual(row['padded_input_shape'], [1,3,48,1536])
        saved = json.dumps(row)
        for key in ('"reference_text":', '"candidate_text":', '"text":', '"tensor":'):
            self.assertNotIn(key, saved)
        self.assertEqual(tool.summarize([row], 2)['status'], 'invalid_comparison')

    def test_borrowed_output_consumed_and_padding_change_rejected(self):
        class Borrowed(fixture.FakeRec):
            def run_tensor(self, tensor):
                self.raw[:] = [[[0,1],[1,0]]] if tensor.shape[3] == 1100 else [[[1,0],[1,0]]]
                return self.raw
        row = tool.compare_sample(fixture.FakeRec(), Borrowed(), self.sample, [1536])
        s = tool.summarize([row], 1)
        self.assertEqual(s['original_cpu_trt']['status'], tool.SUCCESS)
        self.assertEqual(s['padding_trt']['changed_text_ids'], ['test#0'])
        self.assertEqual(s['status'], tool.CHANGED)

    def test_padding_cannot_hide_original_failure(self):
        gpu = fixture.FakeRec()
        gpu.raw[:] = [[[1,0],[1,0]]]
        row = tool.compare_sample(fixture.FakeRec(), gpu, self.sample, [1536])
        s = tool.summarize([row], 1)
        self.assertEqual(s['padding_trt']['status'], tool.SUCCESS)
        self.assertEqual(s['original_cpu_trt']['status'], tool.CHANGED)
        self.assertEqual(s['status'], tool.CHANGED)

    def test_ascii_space_difference_is_diagnostic_not_permission(self):
        class Spaced(fixture.FakeRec):
            characters = ['blank', 'A', ' ']

            def run_tensor(self, tensor):
                indices = [1, 2] if tensor.shape[3] == 1100 else [1, 0]
                return np.eye(3, dtype=np.float32)[indices][None]
        row = tool.compare_sample(Spaced(), Spaced(), self.sample, [1536])
        s = tool.summarize([row], 1)
        self.assertEqual(s['status'], tool.CHANGED)
        self.assertEqual(s['padding_trt']['ascii_space_only_change_ids'], ['test#0'])

    def test_nonfinite_padding_or_original_invalidates(self):
        for fail_at in (1,2):
            class Nonfinite(fixture.FakeRec):
                def run_tensor(self, tensor):
                    raw = super().run_tensor(tensor)
                    if self.calls == fail_at:
                        raw[:] = np.nan
                    return raw
            row = tool.compare_sample(fixture.FakeRec(), Nonfinite(), self.sample, [1536])
            self.assertEqual(tool.summarize([row], 1)['status'], 'invalid_comparison')
            json.dumps(row, allow_nan=False)

    def cli(self, root):
        args = self.fixture.cli(root)
        with patch('compare_rec_expanded.Recognizer', fixture.FakeRec), patch('compare_rec_expanded.TensorRTRecognizer', fixture.FakeRec), redirect_stdout(io.StringIO()):
            self.assertEqual(expanded.main(args), 0)
        pairs = dict(zip(args[::2], args[1::2]))
        for key in ('--reference-report', '--reference-image', '--output'):
            pairs.pop(key)
        pairs['--acceptance-report'] = str(root/'result.json')
        pairs['--output-dir'] = str(root/'dataset')
        return [v for pair in pairs.items() for v in pair]

    def run_cli(self, args):
        with patch('compare_rec_dataset.Recognizer', fixture.FakeRec), patch('compare_rec_dataset.TensorRTRecognizer', fixture.FakeRec), redirect_stdout(io.StringIO()):
            return tool.main(args)

    def test_cli_checkpoint_success_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.cli(root)
            self.assertEqual(self.run_cli(args), 0)
            r = json.loads((root/'dataset/summary.json').read_text())
            self.assertEqual(r['summary']['compared_count'], 1)
            self.assertEqual(len(r['images']), 1)
            self.assertTrue((root/'dataset/image_0001.json').exists())
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.run_cli(args)

    def test_acceptance_hash_and_record_failure_before_models(self):
        for mode in ('hash','records'):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = self.cli(root)
                path = root/'result.json'
                r = json.loads(path.read_text())
                if mode == 'hash': r['engine_sha256'] = 'wrong'
                else: r['records'].pop()
                path.write_text(json.dumps(r))
                with patch('compare_rec_dataset.Recognizer') as cpu, redirect_stdout(io.StringIO()):
                    self.assertEqual(tool.main(args), 2)
                cpu.assert_not_called()

    def test_preflight_changed_image_hash_or_tensor_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.cli(root)
            scan = json.loads((root/'scan.json').read_text())
            tool.preflight(root, scan, [1536])
            changed = deepcopy(scan)
            changed['images'][0]['branches']['trt']['records'][0]['input_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'tensor differs'):
                tool.preflight(root, changed, [1536])
            (root/'long.png').write_bytes(b'changed')
            with self.assertRaises(ValueError): tool.preflight(root, scan, [1536])

    def test_profile_bucket_and_path_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.cli(root)
            scan = json.loads((root/'scan.json').read_text())
            for buckets in ([1024], [1600], [32,64,128,160,256,512,960,1536]):
                with self.assertRaises(ValueError): tool.preflight(root, scan, buckets)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                tool.load_checked_image(root, {'path':'../escape.png'})

    def test_cleanup_error_invalidates_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.cli(root)
            with patch.object(fixture.FakeRec, 'close', side_effect=RuntimeError('cleanup')):
                self.assertEqual(self.run_cli(args), 2)
            r = json.loads((root/'dataset/summary.json').read_text())
            self.assertEqual(r['status'], 'invalid_comparison')
            self.assertIn('cleanup_error', r)

    def test_interrupt_preserves_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.cli(root)
            with patch('compare_rec_dataset.compare_sample', side_effect=KeyboardInterrupt):
                self.assertEqual(self.run_cli(args), 2)
            r = json.loads((root/'dataset/summary.json').read_text())
            self.assertEqual(r['summary']['compared_count'], 0)
            self.assertEqual(r['status'], 'invalid_comparison')
            self.assertTrue((root/'dataset/image_0001.json').exists())


if __name__ == '__main__':
    unittest.main()
