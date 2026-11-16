from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import archive_rec_fallback as a
import test_profile_rec_fallback as fixtures


class ArchiveTests(unittest.TestCase):
    def test_summary_recomputed_and_drift_rejected(self):
        plan, report = fixtures.Tests.fixture()
        plan['proof'] = report['proof'] = 'valid'
        report['plan_sha256'] = 'sha'
        report['summary'] = a.d.summarize(plan, report)
        def read(path): return plan if path.name == 'plan.json' else report
        with patch.object(a.d, 'provenance', return_value={'proof': 'valid'}), patch.object(a.n, 'read', side_effect=read), patch.object(a.q, 'digest', return_value='sha'):
            self.assertEqual(a.validate_reports(Path('.')), report['summary'])
            report['summary']['measured_pages'] = 0
            with self.assertRaises(ValueError): a.validate_reports(Path('.'))

    def test_technical_allowlist_and_stage(self):
        self.assertEqual(len(a.SCRIPTS), len(set(a.SCRIPTS)))
        self.assertFalse(any(name.endswith(('.engine', '.onnx', '.so', '.npy', '.jpg')) for name in a.SCRIPTS + a.REPORTS))
        with self.assertRaises(ValueError): a.archive(Path('.'))


if __name__ == '__main__': unittest.main()
