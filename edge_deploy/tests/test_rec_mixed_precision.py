from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import rec_mixed_precision as m


class Tests(unittest.TestCase):
    def test_regions_are_complementary(self):
        for i in range(768):
            self.assertNotEqual(m.selected(i, 355, 'tail_fp32'), m.selected(i, 355, 'prefix_fp32'))
        self.assertTrue(m.selected(355, 355, 'tail_fp32'))
        self.assertFalse(m.selected(354, 355, 'tail_fp32'))

    def test_bad_policy_rejected(self):
        with self.assertRaises(ValueError):
            m.selected(1, 355, 'anything')

    def test_probe_gate_rejects_drift_or_wrong_build(self):
        # Fake files only; no Jetson/imported runtime required.
        for status, repeat, digest in [('failed_case_still_changed', True, 'expected'),
                                        ('failed_case_recovered_not_general_acceptance', False, 'expected'),
                                        ('failed_case_recovered_not_general_acceptance', True, 'different')]:
            class Stage:
                def __truediv__(self, key):
                    class File:
                        def read_text(self):
                            return json.dumps({'status': status, 'repeat_exact': repeat, 'build_sha256': digest})
                    return File()
            with patch.object(m.q, 'digest', return_value='expected'):
                with self.assertRaises(ValueError):
                    m.require_probe(Stage())


if __name__ == '__main__':
    unittest.main()
