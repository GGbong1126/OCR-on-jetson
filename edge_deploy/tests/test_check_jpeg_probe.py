import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import check_jpeg_probe as probe


class Tests(unittest.TestCase):
    def result(self, **change):
        row = dict(width=512, height=256, planes=[{}, {}, {}],
                   status='decoded_geometry_checked_not_ocr_validated')
        row.update(change)
        return SimpleNamespace(returncode=0, stdout=json.dumps(row), stderr='')

    def test_geometry(self):
        with patch.object(probe.subprocess, 'run', return_value=self.result()):
            self.assertEqual(probe.check(Path('probe'), Path('input'), 0, (512, 256))['returncode'], 0)
            with self.assertRaises(ValueError):
                probe.check(Path('probe'), Path('input'), 0, (256, 512))

    def test_errors_are_not_success(self):
        with patch.object(probe.subprocess, 'run', return_value=self.result()):
            with self.assertRaises(ValueError):
                probe.check(Path('probe'), Path('input'), 3)
        with patch.object(probe.subprocess, 'run', return_value=self.result(planes=[])):
            with self.assertRaises(ValueError):
                probe.check(Path('probe'), Path('input'), 0, (512, 256))


if __name__ == '__main__':
    unittest.main()
