from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import archive_rec_wide as a


class Tests(unittest.TestCase):
    def test_explicit_technical_allowlist(self):
        names=a.SCRIPTS+a.REPORTS+a.LOGS
        self.assertEqual(len(names),len(set(names)))
        self.assertTrue(all(Path(n).name==n for n in names))
        self.assertFalse(any(n.endswith(('.engine','.cache','.onnx','.npy','.jpg','.so')) for n in names))

    def test_wrong_stage_refused(self):
        with self.assertRaises(ValueError):a.archive(Path('.'))


if __name__=='__main__':unittest.main()
