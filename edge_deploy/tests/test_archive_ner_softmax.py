from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import archive_ner_softmax as a


class Tests(unittest.TestCase):
    def test_technical_allowlist(self):
        self.assertEqual(len(a.FILES),len(set(a.FILES)))
        for name in a.FILES:
            self.assertEqual(Path(name).name,name)
            self.assertIn(Path(name).suffix,('.json','.py','.cu','.log'))
        self.assertNotIn('softmax.engine',a.FILES);self.assertNotIn('libner_softmax.so',a.FILES)

    def test_source_and_tests_included(self):
        self.assertIn('ner_mask_softmax.cu',a.SCRIPTS)
        self.assertIn('test_ner_softmax_probe.py',a.SCRIPTS)
        self.assertIn('test_archive_ner_softmax.py',a.SCRIPTS)

    def test_summary_fail_closed(self):
        with self.assertRaises(ValueError):a.summarize({'status':'failed'},{},{},{})


if __name__=='__main__':unittest.main()
