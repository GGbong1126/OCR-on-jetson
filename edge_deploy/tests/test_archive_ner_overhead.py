import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import archive_ner_overhead as a
import test_benchmark_ner_protected as benchmark_tests


class Tests(unittest.TestCase):
    def test_allowlist(self):
        self.assertEqual(len(a.FILES),len(set(a.FILES)))
        self.assertEqual(len(a.REPORTS),6)
        for name in a.FILES:
            self.assertEqual(Path(name).name,name);self.assertIn(Path(name).suffix,('.py','.json','.log'))
        self.assertNotIn('ner_protected.engine',a.FILES)

    def test_safe_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);file=root/'test.json';file.write_text('{}')
            r=a.safe_files({'test.json':file},root)
            self.assertEqual(r[0]['bytes'],2);self.assertEqual(r[0]['sha256'],a.q.digest(file))
            for name,path,base in (('../test.json',file,root),('test.json',file,root/'elsewhere'),('test.json',root/'missing',root)):
                with self.assertRaises(ValueError):a.safe_files({name:path},base)

    def test_report_hash_and_provenance(self):
        plan,run,reports=benchmark_tests.Tests.fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run['plan_sha256']='p'
            for (r,v),report in reports.items():(root/('benchmark_%s_%s.json'%(v,r))).write_text(json.dumps(report))
            run['benchmark_reports_sha256']={name:a.q.digest(root/name) for name in a.REPORTS[2:]}
            (root/'plan.json').write_text(json.dumps(plan));(root/'run.json').write_text(json.dumps(run))
            with patch.object(a.b,'metadata',return_value={'plan_sha256':'p'}):
                self.assertEqual(a.validate_reports(root)['status'],'performance_gate_passed_not_promoted')
                path=root/a.REPORTS[2];path.write_text(path.read_text()+' ')
                with self.assertRaises(ValueError):a.validate_reports(root)
            with patch.object(a.b,'metadata',return_value={'plan_sha256':'wrong'}):
                with self.assertRaises(ValueError):a.validate_reports(root)


if __name__=='__main__':unittest.main()
