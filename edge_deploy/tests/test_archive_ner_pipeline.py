import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import archive_ner_pipeline as a
import test_benchmark_ner_pipeline as fixtures


class Tests(unittest.TestCase):
    def test_allowlist(self):
        self.assertEqual(len(a.FILES),len(set(a.FILES)))
        self.assertEqual(len(a.REPORTS),6)
        for name in a.FILES:
            self.assertEqual(Path(name).name,name);self.assertIn(Path(name).suffix,('.py','.json','.log'))

    def test_hashes_provenance_and_call_consistency(self):
        plan,run,reports=fixtures.Tests.fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run['plan_sha256']='p'
            def save():
                for (r,v),report in reports.items():(root/('pipeline_%s_%s.json'%(v,r))).write_text(json.dumps(report))
                run['reports_sha256']={name:a.q.digest(root/name) for name in a.REPORTS[2:]}
                (root/'plan.json').write_text(json.dumps(plan));(root/'run.json').write_text(json.dumps(run))
            save()
            with patch.object(a.p,'metadata',return_value={'plan_sha256':'p'}):
                self.assertEqual(a.validate_reports(root)['model_call_counts_consistent_cases'],26)
                path=root/a.REPORTS[2];path.write_text(path.read_text()+' ')
                with self.assertRaises(ValueError):a.validate_reports(root)
                save();r=reports[('r1','candidate')];r['rows'][0]['counters']['enhance']['calls']+=1;r['execution']['enhance']['calls']+=1;save()
                with self.assertRaisesRegex(ValueError,'invocation drift'):a.validate_reports(root)
            with patch.object(a.p,'metadata',return_value={'plan_sha256':'wrong'}):
                with self.assertRaises(ValueError):a.validate_reports(root)


if __name__=='__main__':unittest.main()
