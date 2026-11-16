import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import archive_ner_norm as a


class Tests(unittest.TestCase):
    @staticmethod
    def fixture():
        images=[dict(image_sha256=str(i//2),image_type=3 if i%2==0 else 1) for i in range(6)]
        plan=dict(status='prepared',rules=a.p.RULES,norms=[{}]*25,prior_real_rows=images)
        build=dict(status='built_not_validated',variants={k:dict(flags=f,plugin_serialization_roundtrip=True) for k,f in (('fp32',1024),('half_boundary',1025))})
        metric=dict(all_finite=True,allclose=True,mae=0.,max_abs_error=0.)
        variants={k:dict(repeat_exact=True,reference=metric.copy(),original_output=metric.copy()) for k in build['variants']}
        real=[dict(r,norm_index=j,input_abs_max=1.,variants=copy.deepcopy(variants)) for r in images for j in range(25)]
        synthetic=[dict(length=l,kind=k,variants=copy.deepcopy(variants)) for l in (2,128,512) for k in ('normal','constant','near_constant','alternating_large')]
        check=dict(status='isolated_passed_not_integrated',rules=a.p.RULES,real_rows=real,synthetic_rows=synthetic,inspection={k:{} for k in variants},
            execution={k:dict(calls=324,errors=0) for k in variants},real_same_input_passed=True,original_fp32_control_passed=True,synthetic_same_input_passed=True)
        return [plan,build,check,{k:{} for k in variants}]

    def run_summary(self,f):
        with patch.object(a.p,'inspect',return_value={}):return a.summarize(*f)

    def test_counts_no_promotion(self):
        r=self.run_summary(self.fixture());self.assertEqual(r['plugin_calls'],648);self.assertFalse(r['full_ner_candidate']);self.assertFalse(r['accepted_for_release'])

    def test_missing_norm_or_synthetic_rejected(self):
        for key in ('real_rows','synthetic_rows'):
            f=self.fixture();f[2][key].pop()
            with self.assertRaises(ValueError):self.run_summary(f)

    def test_false_pass_rejected(self):
        f=self.fixture();f[2]['real_rows'][0]['variants']['fp32']['reference']['allclose']=False
        with self.assertRaises(ValueError):self.run_summary(f)

    def test_failure_retained(self):
        f=self.fixture();f[2]['synthetic_rows'][2]['variants']['fp32']['reference']['allclose']=False
        f[2].update(status='isolated_numeric_not_accepted',synthetic_same_input_passed=False)
        r=self.run_summary(f);self.assertEqual(r['status'],'isolated_numeric_not_accepted');self.assertEqual(len(r['variants']['fp32']['failed_synthetic']),1)

    def test_errors_and_flags_rejected(self):
        for mutate in (lambda f:f[2]['execution']['fp32'].update(errors=1),lambda f:f[1]['variants']['fp32'].update(flags=1025)):
            f=self.fixture();mutate(f)
            with self.assertRaises(ValueError):self.run_summary(f)


if __name__=='__main__':unittest.main()
