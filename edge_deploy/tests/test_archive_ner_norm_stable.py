import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import archive_ner_norm_stable as a


class Tests(unittest.TestCase):
    @staticmethod
    def fixture():
        m=dict(allclose=True,all_finite=True,mae=1e-8,max_abs_error=1e-7)
        def row():
            r={k:m.copy() for k in ('original_vs_fp64','old_vs_fp64','candidate_vs_fp64','candidate_vs_original','emulator_vs_candidate')}
            r.update(repeat_exact=True,emulator_exact=True,input_sha256='hash');r['old_vs_fp64']['mae']=.01
            return r
        real=[dict(row(),image_sha256=str(i//2),image_type=3 if i%2==0 else 1,norm_index=j) for i in range(6) for j in range(25)]
        syn=[dict(row(),length=l,kind=k) for l,k,_ in a.p.synthetic()]
        for r in syn:
            if r['kind']=='near_constant':r['candidate_vs_original']['allclose']=False
        plan=dict(status='prepared',rules=a.p.RULES,synthetic=[{k:r[k] for k in ('length','kind','input_sha256')} for r in syn])
        build=dict(status='built_not_validated',flags=1024,serialization_roundtrip=True)
        check=dict(status='isolated_stability_passed_not_integrated',rules=a.p.RULES,real_rows=real,synthetic_rows=syn,inspection={},execution={'old':dict(calls=180,errors=0),'candidate':dict(calls=360,errors=0)})
        return [plan,build,check,{}]

    def run_summary(self,f):
        with patch.object(a.p.p,'inspect',return_value={}):return a.summarize(*f)

    def test_independent_gate_preserves_legacy_failures(self):
        r=self.run_summary(self.fixture());self.assertEqual(len(r['legacy_cpu_synthetic_failed']),3)
        self.assertEqual(r['status'],'isolated_stability_passed_not_integrated');self.assertFalse(r['accepted_for_release'])

    def test_missing_and_duplicate_rejected(self):
        for mutate in (lambda f:f[2]['real_rows'].pop(),lambda f:f[2]['real_rows'].__setitem__(1,copy.deepcopy(f[2]['real_rows'][0])),lambda f:f[2]['synthetic_rows'].pop()):
            f=self.fixture();mutate(f)
            with self.assertRaises(ValueError):self.run_summary(f)

    def test_counter_or_error_rejected(self):
        f=self.fixture();f[2]['execution']['candidate']['calls']=359
        with self.assertRaises(ValueError):self.run_summary(f)
        f=self.fixture();f[2]['execution']['old']['errors']=1
        with self.assertRaises(ValueError):self.run_summary(f)

    def test_false_pass_rejected(self):
        f=self.fixture();f[2]['real_rows'][0]['candidate_vs_fp64']['allclose']=False
        with self.assertRaises(ValueError):self.run_summary(f)

    def test_unstable_emulation_rejected(self):
        f=self.fixture();f[2]['real_rows'][0]['emulator_vs_candidate']['allclose']=False
        with self.assertRaises(ValueError):self.run_summary(f)


if __name__=='__main__':unittest.main()
