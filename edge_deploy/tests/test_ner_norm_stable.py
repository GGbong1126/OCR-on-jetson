from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import ner_norm_stable as s


class Tests(unittest.TestCase):
    def test_centered_preserves_constant(self):
        x=np.full((1,2,768),1000.,np.float32);g=np.ones(768,np.float32);b=np.arange(768,dtype=np.float32)
        self.assertTrue(np.array_equal(s.centered(x,g,b,1e-12),np.broadcast_to(b,x.shape)))

    def test_synthetic_math_and_reproducibility(self):
        cases=list(s.synthetic());self.assertEqual(len(cases),30)
        for (l,k,x),(ll,kk,y) in zip(cases,s.synthetic()):
            self.assertEqual((l,k),(ll,kk));self.assertTrue(np.array_equal(x,y))
            g=np.ones(768,np.float32);b=np.zeros(768,np.float32)
            self.assertTrue(s.metric(s.d.precise(x,g,b,1e-12),s.centered(x,g,b,1e-12))['allclose'],(l,k))

    def test_near_constant_error_improvement(self):
        for l,k,x in s.synthetic():
            if k!='near_constant':continue
            g=np.ones(768,np.float32);b=np.zeros(768,np.float32);ref=s.d.precise(x,g,b,1e-12)
            before=s.metric(ref,s.d.emulate(x,g,b,1e-12))['mae'];after=s.metric(ref,s.centered(x,g,b,1e-12))['mae']
            self.assertLess(after,.01*before)

    def test_metrics_use_high_precision_reference(self):
        x=np.ones((1,2),np.float64);y=x.astype(np.float32)
        self.assertTrue(s.metric(x,y)['allclose'])
        with self.assertRaises(ValueError):s.metric(x,y*np.nan)

    def test_separate_reference_gates(self):
        real=[dict(candidate_vs_fp64={'allclose':True},candidate_vs_original={'allclose':True})]
        syn=[dict(kind='near_constant',candidate_vs_fp64={'allclose':True,'mae':1e-8},old_vs_fp64={'mae':.01},candidate_vs_original={'allclose':False}) for _ in range(3)]
        self.assertEqual(s.result_status(real,syn),'isolated_stability_passed_not_integrated')
        real[0]['candidate_vs_original']['allclose']=False
        self.assertEqual(s.result_status(real,syn),'isolated_stability_not_accepted')

    def test_no_promotion(self):
        self.assertFalse(s.RULES['accepted_for_release']);self.assertEqual(s.RULES['benchmark_calls'],0);self.assertTrue(s.RULES['preserve_old_cpu_gate'])


if __name__=='__main__':unittest.main()
