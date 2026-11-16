from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import decompose_det_error as m


class Tests(unittest.TestCase):
    def test_half_roundtrip_no_clipping(self):
        x=np.array([.1,-.1,65504],np.float32);out=m.round_half(x)
        self.assertEqual(out.dtype,np.float32);self.assertFalse(np.array_equal(x,out))
        self.assertTrue(np.array_equal(out,m.round_half(out)))
        for bad in (np.array([65505],np.float32),np.array([np.nan],np.float32),np.zeros(0,np.float32),np.ones(2,np.float16)):
            with self.assertRaises(ValueError):m.round_half(bad)

    def test_signed_closure_not_percentages(self):
        y=np.array([1.,2.],np.float32);b=y+.5;rounded=b-.25;g=rounded-1.
        r=m.decompose(y,b,rounded,g)
        self.assertEqual(r['signed_closure_max_abs'],0)
        self.assertEqual(r['implementation_residual']['mae'],1)
        self.assertEqual(r['total_gpu_vs_cpu']['mae'],.75)
        self.assertNotEqual(sum(r[k]['mae'] for k in ('operand_rounding','output_rounding','implementation_residual')),r['total_gpu_vs_cpu']['mae'])

    def test_stats_no_values_and_invalid(self):
        self.assertEqual(m.stats([-3,4])['mae'],3.5)
        self.assertNotIn('values',m.stats([1]))
        for bad in ([],[np.inf]):
            with self.assertRaises(ValueError):m.stats(bad)

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):m.decompose(np.ones(2),np.ones(3),np.ones(2),np.ones(2))


if __name__=='__main__':unittest.main()
