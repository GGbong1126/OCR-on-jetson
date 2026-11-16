from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import diagnose_det_boundary as m


class Tests(unittest.TestCase):
    def test_counts_do_not_relax_allclose(self):
        x=np.zeros((1,64,2,2),np.float32);y=x.copy();y[0,0,0,0]=.001
        r=m.error_metrics(x,y)
        self.assertEqual(r['outside_allclose_count'],1);self.assertFalse(r['allclose_diagnostic'])
        self.assertEqual(r['elements'],256);self.assertNotIn('values',r)

    def test_nonfinite_rejected(self):
        x=np.ones((1,64,2,2),np.float32);x[0,0,0,0]=np.inf
        with self.assertRaises(ValueError):m.error_metrics(x,x)

    def test_double_reference_cross_correlation_padding(self):
        x=np.ones((1,256,3,3),np.float32);w=np.zeros((64,256,9,9),np.float32)
        w[:,:,4,4]=1.;out=np.full((1,64,3,3),256.,np.float32)
        r=m.fp64_spots(x,w,out,out)
        self.assertEqual(r['sample_count'],34);self.assertEqual(r['cpu_max_abs_error'],0)
        self.assertEqual(r['gpu_max_abs_error'],0)


if __name__=='__main__':unittest.main()
