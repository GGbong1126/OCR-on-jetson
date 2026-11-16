from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import diagnose_ner_norm as d


class Tests(unittest.TestCase):
    def test_constant_returns_beta(self):
        x=np.full((1,2,768),.125,np.float32);g=np.ones(768,np.float32);b=np.arange(768,dtype=np.float32)/768
        self.assertTrue(np.array_equal(d.emulate(x,g,b,1e-12),np.broadcast_to(b,x.shape)))
        self.assertTrue(np.array_equal(d.precise(x,g,b,1e-12),np.broadcast_to(b,x.shape)))

    def test_tree_sum_ones(self):self.assertTrue(np.all(d.tree_sum(np.ones((1,2,768),np.float32))==768))

    def test_half_rounding_erases_small_variance(self):
        x=next(x for length,kind,x in d.p.synthetic() if length==2 and kind=='near_constant')
        self.assertGreater(float(x.astype(np.float64).var()),0)
        self.assertTrue(np.all(x.astype(np.float16)==np.float16(.125)))

    def test_finite_metric(self):
        x=np.ones((1,2),np.float64);self.assertEqual(d.errors(x,x.astype(np.float32))['max_abs_error'],0)
        with self.assertRaises(ValueError):d.errors(x,x*np.nan)


if __name__=='__main__':unittest.main()
