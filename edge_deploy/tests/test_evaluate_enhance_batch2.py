from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import evaluate_enhance_batch2 as e


class Fake:
    def __init__(self, n): self.n=n; self.calls=0; self.errors=0
    def run(self, names, feeds):
        x=feeds['images']; assert len(x)==self.n; self.calls+=1
        return [x[:,:1].copy()]


class Tests(unittest.TestCase):
    def test_groups_preserve_order_no_padding(self):
        for n in (1,2,3,6,9,14):
            groups=e.batches(n)
            self.assertEqual(sum(groups,[]),list(range(n)))
            self.assertEqual(sum(len(g)==1 for g in groups),n%2)
        for n in (0,-1,1.2,True):
            with self.assertRaises(ValueError): e.batches(n)
    def test_route(self):
        a,b=Fake(1),Fake(2)
        x=np.stack([np.zeros((3,512,512),np.float32),np.ones((3,512,512),np.float32)])
        np.testing.assert_array_equal(e.dispatch(a,b,x),x[:,:1])
        np.testing.assert_array_equal(e.dispatch(a,b,x[:1]),x[:1,:1])
        self.assertEqual((a.calls,b.calls),(1,1))
    def test_no_error_fallback(self):
        a,b=Fake(1),Fake(2); b.errors=1
        with self.assertRaises(RuntimeError): e.dispatch(a,b,np.zeros((1,3,512,512),np.float32))
        self.assertEqual(a.calls,0)
    def test_invalid_inputs(self):
        for x in (np.zeros((3,3,512,512),np.float32),np.zeros((2,3,512,511),np.float32),
                  np.zeros((2,3,512,512),np.float16),np.full((1,3,512,512),np.nan,np.float32)):
            with self.assertRaises(ValueError): e.validate_input(x)
    def test_page_and_negative_control(self):
        e.verify_page(1,['a','b','c'],['a','b','c'],[2,1]); e.verify_page(3,[],[],[])
        for hashes,sizes in ((['b','a','c'],[2,1]),(['a','b','c'],[1,2]),(['a','b','c'],[2,2])):
            with self.assertRaises(ValueError): e.verify_page(1,['a','b','c'],hashes,sizes)
        with self.assertRaises(ValueError): e.verify_page(3,[],[],[1])


if __name__=='__main__': unittest.main()
