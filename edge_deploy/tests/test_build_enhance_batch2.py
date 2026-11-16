from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'jetson'))
import build_enhance_batch2 as b


class Tests(unittest.TestCase):
    def test_contract(self):
        b.contract([('images',[2,3,512,512],'DataType.FLOAT')],[('output',[2,1,512,512],'DataType.FLOAT')])
        for n in (1, 3, -1):
            with self.assertRaises(ValueError):
                b.contract([('images',[n,3,512,512],'DataType.FLOAT')],[('output',[2,1,512,512],'DataType.FLOAT')])
    def test_compare(self):
        x=np.zeros((2,1,4,4),np.float32)
        self.assertTrue(b.compare(x,x)['allclose'])
        self.assertFalse(b.compare(x,x+1)['allclose'])
    def test_invalid(self):
        x=np.zeros((2,1,4,4),np.float32)
        for y in (x[:1],x.astype(np.float16),x+np.nan):
            with self.assertRaises(ValueError): b.compare(x,y)


if __name__=='__main__': unittest.main()
