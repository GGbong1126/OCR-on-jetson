import copy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import det_hotspot_mixed as m


class Tests(unittest.TestCase):
    def test_only_exact_hotspot(self):
        self.assertEqual(m.layer_policy('Conv.87','CONVOLUTION',[0]),'fp16_compute_fp32_output')
        for name in ('Conv.86','Conv.870','Conv.139','Conv.140','Sigmoid.0'):
            self.assertEqual(m.layer_policy(name,'CONVOLUTION',[0]),'fp32')

    def test_shape_and_reorder(self):
        self.assertEqual(m.layer_policy('shape','SHAPE',[]),'unchanged_nonfloat_or_shape')
        self.assertEqual(m.layer_policy('identity','IDENTITY',[0]),'fp32')
        with self.assertRaises(ValueError):m.layer_policy('x','UNKNOWN',[0])
        with self.assertRaises(ValueError):m.layer_policy('Conv.87','CONVOLUTION',[])

    def test_finite_range_and_no_tensor_values(self):
        r=m.tensor_range(np.array([-70000,2],np.float32))
        self.assertEqual(r['abs_gt_fp16_max'],1)
        self.assertNotIn('values',r)
        for x in (np.array([np.inf],np.float32),np.array([],np.float32),np.ones(2,np.float16)):
            with self.assertRaises(ValueError):m.tensor_range(x)

    def test_range_gate(self):
        v=m.tensor_range(np.array([4],np.float32))
        r=dict(status='fp32_ranges_measured_not_fp16_validated',source_sha256=m.SOURCE_SHA,
               nodes=[dict(name='Conv.87',op='Conv',input='in',output='out',parameter_ranges={'w':v})],
               rows=[{'ranges':{'in':copy.deepcopy(v),'out':copy.deepcopy(v)}} for _ in range(6)])
        m.require_ranges(r)
        r['rows'][0]['ranges']['out']['abs_max']=40000
        with self.assertRaises(ValueError):m.require_ranges(r)
        r['rows'].pop()
        with self.assertRaises(ValueError):m.require_ranges(r)

    def test_capture_metadata_only(self):
        x=np.zeros((1,3,32,32),np.float32); y=np.zeros((1,1,32,32),np.float32)
        r=m.describe(dict(x=x,reference=y,image_sha256='image',image_type=3))
        self.assertEqual(set(r),{'image_sha256','image_type','input_shape','input_sha256','reference_sha256'})


if __name__=='__main__':unittest.main()
