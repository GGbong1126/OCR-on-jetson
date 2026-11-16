import copy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import evaluate_det_hotspot as m


class Tests(unittest.TestCase):
    def test_numeric_diagnostic(self):
        a=np.array([0,.09,.3,1],np.float32).reshape(1,1,2,2);b=a.copy();b[0,0,0,1]=.11
        r=m.compare(a,b)
        self.assertEqual(r['mask_changed_at_point1'],1)
        self.assertFalse(r['allclose_diagnostic'])
        self.assertNotIn('passed',r)

    def test_bad_outputs(self):
        a=np.zeros((1,1,2,2),np.float32)
        for b in (a.reshape(1,4),np.full_like(a,np.nan),a.astype(np.float16)):
            with self.assertRaises(ValueError):m.compare(a,b)

    def test_inspector_exact_layer_and_formats(self):
        row=dict(Name='Conv.87',LayerType='CaskConvolution',ParameterType='Convolution',
                 Inputs=[{'Format/Datatype':'FP16'}],Outputs=[{'Format/Datatype':'FP32'}])
        self.assertEqual(m.inspect_layers({'Layers':[row]})['half_format_layer_count'],1)
        for bad in ('Conv.870','Conv.86'):
            v=dict(row,Name=bad)
            with self.assertRaises(ValueError):m.inspect_layers({'Layers':[v]})
        v=copy.deepcopy(row);v['Outputs'][0]['Format/Datatype']='FP16'
        with self.assertRaises(ValueError):m.inspect_layers({'Layers':[v]})
        with self.assertRaises(ValueError):m.inspect_layers({'Layers':['Conv.87']})

    def test_business_geometry_not_gate(self):
        r=dict(same_original_input=True,strict_equal=dict.fromkeys(('FullText','Text','MarkdownBodyLines','NER','raw_line_texts'),True),
               fields={'api_special_text_equal':dict.fromkeys(('title','category_no','page_no'),True)})
        r['strict_equal']['special']=False
        self.assertTrue(m.business_passed(r))
        r['fields']['api_special_text_equal']['page_no']=False
        self.assertFalse(m.business_passed(r))

    def test_protected_diagnostic_no_normalization(self):
        self.assertTrue(m.protected_equal('文字 12-3','另一文字12-3'))
        self.assertFalse(m.protected_equal('12-3','12/3'))
        self.assertFalse(m.protected_equal('12','13'))
        self.assertFalse(m.protected_equal('12','１２'))


if __name__=='__main__':unittest.main()
