import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import evaluate_enhance_fp32 as e


def fixture():
    return {'Layers':[dict(Name='conv',ParameterType='Convolution',LayerType='CaskConvolution',
                          Inputs=[{'Format/Datatype':'Row major linear FP32'}],
                          Outputs=[{'Format/Datatype':'Row major linear FP32'}])]}


class Tests(unittest.TestCase):
    def test_fp32_inspection(self):
        self.assertEqual(e.inspect(fixture())['low_precision_format_count'],0)
    def test_reject_half_int8_missing_formats(self):
        for value in ('Half','FP16','INT8',''):
            f=fixture(); f['Layers'][0]['Inputs'][0]['Format/Datatype']=value
            with self.assertRaises(ValueError): e.inspect(f)
    def test_reject_missing_or_low_precision_nonconv(self):
        with self.assertRaises(ValueError): e.inspect({'Layers':[]})
        f=fixture(); row=copy.deepcopy(f['Layers'][0]); row['ParameterType']='Reformat'
        row['Inputs'][0]['Format/Datatype']='Half'; f['Layers'].append(row)
        with self.assertRaises(ValueError): e.inspect(f)
    def test_description_excludes_arrays(self):
        row=dict(image_sha256='a',input_shape=[1,3,512,512],input_sha256='b',reference_sha256='c',x='private',reference='private')
        self.assertEqual(set(e.describe(row)),{'image_sha256','input_shape','input_sha256','reference_sha256'})


if __name__=='__main__': unittest.main()
