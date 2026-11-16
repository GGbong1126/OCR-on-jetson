import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import evaluate_det_cudnn as m


def fixtures():
    def conv(name,kind):
        return dict(Name=name,ParameterType='Convolution',LayerType='CudnnConvolution',TacticValue='0x5',
                    Inputs=[{'Format/Datatype':kind}],Outputs=[{'Format/Datatype':kind}])
    return dict(Layers=[conv('Conv.87','FP16'),conv('Conv.88 + Relu.1','FP32'),conv('Conv.91','FP32')])


class Tests(unittest.TestCase):
    def test_both_fp32_consumers(self):
        r=m.inspect(fixtures(),{'tactic':5})
        self.assertEqual(set(r['consumers']),{'Conv.88','Conv.91'})
    def test_missing_wrong_half_or_tactic_fail(self):
        for kind in ('missing','half','tactic','other_half'):
            data=fixtures()
            if kind=='missing':data['Layers'].pop()
            if kind=='half':data['Layers'][1]['Inputs'][0]['Format/Datatype']='FP16'
            if kind=='tactic':data['Layers'][0]['TacticValue']='0x6'
            if kind=='other_half':
                row=copy.deepcopy(data['Layers'][0]);row['Name']='Conv.1';data['Layers'].append(row)
            with self.assertRaises(ValueError):m.inspect(data,{'tactic':5})
    def test_names_not_prefix(self):
        self.assertFalse(m.named('Conv.870','Conv.87'));self.assertTrue(m.named('Conv.87 + X','Conv.87'))
    def test_adapter_restores_on_failure(self):
        before=(m.legacy.metadata,m.legacy.candidate)
        with self.assertRaises(RuntimeError):
            with m.adapter():
                self.assertIs(m.legacy.candidate,m.candidate);raise RuntimeError('test')
        self.assertEqual(before,(m.legacy.metadata,m.legacy.candidate))


if __name__=='__main__':unittest.main()
