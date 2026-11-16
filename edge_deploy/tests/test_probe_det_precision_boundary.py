from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import probe_det_precision_boundary as m


class Tests(unittest.TestCase):
    def test_explicit_empty_bias_and_failed_layer(self):
        trt=SimpleNamespace(float32='float',Weights=Mock(return_value='empty_weights'))
        network=Mock();conv=network.add_convolution_nd.return_value
        self.assertIs(m.add_hotspot(network,'input','weights',trt),conv)
        network.add_convolution_nd.assert_called_once_with('input',64,(9,9),'weights','empty_weights')
        trt.Weights.assert_called_once_with('float')
        network.add_convolution_nd.return_value=None
        with self.assertRaises(RuntimeError):m.add_hotspot(network,'input','weights',trt)

    def test_hash_contract(self):
        self.assertRegex(m.RANGES_SHA,r'^[0-9a-f]{64}$')

    def test_precision_boundary(self):
        self.assertEqual(m.policy('fp16_via_cast'),dict(conv_compute='half',conv_output='half',identity_output='float',external_io='float'))
        self.assertEqual(m.policy('fp16_direct_fp32')['conv_output'],'float')
        with self.assertRaises(ValueError):m.policy('int8')

    def node(self):
        return SimpleNamespace(name='Conv.87',op_type='Conv',input=['p2o.pd_op.add.4.0','weight'],output=['p2o.pd_op.conv2d.60.0'])

    def test_source_contract(self):
        w=np.zeros((64,256,9,9),np.float32);a=dict(kernel_shape=[9,9],strides=[1,1],pads=[4,4,4,4])
        m.node_contract(self.node(),w,a)
        a['strides']=[2,2]
        with self.assertRaises(ValueError):m.node_contract(self.node(),w,a)

    def inspector(self,kind='Half'):
        return {'Layers':[{'Name':'Conv.87','LayerType':'CaskConvolution','ParameterType':'Convolution',
            'Inputs':[{'Name':'x','Format/Datatype':kind}], 'Outputs':[{'Name':'conv87','Format/Datatype':kind}]},
            {'Name':'restore','Inputs':[{'Name':'conv87','Format/Datatype':kind}],
             'Outputs':[{'Name':'y','Format/Datatype':'Float'}]}]}

    def test_actual_formats(self):
        self.assertEqual(m.inspect(self.inspector(),'fp16_via_cast')['layer_count'],2)
        m.inspect(self.inspector('Float'),'fp32')
        with self.assertRaises(ValueError):m.inspect(self.inspector('Float'),'fp16_via_cast')
        with self.assertRaises(ValueError):m.inspect({'Layers':['Conv.87']},'fp16_via_cast')

    def test_missing_restoration_or_wrong_layer(self):
        r=self.inspector();r['Layers'][1]['Outputs'][0]['Format/Datatype']='Half'
        with self.assertRaises(ValueError):m.inspect(r,'fp16_via_cast')
        r=self.inspector();r['Layers'][0]['Name']='Conv.870'
        with self.assertRaises(ValueError):m.inspect(r,'fp16_via_cast')

    def test_numeric_is_diagnostic_no_values(self):
        x=np.ones((1,64,2,2),np.float32);r=m.compare(x,x+.01)
        self.assertFalse(r['allclose_diagnostic']);self.assertNotIn('values',r)
        with self.assertRaises(ValueError):m.compare(x,x.astype(np.float16))
        x[0,0,0,0]=np.inf
        with self.assertRaises(ValueError):m.compare(x,x)


if __name__=='__main__':unittest.main()
