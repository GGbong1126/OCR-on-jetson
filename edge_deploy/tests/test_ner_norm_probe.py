import copy
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import ner_norm_probe as p


class Tests(unittest.TestCase):
    @staticmethod
    def spec():
        ins=[['x'],['x','o0'],[],['o1','o2'],['o3'],[],['o4','o5'],['o6'],['o1','o7'],['o8','g'],['o9','b']]
        nodes=[dict(op=op,inputs=names,outputs=['o'+str(i)]) for i,(op,names) in enumerate(zip(p.OPS,ins))]
        for i in (0,4):nodes[i].update(axes=[-1],keepdims=1)
        return dict(nodes=nodes,gamma_name='g',beta_name='b',gamma_shape=[768],beta_shape=[768],power=2.,epsilon=1e-12)

    def test_exact_graph_pattern(self):self.assertEqual(p.validate_spec(self.spec()),('x','o10'))

    def test_reject_axis_power_epsilon_shape(self):
        for field,value in (('power',1.),('epsilon',0.),('epsilon',float('nan')),('gamma_shape',[1,768])):
            s=self.spec();s[field]=value
            with self.assertRaises(ValueError):p.validate_spec(s)
        s=self.spec();s['nodes'][0]['axes']=[1]
        with self.assertRaises(ValueError):p.validate_spec(s)

    def test_reject_wiring(self):
        s=self.spec();s['nodes'][8]['inputs']=['x','o7']
        with self.assertRaises(ValueError):p.validate_spec(s)

    @staticmethod
    def layers():
        t={'Format/Datatype':'Row major linear FP32'}
        return {'Layers':[dict(Name='isolated_norm_fp32',LayerType='PluginV2',Inputs=[t.copy() for _ in range(3)],Outputs=[t.copy()])]}

    def test_visible_float_plugin(self):self.assertTrue(p.inspect(self.layers())['fp32_plugin_boundary'])

    def test_reject_half_or_opaque(self):
        info=self.layers();info['Layers'][0]['Inputs'][0]['Format/Datatype']='Half'
        with self.assertRaises(ValueError):p.inspect(info)
        info=self.layers();info['Layers'].append(dict(Name='ForeignNode',LayerType='Myelin'))
        with self.assertRaises(ValueError):p.inspect(info)

    def test_metrics_and_nonfinite(self):
        x=np.ones((1,2,768),np.float32)
        self.assertTrue(p.metrics(x,x)['allclose'])
        self.assertFalse(p.metrics(x,x*2)['allclose'])
        y=x.copy();y[0,0,0]=np.nan
        with self.assertRaises(ValueError):p.metrics(x,y)

    def test_synthetic_reproducible_coverage(self):
        first=list(p.synthetic());second=list(p.synthetic());self.assertEqual(len(first),12)
        for (length,kind,x),(ll,kk,y) in zip(first,second):
            self.assertEqual((length,kind),(ll,kk));self.assertTrue(np.array_equal(x,y));self.assertEqual(x.shape,(1,length,768))
            self.assertTrue(np.isfinite(x.astype(np.float16)).all())

    def test_no_whole_model_or_promotion(self):
        self.assertEqual(p.RULES['benchmark_calls'],0);self.assertFalse(p.RULES['accepted_for_release'])


if __name__=='__main__':unittest.main()
