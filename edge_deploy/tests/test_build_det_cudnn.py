from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import build_det_cudnn as m


def algo(i=0,impl=1,tactic=5,dtype='HALF'):
    return dict(index=i,implementation=impl,tactic=tactic,io=[dict(dtype=dtype,format='LINEAR',strides=[1])])


class Layer:
    def __init__(self,name,inputs,out):self.name=name;self.inputs=inputs;self.out=out;self.num_inputs=len(inputs);self.type='CONVOLUTION'
    def get_input(self,i):return self.inputs[i]
    def get_output(self,i):return self.out
    def set_input(self,i,x):self.inputs[i]=x
    def set_output_type(self,i,dtype):self.output_dtype=dtype


class Net:
    def __init__(self,extra=False):
        x=NS(name='x');y=NS(name='p2o.pd_op.conv2d.60.0')
        self.layers=[Layer('Conv.87',[x],y),Layer('Conv.88',[y],NS(name='a')),Layer('Conv.91',[y],NS(name='b'))]
        if extra:self.layers.append(Layer('Unexpected',[y],NS(name='c')))
    @property
    def num_layers(self):return len(self.layers)
    def get_layer(self,i):return self.layers[i]
    def add_identity(self,x):
        layer=Layer('new',[x],NS(name='new'));self.layers.append(layer);return layer


class Tests(unittest.TestCase):
    def test_exact_selection(self):
        self.assertEqual(m.choose([algo(0,2),algo(1)],algo()),[1])
        r=algo();r['io'][0]['strides']=[4]
        self.assertEqual(m.choose([r],algo()),[0])
    def test_absent_ambiguous_wrong_precision_fail(self):
        for rows in ([],[algo(0,2)],[algo(0,tactic=6)],[algo(0,dtype='FLOAT')],[algo(0),algo(1)]):
            with self.assertRaises(ValueError):m.choose(rows,algo())
    def test_both_consumers_rewired(self):
        n=Net();r=m.rewire(n,NS(float32='float'))
        self.assertEqual(sorted(v['name'] for v in r),['Conv.88','Conv.91'])
        self.assertIs(n.layers[1].get_input(0),n.layers[3].out)
        self.assertIs(n.layers[2].get_input(0),n.layers[3].out)
        self.assertEqual(n.layers[3].output_dtype,'float')
        self.assertTrue(n.layers[0].out.name.endswith('__fp16'))
    def test_unexpected_consumer_fail_before_mutation(self):
        n=Net(True)
        with self.assertRaises(ValueError):m.rewire(n,NS(float32='float'))
        self.assertEqual(n.num_layers,4)


if __name__=='__main__':unittest.main()
