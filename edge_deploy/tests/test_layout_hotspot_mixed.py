from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'jetson'))
import layout_hotspot_mixed as m


class Layer:
    def __init__(self, name, inputs, out):
        self.name=name; self.inputs=inputs; self.out=out
        self.num_inputs=len(inputs); self.type='CONVOLUTION'
    def get_input(self, i): return self.inputs[i]
    def get_output(self, i): return self.out
    def set_input(self, i, value): self.inputs[i]=value
    def set_output_type(self, i, value): self.dtype=value


class Net:
    def __init__(self, extra=False):
        self.layers=[]
        for name in m.TARGETS:
            out=NS(name=m.OUTPUTS[name]); self.layers.append(Layer(name,[NS(name='x')],out))
            for consumer, index in m.CONSUMERS[name]:
                self.layers.append(Layer(consumer,[out],NS(name=consumer)))
        if extra: self.layers.append(Layer('unexpected',[self.layers[6].out],NS(name='other')))
    @property
    def num_layers(self): return len(self.layers)
    def get_layer(self, i): return self.layers[i]
    def add_identity(self, value):
        layer=Layer('identity',[value],NS(name='new')); self.layers.append(layer); return layer


class Tests(unittest.TestCase):
    def test_callback_error_latches_outside_business_catch(self):
        def fail(): raise ValueError('do not log private error content')
        guard=m.GuardedCall(fail)
        with self.assertRaises(ValueError): guard()
        self.assertEqual(guard.errors,['ValueError'])
        with self.assertRaises(RuntimeError): guard.require_healthy()
        with self.assertRaises(RuntimeError): guard()
        self.assertEqual(m.GuardedCall(lambda: 3)(),3)
    def test_ancestor_closed_minimal_order(self):
        nodes=[NS(input=['x','w'],output=['a']), NS(input=['a'],output=['b']), NS(input=['other'],output=['unused'])]
        self.assertEqual(m.ancestor_indices(nodes,['b'],{'x','w'}),[0,1])
    def test_missing_cycle_duplicate_rejected(self):
        for nodes in ([NS(input=['missing'],output=['a'])],
                      [NS(input=['b'],output=['a']),NS(input=['a'],output=['b'])],
                      [NS(input=['x'],output=['a']),NS(input=['x'],output=['a'])]):
            with self.assertRaises(ValueError): m.ancestor_indices(nodes,['a'],{'x'})
    def test_rewire_all_six_consumers(self):
        net=Net(); rows=m.rewire(net,NS(float32='float'))
        self.assertEqual(len(rows),3); self.assertEqual(net.num_layers,12)
        for i in range(3):
            self.assertIs(net.layers[i*3+1].get_input(0),net.layers[9+i].out)
            self.assertIs(net.layers[i*3+2].get_input(0),net.layers[9+i].out)
            self.assertEqual(net.layers[9+i].dtype,'float')
    def test_reject_before_any_mutation(self):
        net=Net(True)
        with self.assertRaises(ValueError): m.rewire(net,NS(float32='float'))
        self.assertEqual(net.num_layers,10)
        self.assertEqual(net.layers[0].out.name,'Add.35')
    def test_precision_policy(self):
        self.assertEqual(m.layer_policy('Conv.91','CONVOLUTION',[0]),'half_compute_half_output')
        self.assertEqual(m.layer_policy('other','SOFTMAX',[0]),'fp32')
        self.assertEqual(m.layer_policy('grid','PLUGIN_V2',[0]),'fp32')
        self.assertEqual(m.layer_policy('shape','SHAPE',[]),'unchanged_integer_or_shape')
        for name,kind,outputs in [('Conv.91','ACTIVATION',[0]),('x','UNKNOWN',[0]),('Conv.93','CONVOLUTION',[])]:
            with self.assertRaises(ValueError): m.layer_policy(name,kind,outputs)
    def test_range_finite(self):
        self.assertEqual(m.tensor_range(np.array([1,-70000],np.float32))['abs_gt_fp16_max'],1)
        for x in (np.array([np.nan],np.float32),np.array([],np.float32),np.array([1],np.float64)):
            with self.assertRaises(ValueError): m.tensor_range(x)
    def test_input_contract(self):
        x={k:np.zeros(shape,np.float32) for k,shape in m.SHAPES.items()}
        self.assertEqual(set(m.input_metadata(x)),set(m.SHAPES))
        x['im_shape']=x['im_shape'].astype(np.float16)
        with self.assertRaises(ValueError): m.input_metadata(x)
    def test_range_gate(self):
        value=dict(abs_gt_fp16_max=0,abs_max=10)
        report=dict(status='fp32_prefix_ranges_measured_not_fp16_validated',source_sha256=m.SOURCE_SHA,
                    nodes=[dict(name=n,input=n+'i',output=n+'o',parameter_ranges={'w':value,'b':value}) for n in m.TARGETS],
                    rows=[dict(ranges={n+s:dict(value) for n in m.TARGETS for s in ('i','o')}) for _ in range(6)])
        m.require_ranges(report)
        report['rows'][0]['ranges']['Conv.91o']['abs_max']=32753
        with self.assertRaises(ValueError): m.require_ranges(report)
        report['rows']=[]
        with self.assertRaises(ValueError): m.require_ranges(report)


if __name__=='__main__': unittest.main()
