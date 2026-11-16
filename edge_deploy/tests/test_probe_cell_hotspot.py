import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import probe_cell_hotspot as p


class Tests(unittest.TestCase):
    def test_consumer_both_outputs(self):
        value=p.consumer_audit('def _detect_cells_onnx():\n    return outputs[0], outputs[1]\n')
        self.assertTrue(value['both_model_outputs_consumed'])
        for expr in ('outputs[0], outputs[2]','outputs[i]','outputs[0]'):
            with self.assertRaises(ValueError):p.consumer_audit('def _detect_cells_onnx():\n    return '+expr+'\n')

    def test_capture_shape_and_finite(self):
        data={k:np.zeros(v,dtype=np.float32) for k,v in p.SHAPES.items()}
        self.assertEqual(set(p.inputs_meta(data)),set(p.SHAPES))
        data['image'][0,0,0,0]=np.nan
        with self.assertRaises(ValueError):p.inputs_meta(data)
        data['image']=np.zeros((1,3,32,32),np.float32)
        with self.assertRaises(ValueError):p.inputs_meta(data)

    def test_policy_does_not_promote_or_benchmark(self):
        self.assertFalse(p.RULES['accepted_for_release'])
        self.assertFalse(p.RULES['benchmark_authorized'])
        self.assertTrue(p.RULES['fp32_silu_and_external_io'])

    def test_ranges(self):
        v=dict(abs_max=2.,abs_gt_fp16_max=0)
        r=dict(status='original_cell_ranges_measured',rules=copy.deepcopy(p.RULES),
               graph=dict(parameters=[dict(parameters={'w':v})]),rows=[dict(ranges={'x':v})])
        p.range_gate(r)
        for limit in (32753.,float('inf'),float('nan')):
            bad=copy.deepcopy(r);bad['rows'][0]['ranges']['x']['abs_max']=limit
            with self.assertRaises(ValueError):p.range_gate(bad)
        r['rows']=[]
        with self.assertRaises(ValueError):p.range_gate(r)

    def test_compare(self):
        a=np.zeros((1,256,2,2),np.float32)
        self.assertTrue(p.compare(a,a.copy())['allclose_diagnostic'])
        self.assertFalse(p.compare(a,a+1)['allclose_diagnostic'])
        for b in (np.full_like(a,np.inf),np.ones((1,2),np.float32),a.astype(np.float16)):
            with self.assertRaises(ValueError):p.compare(a,b)

    @staticmethod
    def info(variant):
        dtype='Half' if variant=='mixed' else 'Float'
        def tensor(name,kind):return dict(Name=name,**{'Format/Datatype':kind})
        rows=[]
        for n,o in zip(p.TARGETS,p.OUTPUTS):
            rows += [dict(Name=n,ParameterType='Convolution',LayerType='CaskConvolution',Inputs=[tensor('x',dtype)],Outputs=[tensor(o+'__compute',dtype)]),
                     dict(Name=n+'_restore',LayerType='Reformat',Inputs=[tensor(o+'__compute',dtype)],Outputs=[tensor(o,'Float')])]
        return dict(Layers=rows)

    def test_inspector_success(self):
        for v in ('fp32','mixed'):self.assertTrue(p.inspect(self.info(v),v)['external_fp32'])

    def test_inspector_no_hidden_half(self):
        info=self.info('mixed')
        info['Layers'].append(dict(Name='ForeignNode',LayerType='Myelin',Inputs=[{'Format/Datatype':'Half'}],Outputs=[{'Format/Datatype':'Half'}]))
        with self.assertRaises(ValueError):p.inspect(info,'mixed')
        with self.assertRaises(ValueError):p.inspect(self.info('mixed'),'fp32')

    def test_inspector_requires_boundary(self):
        info=self.info('mixed');info['Layers'][1]['Outputs'][0]['Format/Datatype']='Half'
        with self.assertRaises(ValueError):p.inspect(info,'mixed')
        info=self.info('mixed');info['Layers'].pop(0)
        with self.assertRaises(ValueError):p.inspect(info,'mixed')


if __name__=='__main__':unittest.main()
