from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import ner_softmax_probe as p


class Tests(unittest.TestCase):
    def test_shape_and_dtype_guard(self):
        x=np.ones((1,12,2,2),np.float32);m=np.ones((1,2),np.float32)
        p.inputs(x,m)
        for xx,mm in ((x.astype(np.float16),m),(x,m*.5),(x,m[:,:1]),(x*np.nan,m),(x*1e21,m),(x[:,:,:,:1],m)):
            with self.assertRaises(ValueError):p.inputs(xx,mm)

    def test_all_masked_preserves_original_uniform(self):
        x=np.arange(12*127*127,dtype=np.float32).reshape(1,12,127,127);m=np.zeros((1,127),np.float32)
        self.assertTrue(np.isfinite(p.logits(x,m)).all())
        self.assertTrue(np.array_equal(p.oracle(x,m),np.full(x.shape,1/127)))

    def test_one_key_and_masked_zero(self):
        x=np.ones((1,12,2,2),np.float32);m=np.array([[0,1]],np.float32);y=p.oracle(x,m)
        self.assertTrue(np.all(y[...,0]==0));self.assertTrue(np.all(y[...,1]==1))

    def test_synthetic_coverage_reproducible_and_normalized(self):
        a=list(p.synthetic());self.assertEqual(len(a),32)
        for (l,k,x,m),(ll,kk,xx,mm) in zip(a,p.synthetic()):
            self.assertEqual((l,k),(ll,kk));self.assertTrue(np.array_equal(x,xx));self.assertTrue(np.array_equal(m,mm))
            y=p.oracle(x,m);self.assertTrue(np.isfinite(y).all());self.assertTrue(np.allclose(y.sum(axis=-1),1,rtol=0,atol=1e-12))
        self.assertEqual(sorted(set(l for l,k,x,m in a)),[2,127,128,512])

    def test_fp16_mask_constant_would_overflow(self):
        with np.errstate(over='ignore'):v=np.float16(-np.finfo(np.float32).max)
        self.assertFalse(np.isfinite(v));self.assertTrue(np.isfinite(np.float32(-np.finfo(np.float32).max)))

    def test_inspector_rejects_hidden_or_half(self):
        info={'Layers':[dict(Name='fp32_mask_softmax',LayerType='PluginV2',Inputs=[{'Format/Datatype':'Float'}]*2,Outputs=[{'Format/Datatype':'Float'}])]}
        self.assertTrue(p.inspect(info)['visible_plugin'])
        info['Layers'][0]['Outputs']=[{'Format/Datatype':'Half'}]
        with self.assertRaises(ValueError):p.inspect(info)
        info['Layers'][0]['LayerType']='Myelin'
        with self.assertRaises(ValueError):p.inspect(info)

    def test_fail_closed_and_no_release(self):
        r=dict(cpu={'allclose':True},fp64_softmax={'allclose':True},probability_contract_passed=True,repeat_exact=True)
        self.assertEqual(p.result_status([r]*72,[r]*32),'isolated_fp32_mask_softmax_passed_not_integrated')
        self.assertEqual(p.result_status([r]*71,[r]*32),'isolated_not_accepted')
        r['probability_contract_passed']=False
        self.assertEqual(p.result_status([r]*72,[r]*32),'isolated_not_accepted')
        self.assertFalse(p.RULES['accepted_for_release']);self.assertEqual(p.RULES['benchmark_calls'],0)

    def test_invalid_outputs_rejected(self):
        x=np.ones(2,np.float32)
        with self.assertRaises(ValueError):p.metrics(x,x*np.nan)
        with self.assertRaises(ValueError):p.metrics(x,x.astype(np.float64))


if __name__=='__main__':unittest.main()
