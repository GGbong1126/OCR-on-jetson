from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import ner_mixed_trial as n


class Tests(unittest.TestCase):
    def test_only_encoder_linear_targets(self):
        self.assertEqual(len(n.TARGETS),72);self.assertEqual(len(set(n.TARGETS)),72)
        self.assertFalse(any('classifier' in s or 'LayerNorm' in s or 'Softmax' in s for s in n.TARGETS))
        self.assertNotIn('/inner/bert/encoder/layer.0/attention/self/MatMul',n.TARGETS)

    def test_policy_and_no_promotion(self):
        self.assertEqual(n.policy(n.TARGETS[0],'MATRIX_MULTIPLY',[0]),'half_projection')
        self.assertEqual(n.policy('Softmax','SOFTMAX',[0]),'fp32')
        self.assertEqual(n.policy('integer','SHAPE',[]),'integer_shape_unchanged')
        with self.assertRaises(ValueError):n.policy(n.TARGETS[0],'CONVOLUTION',[0])
        with self.assertRaises(ValueError):n.policy('foreign','UNKNOWN',[0])
        self.assertFalse(n.RULES['benchmark_authorized']);self.assertFalse(n.RULES['accepted_for_release'])

    @staticmethod
    def feeds(length=10):return dict(input_ids=np.full((1,length),5,np.int64),attention_mask=np.ones((1,length),np.int64),token_type_ids=np.zeros((1,length),np.int64))

    def test_valid_shape_boundaries(self):
        for length in (2,3,128,512):self.assertEqual(n.input_meta(self.feeds(length))['input_ids']['shape'],[1,length])
        for length in (1,513):
            with self.assertRaises(ValueError):n.input_meta(self.feeds(length))

    def test_bad_ids_dtype_masks(self):
        for k,value in (('input_ids',21128),('input_ids',-1),('attention_mask',2),('token_type_ids',2)):
            f=self.feeds();f[k][0,0]=value
            with self.assertRaises(ValueError):n.input_meta(f)
        f=self.feeds();f['input_ids']=f['input_ids'].astype(np.int32)
        with self.assertRaises(ValueError):n.input_meta(f)

    def test_description_excludes_private_data(self):
        row=dict(image_sha256='image',image_type=3,inputs=n.input_meta(self.feeds()),output_sha256='output',feeds=self.feeds(),reference=np.ones(10))
        self.assertEqual(set(n.describe(row)),{'image_sha256','image_type','inputs','output_sha256'})


if __name__=='__main__':unittest.main()
