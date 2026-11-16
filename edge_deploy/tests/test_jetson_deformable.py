"""CPU-only graph/metric regression; real CUDA evidence is recorded separately."""
import sys
from pathlib import Path
import unittest
import tempfile
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import onnxruntime as ort

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
from lower_deformable import lower_norm, lower_mod, plugin_graph, compact_uniform_constants, remove_empty_concat
from compare_deformable import compare_boxes, image_pixels
from PIL import Image


def run(m,feeds):
    o=ort.SessionOptions();o.intra_op_num_threads=2
    return ort.InferenceSession(m.SerializeToString(),o,providers=['CPUExecutionProvider']).run(None,feeds)


class DeformableTests(unittest.TestCase):
    def test_business_pixels_match_jpeg_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source.png';target=Path(d)/'resized.jpg'
            im=Image.fromarray(np.random.default_rng(1).integers(0,256,(1600,1000,3),dtype=np.uint8))
            im.save(source)
            im.thumbnail((1500,1500),Image.Resampling.LANCZOS)
            im.save(target,format='JPEG',quality=90)
            with Image.open(target) as x:ref=np.asarray(x.convert('RGB'))
            np.testing.assert_array_equal(image_pixels(source,'business'),ref)

    def test_norm_equivalence(self):
        ins=[h.make_tensor_value_info('x',T.FLOAT,[1,7,13])]
        out=[h.make_tensor_value_info('y',T.FLOAT,[1,7,13])]
        rng=np.random.default_rng(6)
        init=[nh.from_array(rng.normal(size=13).astype(np.float32),k) for k in ('scale','bias')]
        node=h.make_node('LayerNormalization',['x','scale','bias'],['y'],axis=2,epsilon=1e-5,name='ln')
        m=h.make_model(h.make_graph([node],'test',ins,out,init),opset_imports=[h.make_opsetid('',17)],ir_version=8)
        x=rng.normal(size=(1,7,13)).astype(np.float32)
        ref=run(m,{'x':x})[0];self.assertEqual(lower_norm(m),['ln'])
        np.testing.assert_allclose(ref,run(m,{'x':x})[0],rtol=1e-5,atol=2e-6)

    def test_remainder_of_topk_indices(self):
        ins=[h.make_tensor_value_info('x',T.FLOAT,[1,103])]
        out=[h.make_tensor_value_info('y',T.INT64,[1,40])]
        init=[nh.from_array(np.array([40],np.int64),'k'),nh.from_array(np.array(25,np.int64),'d')]
        nodes=[h.make_node('TopK',['x','k'],['v','indices']),h.make_node('Mod',['indices','d'],['y'],name='mod')]
        m=h.make_model(h.make_graph(nodes,'test',ins,out,init),opset_imports=[h.make_opsetid('',16)],ir_version=8)
        feeds={'x':np.random.default_rng(7).normal(size=(1,103)).astype(np.float32)}
        ref=run(m,feeds)[0];lower_mod(m);np.testing.assert_array_equal(ref,run(m,feeds)[0])

    def test_reject_arbitrary_mod(self):
        m=h.make_model(h.make_graph([h.make_node('Mod',['x','d'],['y'])],'bad',[],[],
                      [nh.from_array(np.array(25,np.int64),'d')]))
        with self.assertRaises(ValueError):lower_mod(m)

    def test_reject_grid_attributes(self):
        n=h.make_node('GridSample',['x','g'],['y'],align_corners=1,mode='bilinear',padding_mode='zeros')
        m=h.make_model(h.make_graph([n],'bad',[],[]))
        with self.assertRaises(ValueError):plugin_graph(m)

    def test_compressed_constant_and_empty_concat(self):
        a=np.full((1,300,200,200),1e8,np.float32)
        m=h.make_model(h.make_graph([], 'test',[],[h.make_tensor_value_info('a',T.FLOAT,list(a.shape))],
                                   [nh.from_array(a,'a')]),opset_imports=[h.make_opsetid('',16)],ir_version=8)
        self.assertEqual(len(compact_uniform_constants(m)),1)
        self.assertLess(len(m.SerializeToString()),1000)
        np.testing.assert_array_equal(run(m,{})[0],a)

    def test_task_matching_and_failures(self):
        r=[{'class':21,'score':.9,'xyxy':np.array([0,0,100,100])}]
        c=[{'class':21,'score':.9001,'xyxy':np.array([.1,0,100,100])}]
        self.assertTrue(compare_boxes(r,c)['passed'])
        c[0]['class']=22;self.assertFalse(compare_boxes(r,c)['passed'])
        self.assertFalse(compare_boxes(r,[])['passed'])
        self.assertFalse(compare_boxes(r,r+r)['passed'])
        self.assertTrue(compare_boxes([],[])['passed'])


if __name__=='__main__':unittest.main()
