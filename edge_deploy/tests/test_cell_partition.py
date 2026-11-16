from pathlib import Path
import sys
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import cell_partition as c


class Tests(unittest.TestCase):
    def test_initializer_output_preserved(self):
        nodes=[N(input=['image','weight'])]
        self.assertEqual(c.initializer_names(nodes,['fetch_name_0','fetch_name_1'],{'weight','fetch_name_1','unused'}),{'weight','fetch_name_1'})

    def test_partition_complete_and_cuts(self):
        nodes=[N(name='pre',input=['image'],output=[c.p.FEATURES[0]])]
        previous=c.p.FEATURES[0]
        for i,n in enumerate(c.p.BLOCK):
            value='v'+str(i);nodes.append(N(name=n,input=[previous,'weight'],output=[value]));previous=value
        nodes.append(N(name='head',input=[previous,c.p.FEATURES[0]],output=['out']))
        groups,cuts,outs=c.partition_indices(nodes,{'image','weight'})
        self.assertEqual([len(groups[k]) for k in ('prefix','block','suffix')],[1,7,1])
        self.assertEqual(cuts,[c.p.FEATURES[0]]);self.assertEqual(outs,[previous])
        with self.assertRaises(ValueError):c.partition_indices(nodes+nodes[:1],{'image','weight'})

    @staticmethod
    def data():
        return dict(fetch_name_0=np.array([[0,.9,10,10,100,100],[0,.1,20,20,30,30]],np.float32),
                    fetch_name_1=np.array([2],np.int32))

    def test_original_threshold_scaling_and_order(self):
        ref=self.data();out={k:v.copy() for k,v in ref.items()};feeds={'im_shape':np.array([[1280,640]],np.float32)}
        out['fetch_name_0'][0,3]+=.1
        r=c.compare(ref,out,feeds)
        self.assertEqual(r['reference_effective_count'],1);self.assertTrue(r['geometry_passed'])
        self.assertAlmostEqual(r['max_coordinate_delta_pixels'],.2,places=5)
        out['fetch_name_0'][0,3]+=1
        self.assertFalse(c.compare(ref,out,feeds)['geometry_passed'])

    def test_threshold_crossing_and_count_rejected(self):
        ref=self.data();out={k:v.copy() for k,v in ref.items()};feeds={'im_shape':np.array([[640,640]],np.float32)}
        out['fetch_name_0'][1,1]=.3
        self.assertFalse(c.compare(ref,out,feeds)['geometry_passed'])
        out=self.data();out['fetch_name_1'][0]=1
        self.assertFalse(c.compare(ref,out,feeds)['geometry_passed'])

    def test_invalid_and_empty(self):
        ref=self.data();out=self.data();feeds={'im_shape':np.array([[640,640]],np.float32)}
        out['fetch_name_0'][0,2]=np.nan
        with self.assertRaises(ValueError):c.compare(ref,out,feeds)
        ref['fetch_name_1'][0]=0
        r=c.compare(ref,ref,feeds)
        self.assertTrue(r['geometry_passed']);self.assertIsNone(r['min_ordered_iou'])

    def test_finite_gate(self):
        r=dict(count_output_exact=True,reference_effective_count=1,candidate_effective_count=1,ordered_classes_equal=True,min_ordered_iou=.999,max_coordinate_delta_pixels=.1)
        self.assertTrue(c.geometry_passed(r))
        for value in (float('nan'),float('inf'),-.1,1.):
            self.assertFalse(c.geometry_passed(dict(r,max_coordinate_delta_pixels=value)))

    def test_fp32_formats_and_sampling(self):
        t={'Format/Datatype':'Float'}
        info=dict(Layers=[dict(Name='GridSample0',Inputs=[t],Outputs=[t])])
        self.assertEqual(c.inspect_fp32(info,1)['fp32_grid_samples'],1)
        with self.assertRaises(ValueError):c.inspect_fp32(info,18)
        info['Layers'].append(dict(Name='hidden',Inputs=[{'Format/Datatype':'Half'}],Outputs=[]))
        with self.assertRaises(ValueError):c.inspect_fp32(info,1)

    def test_composition_routing_and_error_latch(self):
        x=c.Composed.__new__(c.Composed);x.errors=0;x.calls={'fp32':0,'mixed':0}
        x.parts={k:k for k in ('prefix','suffix','fp32','mixed')};seen=[]
        def run(role,values):
            seen.append((role,set(values)))
            if role=='prefix':return {k:1 for k in c.CUTS}
            if role=='mixed':return {k:2 for k in c.p.OUTPUTS}
            return {k:3 for k in c.FINAL}
        with patch.object(c.p,'inputs_meta'),patch.object(c.Composed,'named',side_effect=run):
            x.run({'image':0,'im_shape':0,'scale_factor':0},'mixed')
            self.assertEqual(seen[1],('mixed',{'Mul.21'}))
            self.assertEqual(seen[2][1],{'Concat.19','Mul.10','Mul.19','Add.47','im_shape','scale_factor'})
            self.assertEqual(x.calls['mixed'],1)
            with self.assertRaises(ValueError):x.run({},'unknown')
            with self.assertRaises(ValueError):x.run({},'mixed')
            self.assertEqual(x.errors,2)

    def test_no_automatic_promotion(self):
        self.assertFalse(c.RULES['benchmark_authorized'])
        self.assertFalse(c.RULES['accepted_for_release'])
        self.assertTrue(c.RULES['strict_business'])


if __name__=='__main__':unittest.main()
