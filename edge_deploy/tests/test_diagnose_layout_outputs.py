import copy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import diagnose_layout_outputs as d
import summarize_layout_outputs as s
from test_layout_outputs import fixture


class Tests(unittest.TestCase):
    def test_order_is_diagnostic(self):
        a=[dict(label=1,score=.9,coordinate=[0,0,10,10]),dict(label=1,score=.8,coordinate=[20,20,30,30])]
        r=d.match_boxes(a,a[::-1]);self.assertEqual(r['unique_matched_count'],2)
        self.assertFalse(r['order_preserved']);self.assertEqual(r['matched_max_coordinate_delta'],0)
    def test_class_and_ambiguous_matches_rejected(self):
        a=[dict(label=1,score=.9,coordinate=[0,0,10,10])];b=copy.deepcopy(a);b[0]['label']=2
        self.assertEqual(d.match_boxes(a,b)['unique_matched_count'],0)
        self.assertEqual(d.match_boxes(a,a+a)['unique_matched_count'],0)
    def test_original_threshold_scaling_and_nms_call(self):
        values=dict(fetch_name_0=np.array([[2,.5,1,2,3,4,0],[2,.29,2,3,4,5,0]],np.float32),fetch_name_1=np.array([2],np.int32))
        def nms(boxes,iou_threshold):self.assertEqual(iou_threshold,.5);return boxes
        before,after=d.business_boxes(values,{'im_shape':np.array([[1600,800]])},nms)
        self.assertEqual(len(before),1);self.assertEqual(after[0]['coordinate'],[1,4,3,8])
    def test_empty_boxes_not_matched(self):
        r=d.match_boxes([],[]);self.assertEqual(r['unique_matched_count'],0);self.assertIsNone(r['matched_min_iou'])
    def test_business_diagnostic_not_acceptance(self):
        _,q,_=fixture()
        q.update(status='diagnostic_completed_not_accepted',accepted_for_release=False,benchmark_authorized=False,
            original_numeric_gate='numeric_not_accepted',output_diagnostic_sha256='diagnostic',business_consistent_count=26)
        for row in q['rows']:row['business_consistent']=row.pop('passed')
        meta={k:q[k] for k in s.META}
        self.assertEqual(s.summarize_business_diagnostic(q,meta,'diagnostic')['business_diagnostic_consistent'],26)
        for kind in ('gate','authorize','duplicate','counter','false_decision','sha'):
            v=copy.deepcopy(q)
            if kind=='gate':v['original_numeric_gate']='passed'
            if kind=='authorize':v['benchmark_authorized']=True
            if kind=='duplicate':v['rows'][0]=copy.deepcopy(v['rows'][1])
            if kind=='counter':v['candidate_execution']['errors']=1
            if kind=='false_decision':v['rows'][0]['strict_equal']['NER']=False
            if kind=='sha':v['output_diagnostic_sha256']='wrong'
            with self.subTest(kind=kind),self.assertRaises(ValueError):s.summarize_business_diagnostic(v,meta,'diagnostic')


if __name__=='__main__':unittest.main()
