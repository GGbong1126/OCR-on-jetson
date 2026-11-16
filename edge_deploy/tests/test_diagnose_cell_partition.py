from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import diagnose_cell_partition as d


class Tests(unittest.TestCase):
    def test_permutation_not_displacement(self):
        x=np.array([[0,.9,0,0,10,10],[0,.8,100,100,120,120]])
        r=d.match(x,x[::-1])
        self.assertEqual(r['matched_count'],2);self.assertEqual(r['reordered_matched_count'],2)
        self.assertEqual(r['max_matched_coordinate_delta_pixels'],0.)

    def test_extra_and_empty(self):
        x=np.array([[0,.9,0,0,10,10],[0,.8,100,100,120,120]])
        r=d.match(x[:1],x)
        self.assertEqual(r['matched_count'],1);self.assertEqual(r['unmatched_candidate_indices'],[1])
        r=d.match(x[:0],x[:0]);self.assertIsNone(r['min_matched_iou'])

    def test_ambiguous_not_double_counted(self):
        x=np.array([[0,.9,0,0,10,10],[0,.9,0,0,10,10]])
        self.assertEqual(d.match(x,x)['matched_count'],0)

    def test_original_threshold_and_scaling(self):
        out=dict(fetch_name_0=np.array([[0,.9,0,0,10,10],[0,.2,100,100,120,120]],np.float32),fetch_name_1=np.array([2],np.int32))
        r=d.effective(out,{'im_shape':np.array([[1280,640]])})
        self.assertEqual(r.shape,(1,6));self.assertEqual(r[0,-1],20.)


if __name__=='__main__':unittest.main()
