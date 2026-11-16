import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

import det_batch2 as batch


class BatchTests(unittest.TestCase):
    def test_odd_mixed_shapes_keep_all_ids(self):
        rows=[dict(id=str(i),input_shape=[1,3,32,w]) for i,w in enumerate((32,64,32,32,64))]
        self.assertEqual(batch.group_pairs(rows),[[0,2],[3],[1,4]])
        self.assertEqual(sorted(i for g in batch.group_pairs(rows) for i in g),list(range(5)))

    def test_duplicate_and_empty_rejected(self):
        for rows in ([],[dict(id='a',input_shape=[1,3,32,32])]*2):
            with self.assertRaises(ValueError):batch.group_pairs(rows)

    def test_invalid_shapes_dtype_and_nan(self):
        for x in (np.zeros((3,3,32,32),np.float32),np.zeros((1,3,33,32),np.float32),
                  np.zeros((1,3,32,32),np.float64),np.full((1,3,32,32),np.nan,np.float32)):
            with self.assertRaises(ValueError):batch.validate(x)

    def session(self):
        return SimpleNamespace(errors=0,closed=False,run=Mock(side_effect=lambda names,f:[f['x'][:,:1].copy()]))

    def test_batch2_and_tail_routes_and_owned_output(self):
        a,b=self.session(),self.session()
        x=np.zeros((2,3,32,32),np.float32);x[1]=1
        y=batch.route(a,b,x)
        self.assertEqual(float(y[1].mean()),1)
        self.assertEqual(a.run.call_count,0);self.assertEqual(b.run.call_count,1)
        batch.route(a,b,x[:1]);self.assertEqual(a.run.call_count,1)
        x[:]=3;self.assertEqual(float(y[0].mean()),0)

    def test_error_does_not_fall_back(self):
        a,b=self.session(),self.session();b.run.side_effect=RuntimeError('GPU')
        with self.assertRaises(RuntimeError):batch.route(a,b,np.zeros((2,3,32,32),np.float32))
        self.assertEqual(a.run.call_count,0)
        b.errors=1
        with self.assertRaises(RuntimeError):batch.route(a,b,np.zeros((1,3,32,32),np.float32))

    def test_output_contract(self):
        a,b=self.session(),self.session();b.run.side_effect=None
        b.run.return_value=[np.zeros((1,1,32,32),np.float32)]
        with self.assertRaises(ValueError):batch.route(a,b,np.zeros((2,3,32,32),np.float32))

    def test_benchmark_demux_preserves_task_ids(self):
        from benchmark_det_batch2 import cycle
        data=[(None,np.full((1,3,32,32),v,np.float32)) for v in range(3)]
        for candidate in (None,self.session()):
            result=cycle(self.session(),candidate,data,[[0,2],[1]])
            self.assertEqual([float(result[i].mean()) for i in range(3)],[0,1,2])

    def test_business_replay_exact_input_and_once_only(self):
        from evaluate_det_batch2 import replay
        x=np.zeros((1,3,32,32),np.float32);y=np.ones((1,1,32,32),np.float32)
        records=[{'tensor':x}];seen=set()
        result=replay(0,records,{0:y},seen,{'x':x})
        y[:]=0;self.assertEqual(float(result[0].mean()),1)
        with self.assertRaises(ValueError):replay(0,records,{0:y},seen,{'x':x})
        with self.assertRaises(AssertionError):replay(0,records,{0:y},set(),{'x':x+1})

    def test_business_batches_pair_distinct_images_per_type(self):
        from evaluate_det_batch2 import business_groups
        rows=[dict(id=str(i)+':'+str(kind),image_sha256=str(i),image_type=kind,input_shape=[1,3,32,32])
              for i in range(3) for kind in (3,1)]
        self.assertEqual(business_groups(rows),[[0,2],[4],[1,3],[5]])
        with self.assertRaises(ValueError):business_groups([dict(rows[0],image_type=2)])
        changed=[dict(row) for row in rows];changed[2]['image_sha256']='0'
        with self.assertRaisesRegex(ValueError,'different source images'):business_groups(changed)


if __name__=='__main__':unittest.main()
