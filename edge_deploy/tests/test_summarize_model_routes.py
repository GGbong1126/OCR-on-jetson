import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import summarize_model_routes as m


def reports():
    a={'status':'profiled_exact_not_optimized_or_gt','role':'rec','round':'r1',
       'image_sha256':'image','engine_sha256':'engine','script_sha256':'script',
       'source_calls':1,'unique_shapes':1,'warmup':3,'repeat_bare':2,'repeat_profile':2,
       'bare_execution':{'errors':0},'profile_execution':{'errors':0},'context_device_bytes':100,
       'cases':[{'source_index':0,'inputs':{'x':{'shape':[1,3,48,32]}},'outputs_sha256':['o'],
                 'bare_output_exact':True,'profile_output_exact':True,'bare_wall_ms':[1,2],
                 'profile':{'layers':[{'name':'Conv.0','fraction_of_layer_sum':1}]}}]}
    b=copy.deepcopy(a); b['round']='r2'
    return a,b


class Tests(unittest.TestCase):
    def test_valid_pair(self):
        r=m.pair(*reports())
        self.assertEqual(r['bare_calls'],4)
        self.assertEqual(r['cases'][0]['layer_name_hint_fractions_mean'],{'conv_named':1})

    def test_foreign_node_not_mislabelled_conv(self):
        self.assertEqual(m.hint('{ForeignNode[Conv.0...Relu.1]}'),'foreign_node')
        self.assertEqual(m.hint('GridSample.1'),'grid_named')

    def test_input_and_engine_mismatch(self):
        a,b=reports();b['engine_sha256']='other'
        with self.assertRaises(ValueError): m.pair(a,b)
        a,b=reports();b['cases'][0]['outputs_sha256']=['changed']
        with self.assertRaises(ValueError): m.pair(a,b)

    def test_partial_or_invalid_timings(self):
        a,b=reports();b['cases'][0]['bare_wall_ms']=[float('nan'),2]
        with self.assertRaises(ValueError): m.pair(a,b)
        a,b=reports();b['cases'][0]['profile_output_exact']=False
        with self.assertRaises(ValueError): m.pair(a,b)


if __name__=='__main__': unittest.main()
