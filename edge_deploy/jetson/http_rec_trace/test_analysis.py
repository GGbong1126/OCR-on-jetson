import json
from pathlib import Path
import unittest
from analysis import analyze


M=dict(owners={str(i):dict(context_id=1,stream_ids=dict(narrow=i*2+10,wide=i*2+11)) for i in range(4)})


def trace(rows):
    header='"Start","Duration","Context","Stream","Name"\n"us","us","","",""\n'
    return header+'\n'.join(','.join(map(str,r)) for r in rows)


class Tests(unittest.TestCase):
    def test_four_real_owners_and_witness(self):
        r=analyze(trace([(i,10-i*2,1,i*2+10,'conv') for i in range(4)]),M)
        self.assertEqual(r['max_distinct_rec_owners_over_threshold'],4)
        self.assertEqual(r['at_least_us'][4],4)
        self.assertEqual({k['owner'] for k in r['peak_witness']},{0,1,2,3})
    def test_wide_narrow_same_owner_not_double_counted(self):
        rows=[(0,10,1,10,'conv'),(1,8,1,11,'conv'),(10,2,1,12,'conv'),(12,2,1,14,'conv'),(14,2,1,16,'conv')]
        r=analyze(trace(rows),M); self.assertEqual(r['max_distinct_rec_owners_over_threshold'],1)
    def test_copy_other_model_and_context_excluded(self):
        rows=[(i*10,5,1,i*2+10,'conv') for i in range(4)]
        rows.extend([(0,99,1,12,'[CUDA memcpy HtoD]'),(0,99,1,99,'other'),(0,99,2,12,'wrong_context')])
        r=analyze(trace(rows),M); self.assertEqual(r['max_distinct_rec_owners_over_threshold'],1)
        self.assertEqual(r['mapped_copy_count'],1); self.assertEqual(r['other_model_kernel_count'],1)
    def test_denied_and_missing_owner_refused(self):
        with self.assertRaises(ValueError): analyze('ERR_NVGPUCTRPERM',M)
        with self.assertRaises(AssertionError): analyze(trace([(0,10,1,10,'conv')]),M)
    def test_touch_and_tiny_overlap_not_peak(self):
        r=analyze(trace([(0,1,1,10,'conv'),(.95,1,1,12,'conv'),(2,1,1,14,'conv'),(3,1,1,16,'conv')]),M)
        self.assertEqual(r['max_distinct_rec_owners_over_threshold'],1)
    def test_inference_method_is_not_overridden(self):
        import ast
        tree=ast.parse((Path(__file__).parent/'backend_trace.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        self.assertEqual([n.name for n in cls.body if isinstance(n,ast.FunctionDef)],['__init__'])
    def test_no_standard_profile_shadow(self):
        self.assertFalse((Path(__file__).parent/'profile.py').exists())
        import cProfile
        self.assertTrue(callable(cProfile.run))


if __name__=='__main__': unittest.main()
