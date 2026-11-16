import unittest
from inspect_trace import analyze


class TraceTests(unittest.TestCase):
    def test_cross_stream_kernel_and_copy_overlap_not_same_stream(self):
        raw='\n'.join(('"Start","Duration","Context","Stream","Name"','"us","us","","",""',
                      '0,10,1,3,det_kernel','2,4,1,4,rec_kernel','3,1,1,5,[CUDA memcpy HtoD]',
                      '4,1,1,4,rec_same_stream','0,10,2,6,other_context'))
        r=analyze(raw,dict(context_id=1,stream_ids=dict(det_n1=3,rec0_narrow=4,rec1_narrow=5)))
        self.assertEqual(r['mapped_activities'],4)
        self.assertGreater(r['counts']['det_rec_kernel_pairs'],0)
        self.assertEqual(r['counts']['rec_rec_kernel_pairs'],0)
        self.assertGreater(r['counts']['copy_compute_pairs'],0)
    def test_denied_or_empty_never_reports_success(self):
        for raw in ('ERR_NVGPUCTRPERM','No kernels were profiled.'):
            with self.assertRaises(ValueError): analyze(raw,dict(context_id=1,stream_ids={}))


if __name__=='__main__': unittest.main()
