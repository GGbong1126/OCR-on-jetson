import os
import sys
import unittest

try:
    import numpy
except ImportError: numpy=None


@unittest.skipUnless(sys.platform=='linux' and numpy is not None,'Existing Jetson Linux/NumPy required')
class BenchmarkChecks(unittest.TestCase):
    def test_snapshot_is_bounded_process_and_numeric_resource_metadata(self):
        from verify_ocr_workers import resource_snapshot
        value=resource_snapshot([os.getpid(),None]); row=value['processes'][str(os.getpid())]
        self.assertGreater(row['VmRSS'],0); self.assertGreaterEqual(row['fd'],3)
        self.assertGreaterEqual(value['cpu_total'],value['cpu_idle'])
        self.assertGreater(value['memory_kib']['MemAvailable'],0)
    def test_latency_scope_and_finite_sample_summary(self):
        from verify_ocr_workers import latency
        value=latency([1,2,3]); self.assertEqual(value['count'],3)
        self.assertEqual(value['mean_ms'],2); self.assertEqual(value['p50_ms'],2)
        self.assertAlmostEqual(value['p95_ms'],2.9)


if __name__=='__main__': unittest.main()
