import unittest
from analyze import concurrency, analyze


class Tests(unittest.TestCase):
    def test_four_streams(self):
        r=concurrency([(0,10,1),(1,9,2),(2,8,3),(3,7,4)])
        self.assertEqual(r['max_rec_streams_over_threshold'],4)
        self.assertEqual(r['at_least_us'][4],4)

    def test_same_stream_not_double_counted(self):
        r=concurrency([(0,10,1),(1,9,1),(2,8,2)])
        self.assertEqual(r['max_rec_streams_over_threshold'],2)
        self.assertEqual(r['at_least_us'][2],6)

    def test_touch_and_tiny_not_peak(self):
        self.assertEqual(concurrency([(0,1,1),(1,2,2)])['max_rec_streams_over_threshold'],1)
        self.assertEqual(concurrency([(0,1,1),(.95,2,2)])['max_rec_streams_over_threshold'],1)

    def test_invalid(self):
        with self.assertRaises(ValueError): concurrency([(2,1,1)])
        with self.assertRaises(ValueError): analyze('ERR_NVGPUCTRPERM',{})
        with self.assertRaises(ValueError): analyze('',{})


if __name__=='__main__': unittest.main()
