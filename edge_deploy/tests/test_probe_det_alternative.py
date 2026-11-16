from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import probe_det_alternative as m


def row(i,impl,tactic,ms):
    return dict(index=i,implementation=impl,tactic=tactic,timing_ms=ms,workspace_bytes=100)


class Tests(unittest.TestCase):
    def test_other_implementation_not_just_other_tactic(self):
        rows=[row(0,4,1,2),row(1,4,2,1),row(2,5,3,3),row(3,6,4,4)]
        self.assertEqual(m.choose(rows,1),(4,2))

    def test_missing_or_ambiguous_is_failure(self):
        for rows in ([],[row(0,1,2,1)],[row(0,1,1,1),row(1,2,1,2)]):
            with self.assertRaises(ValueError):m.choose(rows,1)

    def test_no_alternative_is_failure(self):
        with self.assertRaises(ValueError):m.choose([row(0,4,1,1),row(1,4,2,2)],1)

    def test_signed_tactic(self):
        self.assertEqual(m.unsigned(-1),(1<<64)-1)
        self.assertEqual(m.choose([row(0,4,-1,1),row(1,5,2,2)],(1<<64)-1),(4,1))


if __name__=='__main__':unittest.main()
