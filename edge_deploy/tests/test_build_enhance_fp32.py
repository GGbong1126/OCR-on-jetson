from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'jetson'))
import build_enhance_fp32 as b


class Tests(unittest.TestCase):
    def test_contract(self):
        b.validate_contract([('images',[1,3,512,512],'DataType.FLOAT')], [('output',[1,1,512,512],'DataType.FLOAT')])
    def test_reject_changed_batch_precision_shape_name(self):
        ref=[('output',[1,1,512,512],'DataType.FLOAT')]
        for item in [('images',[2,3,512,512],'DataType.FLOAT'),('images',[1,3,256,512],'DataType.FLOAT'),
                     ('images',[1,3,512,512],'DataType.HALF'),('x',[1,3,512,512],'DataType.FLOAT')]:
            with self.assertRaises(ValueError): b.validate_contract([item],ref)
    def test_reject_changed_output(self):
        with self.assertRaises(ValueError):
            b.validate_contract([('images',[1,3,512,512],'DataType.FLOAT')], [('output',[1,3,512,512],'DataType.FLOAT')])


if __name__ == '__main__': unittest.main()
