import ast
import math
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'ipc'))
from runtime import rec_functions


class SplitTests(unittest.TestCase):
    def test_original_preprocess_and_ctc_reused_exactly(self):
        import cv2
        source=Path(__file__).resolve().parents[4]/'server.py'
        if not source.exists(): source=Path('/home/xchq/work/autoocr/releases/jetson-r2/server.py')
        tree=ast.parse(source.read_text(encoding='utf-8'))
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='_ONNXOCREngine')
        f=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_rec_run')
        module=ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])); namespace=dict(np=np,cv2=cv2,math=math)
        exec(compile(module,str(source),'exec'),namespace)
        feeds=[]
        output=np.zeros((1,9,6),np.float32)
        for i,index in enumerate((0,1,1,0,1,2,2,3,0)): output[0,i,index]=.9
        def run(names,value): feeds.append(value['x'].copy()); return [output]
        engine=SimpleNamespace(_char_dict=['blank','甲','乙',' ','x','y'],
                               rec_session=SimpleNamespace(run=run,get_inputs=lambda:[SimpleNamespace(name='x')]))
        engine._rec_run=MethodType(namespace['_rec_run'],engine)
        prepare,decode=rec_functions(engine)
        rng=np.random.RandomState(164)
        for h,w in ((48,1),(48,31),(48,32),(31,99),(48,1536),(48,2081),(48,3201)):
            image=rng.randint(0,256,(h,w,3),np.uint8)
            expected=engine._rec_run(image); candidate=prepare(image)
            self.assertTrue(np.array_equal(feeds[-1],candidate['x']))
            self.assertEqual(expected,decode(output))
            self.assertEqual(candidate['x'].dtype,np.float32)
            self.assertEqual(candidate['x'].shape[:3],(1,3,48))


if __name__=='__main__': unittest.main()
