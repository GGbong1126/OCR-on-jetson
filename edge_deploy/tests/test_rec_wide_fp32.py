import ast
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import rec_wide_fp32 as w


class Fake:
    def __init__(self,profile):
        self.profiles={'x':profile}; self.calls=self.errors=0; self.closed=False; self.received=[]
    def in_profile(self,feeds):
        if set(feeds)!={'x'}: raise ValueError('names')
        s=feeds['x'].shape; lo,_,hi=self.profiles['x']
        return len(s)==4 and all(a<=v<=b for v,a,b in zip(s,lo,hi))
    def run(self,names,feeds):
        self.calls+=1; self.received.append(feeds); return [feeds['x']]
    def stats(self):return dict(calls=self.calls,errors=self.errors,profiles=self.profiles)
    def close(self):self.closed=True


class Tests(unittest.TestCase):
    def route(self):return w.Route(Fake(w.old.PROFILE),Fake(w.PROFILE))
    def test_original_inputs_routed_without_copy(self):
        r=self.route()
        for width in (32,1536,1537,1622,1695,1697,2081,2304):
            x=np.zeros((1,3,48,width),np.float32); feed={'x':x}
            self.assertTrue(r.in_profile(feed)); self.assertIs(r.run(None,feed)[0],x)
            target=r.base if width<=1536 else r.wide
            self.assertIs(target.received[-1],feed)
        self.assertEqual((r.base.calls,r.wide.calls),(2,6))
        self.assertEqual(r.stats()['calls'],8)

    def test_outside_union_left_to_outer_cpu(self):
        r=self.route()
        for shape in ((1,3,48,31),(1,3,48,2305),(2,3,48,1696),(1,3,49,1696)):
            feed={'x':np.zeros(shape,np.float32)}
            self.assertFalse(r.in_profile(feed))
            with self.assertRaises(ValueError):r.run(None,feed)
        self.assertEqual((r.base.calls,r.wide.calls),(0,0))

    def test_error_never_falls_back(self):
        r=self.route(); feed={'x':np.zeros((1,3,48,1700),np.float32)}
        def failed(names,feeds):r.wide.errors+=1; raise RuntimeError('kernel')
        r.wide.run=failed
        with self.assertRaises(RuntimeError):r.run(None,feed)
        with self.assertRaises(RuntimeError):r.run(None,{'x':np.zeros((1,3,48,160),np.float32)})
        self.assertEqual(r.base.calls,0)

    def test_closed_session_rejected_and_both_closed(self):
        r=self.route(); r.wide.closed=True
        with self.assertRaises(RuntimeError):r.run(None,{'x':np.zeros((1,3,48,160),np.float32)})
        r.close(); self.assertTrue(r.base.closed)

    def test_close_base_even_on_wide_error(self):
        r=self.route()
        def failed():raise RuntimeError('close')
        r.wide.close=failed
        with self.assertRaises(RuntimeError):r.close()
        self.assertTrue(r.base.closed)

    def test_expected_counts_move_only_cpu_rec(self):
        case={'counters':{'rec':{'calls':50,'cpu_fallback_calls':2},'det':{'calls':1,'cpu_fallback_calls':0}}}
        before=copy.deepcopy(case)
        self.assertEqual(w.expected_counters(case,'candidate')['rec'],{'calls':52,'cpu_fallback_calls':0})
        self.assertEqual(w.expected_counters(case,'reference'),case['counters']); self.assertEqual(case,before)
        with self.assertRaises(ValueError):w.expected_counters(case,'bad')

    def test_visible_formats_and_coverage(self):
        info={'Layers':[{'Name':'conv','Inputs':[{'Format/Datatype':'Row major linear FP32'}],'Outputs':[{'Format/Datatype':'Float'}]}]}
        self.assertTrue(w.inspection(info,1)['passed'])
        self.assertFalse(w.inspection(info,2)['passed']); self.assertFalse(w.inspection({'Layers':[]},0)['passed'])
        for dtype in ('Half','FP16','Int8'):
            info['Layers'][0]['Outputs'][0]['Format/Datatype']=dtype
            self.assertFalse(w.inspection(info,1)['passed'])

    def test_frozen_profile_and_no_quantization(self):
        self.assertEqual(w.PROFILE,[[1,3,48,1537],[1,3,48,1696],[1,3,48,2304]])
        self.assertFalse(any(w.RULES[k] for k in ('fp16','int8','tf32','accepted_for_release')))
        self.assertEqual(w.BOUNDARIES,[1537,1538,1696,2303,2304])

    def test_existing_outer_adapter_uses_union_then_original_cpu(self):
        # Exercise the actual unchanged adapter classes without loading Windows
        # torch/ORT/CUDA runtimes. Only backend work is faked.
        source=Path(w.__file__).with_name('jetson_models.py')
        if not source.is_file():
            source=w.q.RELEASE/'edge_deploy/jetson/jetson_models.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        classes=[v for v in tree.body if isinstance(v,ast.ClassDef) and v.name in ('CheckedCPU','OCRSession')]
        cpu_inputs=[]; created=[]
        class CPU:
            def run(self,names,feeds):cpu_inputs.append(feeds['x']);return [feeds['x']]
        def factory(path):created.append(path);return CPU()
        scope={'np':np,'cpu_session':factory}
        exec(compile(ast.Module(body=classes,type_ignores=[]),str(source),'exec'),scope)
        route=self.route();route.base.role='rec'
        outer=scope['OCRSession'](route,Path('original.onnx'))
        for width in (1536,1537,2081,2304,2305,2306):
            x=np.zeros((1,3,48,width),np.float32)
            self.assertIs(outer.run(None,{'x':x})[0],x)
        self.assertEqual((route.base.calls,route.wide.calls,len(cpu_inputs),len(created)),(1,3,2,1))
        self.assertEqual(outer.stats()['cpu_fallback']['calls'],2)
        route.wide.errors=1
        with self.assertRaises(RuntimeError):outer.run(None,{'x':np.zeros((1,3,48,2305),np.float32)})
        self.assertEqual(len(cpu_inputs),2)


if __name__=='__main__':unittest.main()
