from pathlib import Path
from types import SimpleNamespace as NS
import copy
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import layout_outputs_fp32 as b
import evaluate_layout_outputs as e
import summarize_layout_outputs as s


def fixture():
    meta=dict.fromkeys(s.META,'sha')
    values=[dict(image_sha256=str(i//2),image_type=(3,1)[i%2],inputs={'image':str(i)},
        reference_sha256={'a':str(i)},raw={k:dict(all_finite=True,allclose_diagnostic=True,exactly_equal=True) for k in b.KEEP},
        repeat_exact=True,passed=True) for i in range(6)]
    check=dict(meta,status='retained_outputs_checked',rows=values,execution=dict(calls=12,errors=0))
    rows=[dict(image_sha256=str(i//2),image_type=(3,1)[i%2],passed=True,
        strict_equal=dict.fromkeys(('FullText','Text','MarkdownBodyLines','NER','raw_line_texts'),True),
        fields=dict(api_special_text_equal=dict.fromkeys(('title','category_no','page_no'),True)),
        number_symbol_sequence_equal=True,same_original_input=True,boxes_exactly_equal=True,
        cpu_fallback_calls={v:dict(det=0,rec=0) for v in ('reference','candidate')}) for i in range(26)]
    quality=dict(meta,status='all_26_business_consistent_not_gt',rows=rows,candidate_execution=dict(calls=26,errors=0))
    benchmarks=[]
    for v in ('reference','candidate'):
        for r in ('r1','r2'):
            ms=10. if v=='reference' else 8.
            benchmarks.append(dict(meta,status='measured',variant=v,round=r,warmup=3,repeat=10,
                execution=dict(calls=84,errors=0),pre_post_outputs_exact=True,
                inputs=[{k:x[k] for k in ('image_sha256','image_type','inputs','reference_sha256')} for x in values],
                rows=[dict(cycle=c,index=i,wall_ms=ms) for c in range(10) for i in range(6)],
                timing=e.old.summary([ms]*60),before=dict(memory={'NvMapMemUsed_kib':1000}),
                after_warmup=dict(memory={'NvMapMemUsed_kib':103400})))
    return check,quality,benchmarks


class Tests(unittest.TestCase):
    def test_consumer_0_1_only(self):
        text='def _detect_layout_onnx():\n outputs = session.run(None,inputs)\n a=outputs[0]\n b=outputs[1][0] if len(outputs)>1 else 0\n'
        self.assertEqual(b.audit_consumer(text)['consumed_indices'],[0,1])
        for extra in (' return outputs[2]\n',' return outputs\n',' return outputs[i]\n',' sink(outputs)\n'):
            with self.assertRaises(ValueError): b.audit_consumer(text+extra)
    def test_retained_real_outputs_reordered(self):
        class Native:
            def run(self,names,feeds):
                self.assertion=names is None
                return [np.array([1]),np.array([2])]
            def get_outputs(self): return [NS(name=k) for k in reversed(b.KEEP)]
        native=Native(); out=e.retained_outputs(native,{})
        self.assertTrue(native.assertion); self.assertEqual([int(v[0]) for v in out],[2,1])
        native.get_outputs=lambda:[NS(name=k) for k in (*b.KEEP,b.DROP)]
        with self.assertRaises(ValueError): e.retained_outputs(native,{})
    def test_numeric_gate(self):
        row=fixture()[0]['rows'][0]; self.assertTrue(e.numeric_passed(row))
        for key in ('all_finite','allclose_diagnostic','exactly_equal'):
            changed=copy.deepcopy(row);changed['raw']['fetch_name_1'][key]=False
            self.assertFalse(e.numeric_passed(changed))
        row['repeat_exact']=False;self.assertFalse(e.numeric_passed(row))
    def test_inspection_requires_fp32_plugins_no_half(self):
        def tensors(kind): return [dict(Name='t',**{'Format/Datatype':kind})]
        info=dict(Layers=[dict(Name='GridSample.'+str(i),Inputs=tensors('Float'),Outputs=tensors('Float')) for i in range(18)])
        self.assertEqual(e.inspect(info)['fp32_grid_sample_count'],18)
        for kind in ('Half','Int8','BF16',''):
            changed=copy.deepcopy(info);changed['Layers'][0]['Outputs']=tensors(kind)
            with self.assertRaises(ValueError):e.inspect(changed)
        info['Layers'].pop()
        with self.assertRaises(ValueError):e.inspect(info)
    def test_success_is_not_release(self):
        r=s.summarize(*fixture());self.assertFalse(r['accepted_for_release'])
        self.assertEqual(r['status'],'eligible_for_full_page_performance_test');self.assertEqual(r['benchmark_calls'],240)
    def test_failed_business_no_timing(self):
        c,q,t=fixture();q['rows'][0]['strict_equal']['FullText']=False;q['rows'][0]['passed']=False;q['status']='differences_not_accepted'
        self.assertEqual(s.summarize(c,q,[])['status'],'quality_not_accepted')
        with self.assertRaises(ValueError):s.summarize(c,q,t)
    def test_invalid_samples_and_provenance(self):
        for kind in ('duplicate','errors','hash','calls','order','meta','nan','stored','numeric'):
            c,q,t=fixture()
            if kind=='duplicate':q['rows'][0]=copy.deepcopy(q['rows'][1])
            if kind=='errors':q['candidate_execution']['errors']=1
            if kind=='hash':t[0]['inputs'][0]['reference_sha256']={'bad':'hash'}
            if kind=='calls':t[0]['execution']['calls']=83
            if kind=='order':t[0]['rows'].reverse()
            if kind=='meta':t[0]['engine_sha256']='bad'
            if kind=='nan':t[0]['rows'][0]['wall_ms']=float('nan')
            if kind=='stored':q['rows'][0]['passed']=False
            if kind=='numeric':c['rows'][0]['raw']['fetch_name_0']['allclose_diagnostic']=False
            with self.subTest(kind=kind),self.assertRaises(ValueError):s.summarize(c,q,t)
    def test_performance_and_memory_gates(self):
        c,q,t=fixture()
        t[2]['after_warmup']['memory']['NvMapMemUsed_kib']+=257*1024
        self.assertEqual(s.summarize(c,q,t)['status'],'performance_or_resource_not_accepted')
        c,q,t=fixture()
        for r in t:
            if r['variant']=='candidate':
                for row in r['rows']: row['wall_ms']=9.9
                r['timing']=e.old.summary([9.9]*60)
        self.assertEqual(s.summarize(c,q,t)['status'],'performance_or_resource_not_accepted')


if __name__=='__main__':unittest.main()
