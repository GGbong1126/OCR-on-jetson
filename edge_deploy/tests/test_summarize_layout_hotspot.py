import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import summarize_layout_hotspot as s


def fixture():
    meta=dict.fromkeys(s.META,'sha')
    values=[dict(image_sha256=str(i//2),image_type=(3,1)[i%2],inputs={'image':str(i)},
                 reference_sha256={'a':str(i)},raw={'a':dict(all_finite=True)},repeat_exact=True) for i in range(6)]
    check=dict(meta,status='finite_same_inputs_compared_not_business_accepted',rows=values,execution=dict(calls=12,errors=0))
    rows=[dict(image_sha256=str(i//2),image_type=(3,1)[i%2],passed=True,
        strict_equal=dict.fromkeys(('FullText','Text','MarkdownBodyLines','NER','raw_line_texts'),True),
        fields=dict(api_special_text_equal=dict.fromkeys(('title','category_no','page_no'),True)),
        number_symbol_sequence_equal=True,same_original_input=True,boxes_exactly_equal=True,
        cpu_fallback_calls={v:dict(det=0,rec=0) for v in ('reference','candidate')}) for i in range(26)]
    quality=dict(meta,status='all_26_business_consistent_not_gt',rows=rows,candidate_execution=dict(calls=26,errors=0))
    reports=[]
    for v in ('reference','candidate'):
        for r in ('r1','r2'):
            ms=10. if v=='reference' else 8.
            reports.append(dict(meta,status='measured',variant=v,round=r,warmup=3,repeat=10,
                execution=dict(calls=84,errors=0),pre_post_outputs_exact=True,
                inputs=[{k:x[k] for k in ('image_sha256','image_type','inputs','reference_sha256')} for x in values],
                rows=[dict(cycle=c,index=i,wall_ms=ms) for c in range(10) for i in range(6)],
                timing=s.e.old.summary([ms]*60),before=dict(memory={'NvMapMemUsed_kib':1000}),
                after_warmup=dict(memory={'NvMapMemUsed_kib':1000+102400})))
    return check,quality,reports


class Tests(unittest.TestCase):
    def test_success_not_release(self):
        result=s.summarize(*fixture())
        self.assertEqual(result['status'],'eligible_for_full_page_performance_test')
        self.assertEqual(result['benchmark_calls'],240);self.assertFalse(result['accepted_for_release'])
    def test_no_benchmark_on_failure(self):
        c,q,b=fixture();q['rows'][0]['strict_equal']['FullText']=False;q['rows'][0]['passed']=False
        q['status']='differences_not_accepted'
        self.assertEqual(s.summarize(c,q,[])['status'],'quality_not_accepted')
        with self.assertRaises(ValueError):s.summarize(c,q,b)
    def test_quality_pending_timing(self):
        c,q,_=fixture();self.assertEqual(s.summarize(c,q,[])['status'],'quality_passed_performance_pending')
    def test_bad_samples_or_handoffs_rejected(self):
        for kind in ('duplicate','errors','hash','calls','rows','metadata','nan','false_gate'):
            c,q,b=fixture()
            if kind=='duplicate':q['rows'][0]=copy.deepcopy(q['rows'][1])
            if kind=='errors':q['candidate_execution']['errors']=1
            if kind=='hash':b[0]['inputs'][0]['reference_sha256']={'a':'wrong'}
            if kind=='calls':b[0]['execution']['calls']=83
            if kind=='rows':b[0]['rows'].reverse()
            if kind=='metadata':b[0]['plugin_sha256']='wrong'
            if kind=='nan':b[0]['rows'][0]['wall_ms']=float('nan')
            if kind=='false_gate':q['rows'][0]['passed']=False
            with self.subTest(kind=kind),self.assertRaises(ValueError):s.summarize(c,q,b)
    def test_insufficient_gain_is_not_promoted(self):
        c,q,b=fixture()
        for r in b:
            if r['variant']=='candidate':
                for row in r['rows']:row['wall_ms']=9.9
                r['timing']=s.e.old.summary([9.9]*60)
        self.assertEqual(s.summarize(c,q,b)['status'],'performance_or_resource_not_accepted')
    def test_extra_memory_gate(self):
        c,q,b=fixture()
        b[2]['after_warmup']['memory']['NvMapMemUsed_kib']+=257*1024
        self.assertEqual(s.summarize(c,q,b)['status'],'performance_or_resource_not_accepted')


if __name__=='__main__':unittest.main()
