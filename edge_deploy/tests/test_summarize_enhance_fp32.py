import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import summarize_enhance_fp32 as s


def fixtures():
    meta=dict.fromkeys(s.META,'sha')
    inputs=[dict(image_sha256=str(i//9),input_shape=[1,3,512,512],input_sha256=str(i),reference_sha256=str(i),repeat_exact=True) for i in range(27)]
    check=dict(meta,status='finite_original_tiles_not_business_accepted',rows=inputs,execution=dict(errors=0,calls=54))
    rows=[dict(image_sha256=str(i),image_type=1,passed=True,same_original_inputs=True,tile_calls=9,tile_input_sha256=['x']*9,
               strict_equal=dict.fromkeys(('FullText','Text','MarkdownBodyLines','NER','raw_line_texts','special'),True),
               field_diagnostics={'api_special_text_equal':dict.fromkeys(('title','category_no','page_no'),True)},
               full_text_diagnostics={'number_symbol_sequence_equal':True},boxes_exactly_equal=True,
               cpu_fallback_calls={v:dict(det=0,rec=0) for v in ('reference','candidate')}) for i in range(13)]
    control=copy.deepcopy(rows[0]); control.update(image_type=3,tile_calls=0,tile_input_sha256=[]); rows.insert(0,control)
    quality=dict(meta,status='all_cases_consistent_not_gt',rows=rows,candidate_execution={'errors':0})
    reports=[]
    for v in ('reference','candidate'):
        for rnd in ('r1','r2'):
            ms=100 if v=='reference' else 90
            reports.append(dict(meta,status='measured',variant=v,round=rnd,pre_post_outputs_exact=True,warmup=3,repeat=20,
                inputs=[s.e.describe(r) for r in inputs[:9]],execution=dict(errors=0,calls=216),
                rows=[dict(cycle=c,index=i,wall_ms=ms) for c in range(20) for i in range(9)],
                timing=s.e.common.summary([ms]*180),before={'memory':{'NvMapMemUsed_kib':1000}},
                after_warmup={'memory':{'NvMapMemUsed_kib':2000}}))
    return check,quality,reports


class Tests(unittest.TestCase):
    def test_valid_not_promoted(self):
        r=s.summarize(*fixtures()); self.assertEqual(r['benchmark_calls'],720)
        self.assertEqual(r['status'],'eligible_for_full_page_performance_test'); self.assertFalse(r['accepted_for_release'])
    def test_reject_quality_before_timing(self):
        c,q,b=fixtures(); q['rows'][1]['strict_equal']['FullText']=False; q['rows'][1]['passed']=False
        q['status']='differences_not_accepted'
        self.assertEqual(s.summarize(c,q,[])['benchmark_calls'],0)
        with self.assertRaises(ValueError): s.summarize(c,q,b)
    def test_extra_fallback_blocks_timing(self):
        c,q,b=fixtures(); q['rows'][1]['cpu_fallback_calls']['candidate']['rec']=1
        self.assertFalse(s.e.quality_ready(q['rows']))
        self.assertEqual(s.summarize(c,q,[])['status'],'quality_or_fallback_not_accepted')
        with self.assertRaises(ValueError): s.summarize(c,q,b)
    def test_missing_changed_timing_rejected(self):
        for kind in ('count','order','input','hash','stats'):
            c,q,b=fixtures()
            if kind=='count': b.pop()
            if kind=='order': b[0]['rows'].reverse()
            if kind=='input': b[0]['inputs'][0]['input_sha256']='changed'
            if kind=='hash': b[0]['engine_sha256']='changed'
            if kind=='stats': b[0]['timing']['mean_ms']=1
            with self.assertRaises(ValueError): s.summarize(c,q,b)
    def test_reject_weak_speed_or_memory(self):
        for kind in ('speed','memory'):
            c,q,b=fixtures()
            if kind=='speed':
                b[2]['rows']=[dict(r,wall_ms=99) for r in b[2]['rows']]
                b[2]['timing']=s.e.common.summary([99]*180)
            else: b[2]['after_warmup']['memory']['NvMapMemUsed_kib']+=300*1024
            self.assertEqual(s.summarize(c,q,b)['status'],'performance_or_resource_not_accepted')


if __name__=='__main__': unittest.main()
