import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'jetson'))
import summarize_rec_short as m


def fixtures():
    meta = dict(script_sha256='s', build_sha256='b', engine_sha256='e', deployment_sha256='d')
    check = dict(meta, status='short_ctc_consistent_not_gt', rows=[{'token_sequence_equal':True}], execution={'errors':0})
    row = dict(passed=True, image_sha256='img', image_type=3,
               strict_equal=dict.fromkeys(('FullText','Text','MarkdownBodyLines','NER','raw_line_texts'),True),
               fields={'api_special_text_equal':dict.fromkeys(('title','category_no','page_no'),True)})
    evaluation = dict(meta, status='all_26_business_consistent_not_gt',
                      rows=[dict(copy.deepcopy(row), image_sha256=str(i), image_type=t) for i in range(13) for t in (3,1)],
                      short_execution={'errors':0}, broad_execution={'errors':0})
    reports=[]
    for v in ('reference','routed'):
        for r in ('r1','r2'):
            ms=10 if v=='reference' else 9
            rows=[dict(cycle=c,index=i,wall_ms=ms) for c in range(20) for i in range(50)]
            reports.append(dict(meta, variant=v, round=r, status='measured', pre_post_tokens_equal=True,
                execution={'errors':0}, image_sha256='im', input_shapes=[[1,3,48,32]]*50, input_sha256=['x']*50,
                warmup=3, repeat=20, rows=rows, timing=m.old.summary([ms]*1000),
                before={'memory':{'NvMapMemUsed_kib':1000}},
                after_warmup={'memory':{'NvMapMemUsed_kib':2000 if v=='reference' else 3000}},
                after={'memory':{'NvMapMemUsed_kib':3000}}))
    return check,evaluation,reports


class Tests(unittest.TestCase):
    def test_gates_pass(self):
        self.assertEqual(m.summarize(*fixtures())['status'],'eligible_for_full_page_performance_test')

    def test_no_gain_not_selected(self):
        c,e,rs=fixtures()
        for r in rs:
            for v in r['rows']: v['wall_ms']=10
            r['timing']=m.old.summary([10]*1000)
        self.assertEqual(m.summarize(c,e,rs)['status'],'not_selected_for_release')

    def test_quality_and_identity_fail(self):
        c,e,rs=fixtures(); e['rows'][0]['fields']['api_special_text_equal']['page_no']=False
        with self.assertRaises(ValueError): m.summarize(c,e,rs)
        c,e,rs=fixtures(); rs[1]['input_sha256'][0]='changed'
        with self.assertRaises(ValueError): m.summarize(c,e,rs)

    def test_incomplete_and_wrong_stats(self):
        c,e,rs=fixtures();rs[0]['rows'].pop()
        with self.assertRaises(ValueError): m.summarize(c,e,rs)
        c,e,rs=fixtures();rs[0]['timing']['mean_ms']=0
        with self.assertRaises(ValueError): m.summarize(c,e,rs)

    def test_memory_budget(self):
        c,e,rs=fixtures();rs[2]['after_warmup']['memory']['NvMapMemUsed_kib']+=500*1024
        self.assertEqual(m.summarize(c,e,rs)['status'],'not_selected_for_release')


if __name__=='__main__': unittest.main()
