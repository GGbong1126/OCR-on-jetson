import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import summarize_det_hotspot as m


def fixtures():
    meta=dict.fromkeys(m.META,'sha')
    inputs=[dict(image_sha256=str(i//2),image_type=3 if i%2==0 else 1,input_shape=[1,3,960,672],input_sha256=str(i),reference_sha256=str(i)) for i in range(6)]
    check=dict(meta,status='finite_same_inputs_compared_not_business_accepted',rows=[dict(r,repeat_exact=True) for r in inputs],execution={'errors':0})
    rows=[dict(image_sha256=str(i),image_type=t,passed=True,same_original_input=True,boxes_exactly_equal=True,number_symbol_sequence_equal=True,
               strict_equal=dict.fromkeys(('FullText','Text','MarkdownBodyLines','NER','raw_line_texts','special'),True),
               fields={'api_special_text_equal':dict.fromkeys(('title','category_no','page_no'),True)},
               cpu_fallback_calls={v:{'det':0,'rec':0} for v in ('reference','candidate')}) for i in range(13) for t in (3,1)]
    quality=dict(meta,status='all_26_business_consistent_not_gt',rows=rows,candidate_execution={'errors':0})
    reports=[]
    for v in ('reference','mixed'):
        for r in ('r1','r2'):
            ms=10 if v=='reference' else 8
            reports.append(dict(meta,status='measured',variant=v,round=r,pre_post_outputs_exact=True,execution={'errors':0},
                warmup=3,repeat=10,inputs=copy.deepcopy(inputs),rows=[dict(cycle=c,index=i,wall_ms=ms) for c in range(10) for i in range(6)],
                timing=m.old.summary([ms]*60),before={'memory':{'NvMapMemUsed_kib':100}},after_warmup={'memory':{'NvMapMemUsed_kib':200}}))
    return check,quality,reports


class Tests(unittest.TestCase):
    def failed_fixture(self):
        v={'abs_max':4.,'abs_gt_fp16_max':0}
        ranges=dict(status='fp32_ranges_measured_not_fp16_validated',source_sha256=m.build.SOURCE_SHA,
            nodes=[dict(name='Conv.87',op='Conv',input='in',output='out',parameter_ranges={'w':v})],
            rows=[{'ranges':{'in':v,'out':v}} for _ in range(6)])
        record=dict(status='failed',source_sha256=m.build.SOURCE_SHA,deployment_sha256=m.build.DEPLOYMENT_SHA,
            targets=['Conv.87'],profile=[list(v) for v in m.build.PROFILE],workspace_bytes=256*1024**2,
            flags=1025,build_s=947.2,constraints=[dict(name='Conv.87',kind='CONVOLUTION',policy='fp16_compute_fp32_output')])
        log='No implementation of layer Conv.87 obeys the requested constraints. kOBEY_PRECISION_CONSTRAINTS was set'
        return record,ranges,log,[]

    def test_failed_build_no_quality_or_performance_claim(self):
        r=m.summarize_failed_build(*self.failed_fixture())
        self.assertEqual(r['status'],'build_constraints_not_satisfied')
        self.assertEqual(r['candidate_inference_calls'],0)
        self.assertFalse(r['engine_created'])

    def test_failed_build_rejects_artifacts(self):
        r,g,l,_=self.failed_fixture()
        for n in ('det_hotspot.engine','check.json','benchmark_mixed_r1.json'):
            with self.assertRaises(ValueError):m.summarize_failed_build(r,g,l,[n])

    def test_failed_build_requires_specific_error_and_policy(self):
        r,g,l,a=self.failed_fixture()
        with self.assertRaises(ValueError):m.summarize_failed_build(r,g,'out of memory',a)
        r['flags']=1
        with self.assertRaises(ValueError):m.summarize_failed_build(r,g,l,a)

    def test_valid_not_release(self):
        r=m.summarize(*fixtures());self.assertEqual(r['status'],'eligible_for_full_page_performance_test');self.assertFalse(r['deployment_changed'])

    def test_failed_quality_no_benchmark(self):
        c,q,rs=fixtures();q['rows'][0]['strict_equal']['FullText']=False;q['rows'][0]['passed']=False;q['status']='differences_not_accepted'
        self.assertEqual(m.summarize(c,q,[])['status'],'quality_not_accepted')
        with self.assertRaises(ValueError):m.summarize(c,q,rs)

    def test_reordered_or_changed_input(self):
        c,q,rs=fixtures();rs[0]['rows'].reverse()
        with self.assertRaises(ValueError):m.summarize(c,q,rs)
        c,q,rs=fixtures();rs[0]['inputs'][0]['input_sha256']='other'
        with self.assertRaises(ValueError):m.summarize(c,q,rs)

    def test_extra_fallback_rejected(self):
        c,q,rs=fixtures();q['rows'][0]['cpu_fallback_calls']['candidate']['rec']=1
        self.assertEqual(m.summarize(c,q,rs)['status'],'performance_or_resource_not_accepted')

    def test_small_gain_or_bad_summary(self):
        c,q,rs=fixtures()
        for r in rs:
            r['timing']=m.old.summary([10]*60)
            for v in r['rows']:v['wall_ms']=10
        self.assertEqual(m.summarize(c,q,rs)['status'],'performance_or_resource_not_accepted')
        rs[0]['timing']['mean_ms']=1
        with self.assertRaises(ValueError):m.summarize(c,q,rs)


if __name__=='__main__':unittest.main()
