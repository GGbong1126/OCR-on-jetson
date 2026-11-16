import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import summarize_layout_semantic as s
from test_layout_semantic_trial import row


def fixture():
    meta=dict.fromkeys(s.META,'sha');rows=[]
    for i in range(26):
        r=row();r.update(image_sha256=str(i//2),image_type=(3,1)[i%2],passed=True,
                        raw_numeric_diagnostic={'output':dict(allclose_diagnostic=True)})
        rows.append(r)
    plan=dict(meta,rules=copy.deepcopy(s.m.RULES),image_cases=[{k:r[k] for k in ('image_sha256','image_type')} for r in rows])
    q=dict(meta,rows=rows,status='semantic_business_consistent_not_gt',old_raw_gate='numeric_not_accepted',candidate_execution=dict(calls=26,errors=0))
    inputs=[dict(index=i) for i in range(6)];reports=[]
    for j,(variant,rnd) in enumerate((('reference','r1'),('candidate','r1'),('candidate','r2'),('reference','r2'))):
        ms=10. if variant=='reference' else 8.
        reports.append(dict(meta,status='measured',variant=variant,round=rnd,warmup=3,repeat=10,
            rows=[dict(cycle=c,index=i,wall_ms=ms) for c in range(10) for i in range(6)],
            timing=s.m.old.summary([ms]*60),execution=dict(calls=84,errors=0),pre_post_outputs_exact=True,
            inputs=copy.deepcopy(inputs),started_ns=j*2,ended_ns=j*2+1,
            before=dict(memory={'NvMapMemUsed_kib':1000}),after_warmup=dict(memory={'NvMapMemUsed_kib':103400})))
    return plan,q,reports,inputs


class Tests(unittest.TestCase):
    def test_success_still_not_release(self):
        r=s.summarize(*fixture());self.assertFalse(r['accepted_for_release'])
        self.assertEqual(r['old_raw_gate'],'numeric_not_accepted');self.assertEqual(r['benchmark_calls'],240)
        self.assertEqual(r['status'],'eligible_for_full_page_performance_test')
    def test_quality_failure_cannot_time(self):
        p,q,b,x=fixture();q['rows'][0]['strict_equal']['NER']=False;q['rows'][0]['passed']=False;q['status']='quality_not_accepted'
        self.assertEqual(s.summarize(p,q,[],x)['status'],'quality_not_accepted')
        with self.assertRaises(ValueError):s.summarize(p,q,b,x)
    def test_order_overlap_samples_hashes_errors(self):
        for kind in ('order','overlap','sequence','calls','errors','nan','input','meta','duplicate','changed_policy','history','false_pass'):
            p,q,b,x=fixture()
            if kind=='order':b[0]['started_ns']=10;b[0]['ended_ns']=11
            if kind=='overlap':b[0]['ended_ns']=3
            if kind=='sequence':b[0]['rows'].reverse()
            if kind=='calls':b[0]['execution']['calls']=83
            if kind=='errors':q['candidate_execution']['errors']=1
            if kind=='nan':b[0]['rows'][0]['wall_ms']=float('nan')
            if kind=='input':b[0]['inputs'][0]={'bad':'input'}
            if kind=='meta':b[0]['plan_sha256']='changed'
            if kind=='duplicate':q['rows'][0]=copy.deepcopy(q['rows'][1])
            if kind=='changed_policy':p['rules']['max_coordinate_delta_pixels']=1
            if kind=='history':q['old_raw_gate']='passed'
            if kind=='false_pass':q['rows'][0]['effective_match']['matched_max_coordinate_delta']=2
            with self.subTest(kind=kind),self.assertRaises(ValueError):s.summarize(p,q,b,x)
    def test_insufficient_speed_or_memory(self):
        p,q,b,x=fixture()
        for r in b:
            if r['variant']=='candidate':
                for v in r['rows']:v['wall_ms']=9.9
                r['timing']=s.m.old.summary([9.9]*60)
        self.assertEqual(s.summarize(p,q,b,x)['status'],'performance_or_resource_not_accepted')
        p,q,b,x=fixture();b[1]['after_warmup']['memory']['NvMapMemUsed_kib']+=257*1024
        self.assertEqual(s.summarize(p,q,b,x)['status'],'performance_or_resource_not_accepted')


if __name__=='__main__':unittest.main()
