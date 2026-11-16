import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import benchmark_rec_wide as b

ROLES=('det','rec','layout','cell','enhance','yolo','ner')


class Tests(unittest.TestCase):
    @staticmethod
    def fixture():
        cases=[]
        for i in range(26):
            cases.append(dict(index=i,image_sha256=str(i//2),image_type=(3,1)[i%2],signature={'sha':str(i)},
                counters={k:dict(calls=10 if k=='rec' else 1,cpu_fallback_calls=2 if k=='rec' and i in (14,15) else 0) for k in ROLES}))
        plan=dict(rules=copy.deepcopy(b.w.RULES),cases=cases)
        run=dict(rules=copy.deepcopy(b.w.RULES),status='measured_not_summarized',parent_pid=99,startup_environment={},workers=[])
        reports={}
        for j,(r,v) in enumerate(b.p.ORDER):
            run['workers'].append(dict(round=r,variant=v,pid=100+j,exitcode=0))
            x=dict(rules=copy.deepcopy(b.w.RULES),status='measured',round=r,variant=v,pid=100+j,parent_pid=99,
                spawn_method='spawn',startup_environment={},warmup=[],rows=[],before_load={'memory':{'NvMapMemUsed_kib':1024}},
                after_warmup={'memory':{'NvMapMemUsed_kib':103424}},warm_nvmap_delta_mib=100.,
                wide_execution=dict(calls=12,errors=0,profiles={'x':b.w.PROFILE}) if v=='candidate' else None)
            x['after_setup_execution']={k:dict(calls=1,errors=0,cpu_fallback=dict(calls=0,errors=0)) for k in ROLES}
            running=copy.deepcopy(x['after_setup_execution'])
            for phase,cycles in (('warmup',1),('rows',2)):
                for c in range(cycles):
                    for i in b.p.page_indices(r):
                        row=copy.deepcopy(cases[i]);row.update(cycle=c,wall_ms=90. if v=='candidate' else 100.,
                            counters=b.w.expected_counters(cases[i],v),wide_calls=2 if v=='candidate' and i in (14,15) else 0)
                        x[phase].append(row)
                        for k,counts in row['counters'].items():
                            running[k]['calls']+=counts['calls'];running[k]['cpu_fallback']['calls']+=counts['cpu_fallback_calls']
                x['after_warmup_execution' if phase=='warmup' else 'execution']=copy.deepcopy(running)
            x['timing']=b.timing(x['rows']);reports[(r,v)]=x
        return plan,run,reports

    def test_two_round_full_pipeline_gate(self):
        result=b.summarize(*self.fixture())
        self.assertEqual(result['status'],'wide_pipeline_gate_passed_not_promoted')
        self.assertFalse(result['accepted_for_release'])
        self.assertEqual(result['measured_pages'],208)

    def test_reverse_second_round(self):
        f=self.fixture();x=f[2][('r2','candidate')]
        self.assertEqual([v['index'] for v in x['warmup']],list(range(25,-1,-1)))

    def test_incomplete_or_drift_rejected(self):
        for kind in ('order','pid','env','coverage','signature','counts','wide','errors','total','profile','negative','stored'):
            f=self.fixture();x=f[2][('r2','candidate')];row=x['rows'][0]
            if kind=='order':f[1]['workers'].reverse()
            if kind=='pid':x['pid']=99
            if kind=='env':x['startup_environment']={'CUDA_VISIBLE_DEVICES':'-1'}
            if kind=='coverage':x['warmup'].pop()
            if kind=='signature':row['signature']={}
            if kind=='counts':row['counters']['rec']['cpu_fallback_calls']=1
            if kind=='wide':x['wide_execution']['calls']=0
            if kind=='errors':x['execution']['rec']['errors']=1
            if kind=='total':x['execution']['ner']['calls']+=1
            if kind=='profile':x['wide_execution']['profiles']={}
            if kind=='negative':row['wall_ms']=-1
            if kind=='stored':x['timing']['overall']['mean_ms']=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):b.summarize(*f)

    def test_each_round_must_pass(self):
        f=self.fixture();x=f[2][('r2','candidate')]
        for z in x['rows']:z['wall_ms']=101.
        x['timing']=b.timing(x['rows'])
        self.assertEqual(b.summarize(*f)['status'],'wide_pipeline_gate_not_met')

    def test_unaffected_pages_cannot_regress(self):
        f=self.fixture();x=f[2][('r1','candidate')];ref=f[2][('r1','reference')]
        for z in ref['rows']:z['wall_ms']=100. if z['index'] not in (14,15) else 1000.
        ref['timing']=b.timing(ref['rows'])
        for z in x['rows']:z['wall_ms']=106. if z['index'] not in (14,15) else 10.
        x['timing']=b.timing(x['rows']);result=b.summarize(*f)
        self.assertGreater(result['rounds'][0]['mean_gain_fraction'],.05)
        self.assertFalse(result['rounds'][0]['performance_gate_passed'])

    def test_memory_and_baseline_limits(self):
        for mode in ('extra','spread'):
            f=self.fixture();x=f[2][('r2','candidate')]
            x['after_warmup']['memory']['NvMapMemUsed_kib']+=400*1024
            if mode=='spread':x['before_load']['memory']['NvMapMemUsed_kib']+=40*1024
            x['warm_nvmap_delta_mib']=b.p.b.warm_nvmap_mib(x)
            self.assertEqual(b.summarize(*f)['status'],'wide_pipeline_gate_not_met')


if __name__=='__main__':unittest.main()
