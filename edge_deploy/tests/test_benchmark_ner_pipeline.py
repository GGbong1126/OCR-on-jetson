import copy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import benchmark_ner_pipeline as b

ROLES=('det','rec','layout','cell','yolo','enhance','ner')


class Tests(unittest.TestCase):
    @staticmethod
    def fixture():
        cases=[]
        for i in range(26):
            expect=dict(inputs=[{'id':i}],outputs=[str(i)],business_sha256=str(i),cpu_fallback_calls={'det':0,'rec':0})
            cases.append(dict(index=i,image_sha256=str(i//2),image_type=(3,1)[i%2],expected={v:copy.deepcopy(expect) for v in ('reference','candidate')}))
        plan=dict(status='prepared',rules=copy.deepcopy(b.RULES),cases=cases,engines={v:dict(sha256=v) for v in ('reference','candidate')})
        run=dict(status='measured_not_summarized',rules=copy.deepcopy(b.RULES),parent_pid=99,startup_environment={},workers=[])
        reports={}
        def counts(c):return {k:dict(calls=c,errors=0,cpu_fallback=dict(calls=0,errors=0)) for k in ROLES}
        for j,(r,v) in enumerate(b.ORDER):
            run['workers'].append(dict(round=r,variant=v,pid=100+j,exitcode=0));report=dict(status='measured',rules=copy.deepcopy(b.RULES),round=r,variant=v,pid=100+j,parent_pid=99,
                spawn_method='spawn',startup_environment={},engine=plan['engines'][v],ner_loads=[v],warmup=[],rows=[],
                after_setup_execution=counts(1),after_warmup_execution=counts(27),execution=counts(79),
                before_load={'memory':{'NvMapMemUsed_kib':1024}},after_warmup={'memory':{'NvMapMemUsed_kib':103424}},warm_nvmap_delta_mib=100.)
            for phase,cycles in (('warmup',1),('rows',2)):
                for c in range(cycles):
                    for i in b.page_indices(r):
                        case=cases[i];report[phase].append(dict(index=i,cycle=c,image_sha256=case['image_sha256'],image_type=case['image_type'],
                            wall_ms=100. if v=='reference' else 90.,ner_wall_ms=10.,ner_inputs=[{'id':i}],ner_outputs=[str(i)],
                            signature=dict(business_sha256=str(i),structure_fields_sha256=str(i)),counters={k:dict(calls=1,cpu_fallback_calls=0) for k in ROLES}))
            report['timing']=b.timing(report['rows']);reports[(r,v)]=report
        return plan,run,reports

    def test_two_round_gate_no_automatic_promotion(self):
        r=b.summarize(*self.fixture());self.assertEqual(r['status'],'pipeline_performance_gate_passed_not_promoted')
        self.assertEqual((r['measured_pages'],r['warmup_pages']),(208,104));self.assertFalse(r['accepted_for_release'])

    def test_one_slow_round(self):
        f=self.fixture();r=f[2][('r2','candidate')]
        for z in r['rows']:z['wall_ms']=101.
        r['timing']=b.timing(r['rows']);self.assertEqual(b.summarize(*f)['status'],'pipeline_performance_gate_not_met')

    def test_path_regression_fails(self):
        f=self.fixture();r=f[2][('r1','candidate')]
        for z in r['rows']:z['wall_ms']=106. if z['image_type']==1 else 50.
        r['timing']=b.timing(r['rows']);result=b.summarize(*f)['rounds'][0]
        self.assertGreater(result['mean_gain_fraction'],.05);self.assertFalse(result['performance_gate_passed'])

    def test_strict_protocol(self):
        for kind in ('order','coverage','error','counter','output','input','business','structure','fallback','two_ner','env','pid','clock'):
            f=copy.deepcopy(self.fixture());r=f[2][('r2','candidate')];z=r['rows'][0]
            if kind=='order':f[1]['workers'].reverse()
            if kind=='coverage':r['warmup'].pop()
            if kind=='error':r['execution']['det']['errors']=1
            if kind=='counter':r['execution']['ner']['calls']-=1
            if kind=='output':z['ner_outputs']=['different']
            if kind=='input':z['ner_inputs']=[]
            if kind=='business':z['signature']['business_sha256']='different'
            if kind=='structure':z['signature']['structure_fields_sha256']='different'
            if kind=='fallback':z['counters']['rec']['cpu_fallback_calls']=1
            if kind=='two_ner':r['ner_loads'].append('reference')
            if kind=='env':r['startup_environment']={'CUDA_VISIBLE_DEVICES':'-1'}
            if kind=='pid':r['pid']=99
            if kind=='clock':z['ner_wall_ms']=1000.;r['timing']=b.timing(r['rows'])
            with self.subTest(kind=kind),self.assertRaises(ValueError):b.summarize(*f)

    def test_reverse_case_order(self):
        self.assertEqual(b.page_indices('r1'),list(range(26)))
        self.assertEqual(b.page_indices('r2'),list(range(25,-1,-1)))

    def test_memory_gate(self):
        f=self.fixture();r=f[2][('r1','candidate')];r['warm_nvmap_delta_mib']+=300;r['after_warmup']['memory']['NvMapMemUsed_kib']+=300*1024
        self.assertEqual(b.summarize(*f)['status'],'pipeline_performance_gate_not_met')

    def test_counter_errors_and_reset(self):
        before={'ner':dict(calls=2,errors=0)}
        for after in ({'ner':dict(calls=1,errors=0)},{'ner':dict(calls=3,errors=1)},{'rec':dict(calls=3,errors=0)}):
            with self.assertRaises(ValueError):b.counters(after,before)

    def test_signature_keeps_symbols_and_fields(self):
        data=dict(FullText='1-2',Text=[],MarkdownBodyLines=[],NER=[],special={'page_no':'1-2'},Box=[],RawLines=[{'text':'1-2'}])
        fn=lambda value,key:value.get(key,'')
        a=b.signature(data,fn);copy_data=copy.deepcopy(data);copy_data['FullText']='12'
        self.assertNotEqual(a,b.signature(copy_data,fn));copy_data=copy.deepcopy(data);copy_data['special']['page_no']='12'
        self.assertNotEqual(a,b.signature(copy_data,fn));self.assertEqual(a,b.signature(data,fn))

    def test_setup_loads_only_selected_ner_and_restores_constructor(self):
        for variant in ('reference','candidate'):
            calls=[]
            class Models:
                def native(self,role):calls.append(('original',role));return role
            saved=Models.native;server=SimpleNamespace(cfg=SimpleNamespace(NER_SCORE_THRESHOLD=.98))
            engine=dict(path='/isolated/ner.engine',sha256=variant,plugins=[dict(path='/isolated/plugin.so',sha256='plugin')])
            def session(path,sha,role,plugins):calls.append(('selected',sha));return SimpleNamespace(sha=sha,plugins=plugins)
            def load(s,root,cfg):
                m=Models();m.native('det');value=m.native('ner');self.assertEqual(value.sha,variant)
                return object(),m
            modules={'serve_jetson':SimpleNamespace(deployment=lambda:(server,'root',{}),load_models=load),
                'jetson_models':SimpleNamespace(JetsonModels=Models),'trt82_backend':SimpleNamespace(Session=session)}
            with patch.dict(sys.modules,modules):result=b.setup(SimpleNamespace(),variant,engine)
            self.assertIs(Models.native,saved);self.assertEqual(result[-1],[variant]);self.assertEqual(calls,[('original','det'),('selected',variant)])

    def test_setup_restores_constructor_on_failure(self):
        class Models:
            def native(self,role):return role
        saved=Models.native
        def fail(*args):raise ValueError('load failed')
        modules={'serve_jetson':SimpleNamespace(deployment=lambda:(None,None,None),load_models=fail),
            'jetson_models':SimpleNamespace(JetsonModels=Models),'trt82_backend':SimpleNamespace(Session=None)}
        with patch.dict(sys.modules,modules),self.assertRaises(ValueError):b.setup(SimpleNamespace(),'candidate',{})
        self.assertIs(Models.native,saved)


if __name__=='__main__':unittest.main()
