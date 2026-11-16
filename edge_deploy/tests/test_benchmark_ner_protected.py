import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import benchmark_ner_protected as b


class Tests(unittest.TestCase):
    @staticmethod
    def fixture():
        inputs=[dict(image_sha256=str(i//2),image_type=(3,1)[i%2],inputs={'id':i}) for i in range(6)]
        plan=dict(status='prepared',rules=copy.deepcopy(b.RULES),inputs=inputs,
            expected_hashes={v:[str(i) for i in range(6)] for v in ('reference','candidate')},
            engines={v:{'path':v} for v in ('reference','candidate')})
        run=dict(status='measured_not_summarized',rules=copy.deepcopy(b.RULES),parent_pid=99,workers=[],
            inputs=[r['inputs'] for r in inputs],capture_execution={'ner':dict(calls=7,errors=0)},
            environment_before_capture={k:None for k in b.ENV_KEYS},environment_for_workers={k:None for k in b.ENV_KEYS},
            capture_pages=[dict(image_sha256=r['image_sha256'],image_type=r['image_type'],ner_calls=1) for r in inputs])
        reports={}
        for j,(r,v) in enumerate(b.ORDER):
            run['workers'].append(dict(round=r,variant=v,pid=j+100,exitcode=0))
            rows=[dict(cycle=c,index=i,wall_ms=100. if v=='reference' else 90.) for c in range(10) for i in range(6)]
            reports[(r,v)]=dict(status='measured',rules=copy.deepcopy(b.RULES),round=r,variant=v,pid=j+100,parent_pid=99,
                spawn_method='spawn',inputs=run['inputs'],engine=plan['engines'][v],expected_output_sha256=plan['expected_hashes'][v],
                all_outputs_exact=True,runtime_environment={k:None for k in b.ENV_KEYS},rows=rows,timing=b.stats(rows),execution=dict(calls=84,errors=0),
                phase_calls=dict(warmup=18,timed=60,postcheck=6),warm_nvmap_delta_mib=100.,context_device_bytes=10,
                before_load={'memory':{'NvMapMemUsed_kib':1024}},after_warmup={'memory':{'NvMapMemUsed_kib':103424}})
        return plan,run,reports

    def test_good_performance_is_not_promotion(self):
        r=b.summarize(*self.fixture())
        self.assertEqual(r['status'],'performance_gate_passed_not_promoted')
        self.assertEqual(r['timed_calls'],240);self.assertFalse(r['accepted_for_release'])

    def test_one_slow_round_fails_gate(self):
        f=self.fixture();report=f[2][('r2','candidate')]
        for r in report['rows']:r['wall_ms']=110.
        report['timing']=b.stats(report['rows'])
        self.assertEqual(b.summarize(*f)['status'],'performance_gate_not_met')

    def test_p95_regression_fails_despite_mean_gain(self):
        f=self.fixture();report=f[2][('r1','candidate')]
        for i,r in enumerate(report['rows']):r['wall_ms']=200. if i<4 else 50.
        report['timing']=b.stats(report['rows']);r=b.summarize(*f)['rounds'][0]
        self.assertGreater(r['mean_gain_fraction'],.05);self.assertFalse(r['performance_gate_passed'])

    def test_memory_limit(self):
        f=self.fixture();r=f[2][('r1','candidate')];r['after_warmup']['memory']['NvMapMemUsed_kib']+=300*1024;r['warm_nvmap_delta_mib']+=300
        self.assertEqual(b.summarize(*f)['status'],'performance_gate_not_met')

    def test_unstable_global_baseline(self):
        f=self.fixture();r=f[2][('r2','candidate')]
        for k in ('before_load','after_warmup'):r[k]['memory']['NvMapMemUsed_kib']+=33*1024
        result=b.summarize(*f);self.assertFalse(result['global_memory_comparison_stable']);self.assertEqual(result['status'],'performance_gate_not_met')

    def test_protocol_and_counter_drift_fail_closed(self):
        for kind in ('order','pid','parent','calls','errors','phase','input','hash','engine','rows','stats','method','extra','capture','rules','environment'):
            f=copy.deepcopy(self.fixture());r=f[2][('r1','candidate')]
            if kind=='order':f[1]['workers'].reverse()
            if kind=='pid':f[1]['workers'][1]['pid']=100
            if kind=='parent':r['parent_pid']=98
            if kind=='calls':r['execution']['calls']=83
            if kind=='errors':r['execution']['errors']=1
            if kind=='phase':r['phase_calls']['warmup']=0
            if kind=='input':r['inputs']=[]
            if kind=='hash':r['all_outputs_exact']=False
            if kind=='engine':r['engine']={}
            if kind=='rows':r['rows'].pop()
            if kind=='stats':r['timing']['overall']['mean_ms']=0
            if kind=='method':r['spawn_method']='fork'
            if kind=='extra':f[2][('r3','candidate')]=r
            if kind=='capture':f[1]['capture_execution']['ner']['errors']=1
            if kind=='rules':r['rules']['minimum_mean_gain']=0
            if kind=='environment':r['runtime_environment']['CUDA_VISIBLE_DEVICES']='-1'
            with self.subTest(kind=kind),self.assertRaises(ValueError):b.summarize(*f)

    def test_invalid_latency(self):
        for value in (float('nan'),float('inf'),-1):
            with self.assertRaises(ValueError):b.stats([dict(index=i,wall_ms=value) for i in range(6)])

    def test_missing_or_negative_memory(self):
        r=self.fixture()[2][('r1','candidate')]
        r['after_warmup']['memory']['NvMapMemUsed_kib']=0
        with self.assertRaises(ValueError):b.warm_nvmap_mib(r)
        r['after_warmup']['memory']['NvMapMemUsed_kib']=None
        with self.assertRaises(ValueError):b.warm_nvmap_mib(r)

    def test_restore_only_known_keys_and_preserve_unset(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0','LD_LIBRARY_PATH':'original','UNRELATED':'keep'},clear=True):
            before=b.runtime_environment();os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['LD_LIBRARY_PATH']='changed';os.environ['LD_PRELOAD']='added'
            b.restore_environment(before)
            self.assertEqual(b.runtime_environment(),before);self.assertNotIn('LD_PRELOAD',os.environ)
            self.assertEqual(os.environ['UNRELATED'],'keep')
            with self.assertRaises(ValueError):b.restore_environment({'UNRELATED':'changed'})


if __name__=='__main__':unittest.main()
