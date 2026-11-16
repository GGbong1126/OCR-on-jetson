import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import archive_ner_protected as a


class Tests(unittest.TestCase):
    @staticmethod
    def fixture():
        original=[dict(image_sha256=str(i//2),image_type=(3,1)[i%2],inputs={},output_sha256='x') for i in range(6)]
        plan=dict(status='prepared',rules=a.p.RULES,rows=original,boundaries=[dict(k,inputs=a.n.input_meta(f)) for k,f in a.p.boundaries()])
        build=dict(status='built_not_validated',flags=1024,boundaries=[dict(name=n) for n in a.p.plugin_names()],build_s=1,engine_bytes=100)
        metric=dict(all_finite=True,active_token_labels_equal=True,allclose_diagnostic=True,max_abs_error=0)
        real=[dict(r,metrics=metric,repeat_exact=True) for r in original]
        boundary=[dict(r,metrics=metric,repeat_exact=True,all_token_labels_equal=True) for r in plan['boundaries']]
        layers={'Layers':[]}
        check=dict(status='finite_checked_not_business_accepted',rules=a.p.RULES,rows=real,boundaries=boundary,
            candidate_execution={'calls':92,'errors':0},boundary_reference_execution={'calls':40,'errors':0},inspection=a.p.inspect(layers))
        return plan,build,check,None,layers

    def test_failed_visibility_cannot_be_promoted(self):
        r=a.summarize(*self.fixture());self.assertEqual(r['status'],'structure_or_labels_not_accepted')
        self.assertFalse(r['accepted_for_release']);self.assertEqual(r['business_cases'],0)

    def test_counters_coverage_and_flags_fail_closed(self):
        for kind in ('counter','missing','flags','repeat'):
            f=copy.deepcopy(self.fixture())
            if kind=='counter':f[2]['candidate_execution']['calls']-=1
            if kind=='missing':f[2]['boundaries'].pop()
            if kind=='flags':f[1]['flags']=1025
            if kind=='repeat':f[2]['rows'][0]['repeat_exact']=False
            with self.assertRaises(ValueError):a.summarize(*f)

    def test_missing_quality_after_passed_gate_rejected(self):
        f=list(self.fixture());f[4]={'Layers':[dict(Name=n,LayerType='PluginV2',Inputs=[{'Format/Datatype':'Float'}]*(3 if '_norm_' in n else 2),Outputs=[{'Format/Datatype':'Float'}]) for n in a.p.plugin_names()]}
        f[2]['inspection']=a.p.inspect(f[4])
        with self.assertRaises(ValueError):a.summarize(*f)

    def test_technical_archive_allowlist(self):
        self.assertEqual(len(a.FILES),len(set(a.FILES)))
        for name in a.FILES:
            self.assertEqual(Path(name).name,name);self.assertIn(Path(name).suffix,('.json','.py','.log'))
        self.assertNotIn('ner_protected.engine',a.FILES)


if __name__=='__main__':unittest.main()
