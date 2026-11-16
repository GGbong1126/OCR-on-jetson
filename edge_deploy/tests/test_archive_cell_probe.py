import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import archive_cell_probe as a
import test_probe_cell_hotspot as fixtures


class SummaryTests(unittest.TestCase):
    def fixture(self):
        meta={k:k for k in ('source_sha256','engine_sha256','deployment_sha256','plugin_sha256','dataset_plan_sha256','script_sha256','helpers_sha256')}
        pages=[dict(image_sha256=str(i),image_type=k,cell_calls=1) for i in range(3) for k in (3,1)]
        ranges=dict(meta,status='original_cell_ranges_measured',rules=copy.deepcopy(a.p.RULES),
            pages=pages,consumer=dict(both_model_outputs_consumed=True),
            graph=dict(graph_nodes=1195,live_nodes=1195,parameters=[dict(parameters={'w':dict(abs_max=1.,abs_gt_fp16_max=0)})]),rows=[])
        check=dict(meta,status='isolated_boundary_compared_not_full_model_accepted',rules=copy.deepcopy(a.p.RULES),rows=[],
            control_allclose_cases=6,mixed_allclose_cases=0,execution={},inspection={})
        for i,page in enumerate(pages):
            r=dict(image_sha256=page['image_sha256'],image_type=page['image_type'],call_index=i,
                   inputs={'x':'hash'},original_output_sha256={'y':'hash'},feature_sha256={'f':'hash'})
            ranges['rows'].append(dict(r,ranges={'x':dict(abs_max=2.,abs_gt_fp16_max=0)}))
            metric={n:dict(mae=.01,max_abs_error=.1,allclose_diagnostic=False) for n in a.p.OUTPUTS}
            control={n:dict(allclose_diagnostic=True) for n in a.p.OUTPUTS}
            check['rows'].append(dict(r,repeat_exact=True,variants=dict(fp32=control,mixed=metric),mixed_vs_fp32=metric))
        builds={v:dict(meta,status='built_not_validated',variant=v,rules=copy.deepcopy(a.p.RULES)) for v in ('fp32','mixed')}
        inspectors={v:fixtures.Tests.info(v) for v in builds}
        for v in builds:
            check['execution'][v]=dict(calls=12,errors=0)
            check['inspection'][v]=a.p.inspect(inspectors[v],v)
        return ranges,builds,check,inspectors

    def test_summary_not_model_acceptance(self):
        out=a.summarize(*self.fixture())
        self.assertEqual(out['real_cell_calls'],6)
        self.assertEqual(out['control_allclose_cases'],6)
        self.assertEqual(out['mixed_allclose_cases'],0)
        self.assertFalse(out['accepted_for_release'])
        self.assertEqual(out['benchmark_calls'],0)

    def test_reject_drift_counts_and_errors(self):
        for mutate in (lambda c:c['rows'][0].update(call_index=4),
                       lambda c:c.update(mixed_allclose_cases=6),
                       lambda c:c['execution']['mixed'].update(errors=1),
                       lambda c:c['execution']['mixed'].update(calls=11),
                       lambda c:c['rows'][0].update(repeat_exact=False)):
            r,b,c,i=self.fixture();mutate(c)
            with self.assertRaises(ValueError):a.summarize(r,b,c,i)

    def test_reject_missing_coverage_and_changed_policy(self):
        r,b,c,i=self.fixture();r['pages'].pop()
        with self.assertRaises(ValueError):a.summarize(r,b,c,i)
        r,b,c,i=self.fixture();b['mixed']['rules']['benchmark_authorized']=True
        with self.assertRaises(ValueError):a.summarize(r,b,c,i)

    def test_reject_invalid_metric_or_format(self):
        r,b,c,i=self.fixture();c['rows'][0]['mixed_vs_fp32'][a.p.OUTPUTS[0]]['mae']=float('nan')
        with self.assertRaises(ValueError):a.summarize(r,b,c,i)
        r,b,c,i=self.fixture();i['mixed']['Layers'][0]['Inputs'][0]['Format/Datatype']='Float'
        with self.assertRaises(ValueError):a.summarize(r,b,c,i)


if __name__=='__main__':unittest.main()
