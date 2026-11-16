import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import archive_cell_partition as a


def execution(counts):
    return dict(errors=0,calls=counts,parts={k:dict(calls=n,errors=0) for k,n in
        dict(prefix=sum(counts.values()),suffix=sum(counts.values()),**counts).items()})


def geometry():
    return dict(count_output_exact=True,reference_effective_count=1,candidate_effective_count=1,
        ordered_classes_equal=True,min_ordered_iou=1.,max_coordinate_delta_pixels=0.,max_score_delta_diagnostic=0.,
        geometry_passed=True,repeat_exact=True,raw={'fetch_name_0':{'allclose':True},'fetch_name_1':{'allclose':True}})


def fixture():
    meta={k:'hash' for k in a.META}
    plan=dict(meta,status='partition_prepared',rules=a.c.RULES,image_cases=[dict(image_sha256=str(i//2),image_type=(3,1)[i%2]) for i in range(26)])
    layers={k:{} for k in ('prefix','suffix','fp32','mixed')}
    builds={k:dict(meta,status='built_not_validated',flags=0) for k in ('prefix','suffix')}
    check=dict(meta,status='full_cell_formats_checked_not_business_accepted',rules=a.c.RULES,inspection=layers,
        execution=execution({'fp32':12,'mixed':12}),control_geometry_passed=True,mixed_geometry_passed=True,
        rows=[dict(variants={k:geometry() for k in ('fp32','mixed')}) for _ in range(6)])
    compare=dict(strict_equal={k:True for k in ('FullText','Text','MarkdownBodyLines','NER','raw_line_texts')},
        fields=dict(api_special_text_equal={k:True for k in ('title','category_no','page_no')},box_class_counts_reference={},box_class_counts_candidate={}),
        box_count_reference=1,box_count_candidate=1,number_symbol_sequence_equal=True,same_original_input=True,
        cpu_fallback_calls={'reference':{},'candidate':{}},cell_comparisons=[geometry()],passed=True)
    keys=('reference_vs_fp32','reference_vs_mixed','fp32_vs_mixed')
    rows=[dict(case,order=list(a.e.BRANCHES[i%3:]+a.e.BRANCHES[:i%3]),original_cell_calls=1,
        inputs={v:['sha'] for v in a.e.BRANCHES},output_sha256={v:['outsha'] for v in a.e.BRANCHES},
        comparisons={k:copy.deepcopy(compare) for k in keys}) for i,case in enumerate(plan['image_cases'])]
    quality=dict(meta,status='quality_checked_not_promoted',rules=a.c.RULES,rows=rows,totals={k:26 for k in keys},
        candidate_execution=execution({'fp32':26,'mixed':26}),original_execution={'cell':{'calls':27,'errors':0}})
    return plan,builds,check,quality,layers


class Tests(unittest.TestCase):
    def summarize(self,args):
        with patch.object(a.p,'inspect',return_value={}),patch.object(a.c,'inspect_fp32',return_value={}):
            return a.summarize(*args)

    def test_complete_quality_never_promotes(self):
        r=self.summarize(fixture())
        self.assertEqual(r['status'],'quality_passed_not_benchmarked')
        self.assertEqual(r['benchmark_calls'],0);self.assertFalse(r['accepted_for_release'])

    def test_failure_not_hidden(self):
        data=fixture();quality=data[3];r=quality['rows'][0]['comparisons']['reference_vs_mixed']
        r['strict_equal']['Text']=False;r['passed']=False;quality['totals']['reference_vs_mixed']=25
        result=self.summarize(data)
        self.assertEqual(result['status'],'quality_not_accepted')
        self.assertEqual(result['full_business']['reference_vs_mixed']['failed_case_indices'],[0])

    def test_missing_or_reordered_cases_rejected(self):
        for mode in ('missing','reordered'):
            data=fixture();rows=data[3]['rows']
            if mode=='missing':rows.pop()
            else:rows.reverse()
            with self.assertRaises(ValueError):self.summarize(data)

    def test_errors_and_call_counts_rejected(self):
        for key in ('errors','calls'):
            data=fixture();data[3]['candidate_execution']['parts']['suffix'][key]=1
            with self.assertRaises(ValueError):self.summarize(data)

    def test_forged_totals_or_input_rejected(self):
        data=fixture();data[3]['totals']['reference_vs_fp32']=25
        with self.assertRaises(ValueError):self.summarize(data)
        data=fixture();data[3]['rows'][0]['inputs']['mixed']=['other']
        with self.assertRaises(ValueError):self.summarize(data)

    def test_provenance_drift_rejected(self):
        data=fixture();data[3]['script_sha256']='changed'
        with self.assertRaises(ValueError):self.summarize(data)


if __name__=='__main__':unittest.main()
