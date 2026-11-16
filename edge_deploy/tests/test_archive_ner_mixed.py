import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import archive_ner_mixed as a


class Tests(unittest.TestCase):
    @staticmethod
    def fixture():
        meta={k:'hash' for k in ('source_sha256','deployment_sha256','prior_archive_sha256','prior_audit_sha256','engine_sha256','dataset_plan_sha256','artifacts_sha256','script_sha256','helpers_sha256')}
        sample=dict(image_sha256='image',image_type=3,inputs={},output_sha256='hash')
        plan=dict(meta,status='prepared',rules=a.n.RULES,graph={'protected_nodes':[]},rows=[sample.copy() for _ in range(6)])
        build=dict(meta,status='built_not_validated',flags=1025,plan_sha256='hash',candidate_engine_sha256='hash',engine_bytes=100,build_s=1.)
        common=dict(meta,plan_sha256='hash',candidate_engine_sha256='hash')
        metric=dict(all_finite=True,active_token_labels_equal=True,allclose_diagnostic=True,max_abs_error=0.)
        check=dict(common,status='finite_checked_not_business_accepted',inspection={'precision_boundary_verified':False},
            rows=[dict(sample,repeat_exact=True,metrics=metric.copy()) for _ in range(6)],
            boundaries=[dict(length=l,metrics=metric.copy()) for l in (2,128,512)],
            candidate_execution={'calls':15,'errors':0},boundary_reference_execution={'calls':3,'errors':0})
        rows=[]
        for i in range(26):
            rows.append(dict(image_sha256=str(i//2),image_type=3 if i%2==0 else 1,order=['reference','candidate'] if i%2==0 else ['candidate','reference'],
                passed=True,same_ner_inputs=True,token_labels_equal=True,inputs={'reference':[{}],'candidate':[{}]},ner_output_sha256={'reference':['hash'],'candidate':['hash']},
                entities=dict(postprocess_calls_reference=1,postprocess_calls_candidate=1,raw_entity_spans_equal=True,cleaned_entity_spans_equal=True,threshold_membership_equal=True,max_aligned_entity_score_delta=.01),
                strict_equal={k:True for k in ('FullText','Text','MarkdownBodyLines','NER','special','raw_line_texts')},
                fields={'api_special_text_equal':{k:True for k in ('title','category_no','page_no')}},cpu_fallback_calls={'reference':{},'candidate':{}}))
        quality=dict(common,status='not_accepted',precision_boundary_verified=False,rows=rows,business_passed_count=26,
            candidate_execution={'calls':26,'errors':0},original_execution={'ner':{'calls':27,'errors':0}})
        probe=dict(common,status='synthetic_masks_finite_not_accuracy_validated',rows=[dict(length=l,segment=s,repeat_exact=True,metrics=metric.copy()) for l in (8,128,512) for s in (0,1)],
            reference_execution={'calls':6,'errors':0},candidate_execution={'calls':12,'errors':0})
        return [plan,build,check,quality,probe,{}]

    def run_summary(self,items):
        with patch.object(a.e,'inspect',return_value={'precision_boundary_verified':False}):return a.summary(*items)

    def test_business_equal_does_not_promote_opaque_candidate(self):
        r=self.run_summary(self.fixture())
        self.assertEqual(r['status'],'not_accepted');self.assertEqual(r['business_passed_count'],26)
        self.assertFalse(r['accepted_for_release']);self.assertEqual(r['benchmark_calls'],0)

    def test_reject_missing_case(self):
        f=self.fixture();f[3]['rows'].pop()
        with self.assertRaises(ValueError):self.run_summary(f)

    def test_reject_duplicate_case(self):
        f=self.fixture();f[3]['rows'][1]=copy.deepcopy(f[3]['rows'][0])
        with self.assertRaises(ValueError):self.run_summary(f)

    def test_reject_provenance_drift(self):
        f=self.fixture();f[4]['source_sha256']='changed'
        with self.assertRaises(ValueError):self.run_summary(f)

    def test_reject_false_quality_gate(self):
        f=self.fixture();f[3]['rows'][0]['entities']['threshold_membership_equal']=False
        with self.assertRaises(ValueError):self.run_summary(f)

    def test_reject_execution_error_and_missing_trace(self):
        for mutate in (lambda f:f[3]['candidate_execution'].update(errors=1),lambda f:f[3]['rows'][0]['entities'].update(postprocess_calls_candidate=0)):
            f=self.fixture();mutate(f)
            with self.assertRaises(ValueError):self.run_summary(f)

    def test_reject_missing_mask_case(self):
        f=self.fixture();f[4]['rows'].pop()
        with self.assertRaises(ValueError):self.run_summary(f)

    def test_synthetic_mask_contract(self):
        for length in (8,128,512):
            for segment in (0,1):
                feeds=a.masks.inputs(length,segment);a.n.input_meta(feeds)
                self.assertEqual(feeds['attention_mask'].sum(),4)
                self.assertTrue(np.all(feeds['input_ids'][0,4:]==0))
                self.assertEqual(feeds['token_type_ids'].sum(),segment*2)
        with self.assertRaises(ValueError):a.masks.inputs(2,0)


if __name__=='__main__':unittest.main()
