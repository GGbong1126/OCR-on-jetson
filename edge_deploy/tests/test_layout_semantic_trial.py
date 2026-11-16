import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import layout_semantic_trial as m


def row():
    return dict(effective_match=dict(reference_count=2,candidate_count=2,unique_matched_count=2,
        unmatched_reference_count=0,unmatched_candidate_count=0,order_preserved=True,
        matched_min_iou=.999,matched_max_coordinate_delta=.1,matched_max_score_delta=.0001),
        count_output_exact=True,above_threshold_counts=[3,3],outputs_finite=True,
        strict_equal=dict.fromkeys(('FullText','Text','MarkdownBodyLines','NER','raw_line_texts'),True),
        fields=dict(api_special_text_equal=dict.fromkeys(('title','category_no','page_no'),True)),
        number_symbol_sequence_equal=True,same_original_input=True,
        cpu_fallback_calls={v:dict(det=0,rec=1) for v in ('reference','candidate')})


class Tests(unittest.TestCase):
    def test_declare_semantic_policy_not_old_allclose(self):
        self.assertFalse(m.RULES['raw_allclose_acceptance']);self.assertTrue(m.row_passed(row()))
        self.assertEqual(m.RULES['mean_improvement_min_percent'],5.)
    def test_geometry_and_structure_failures(self):
        for k,v in dict(candidate_count=3,unique_matched_count=1,unmatched_reference_count=1,
                        unmatched_candidate_count=1,order_preserved=False,matched_min_iou=.98,
                        matched_max_coordinate_delta=.501,matched_max_score_delta=.0011).items():
            r=row();r['effective_match'][k]=v
            with self.subTest(k=k):self.assertFalse(m.geometry_passed(r))
    def test_nonfinite_or_missing_geometry_rejected(self):
        for v in (float('nan'),float('inf'),None):
            r=row();r['effective_match']['matched_max_coordinate_delta']=v
            self.assertFalse(m.geometry_passed(r))
    def test_count_and_threshold_gate(self):
        r=row();r['count_output_exact']=False;self.assertFalse(m.row_passed(r))
        r=row();r['above_threshold_counts'][1]=4;self.assertFalse(m.row_passed(r))
    def test_business_still_strict_and_no_extra_fallback(self):
        for k in ('FullText','NER','Text'):
            r=row();r['strict_equal'][k]=False;self.assertFalse(m.row_passed(r))
        r=row();r['cpu_fallback_calls']['candidate']['rec']=2;self.assertFalse(m.row_passed(r))
        r=row();r['outputs_finite']=False;self.assertFalse(m.row_passed(r))
    def test_empty_is_only_consistency(self):
        r=row();v=r['effective_match']
        v.update(reference_count=0,candidate_count=0,unique_matched_count=0,
                 matched_min_iou=None,matched_max_coordinate_delta=None,matched_max_score_delta=None)
        self.assertTrue(m.geometry_passed(r));r['strict_equal']['FullText']=False
        self.assertFalse(m.row_passed(r))


if __name__=='__main__':unittest.main()
