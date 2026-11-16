import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import evaluate_cell_partition as e


class Tests(unittest.TestCase):
    @staticmethod
    def row():
        return dict(strict_equal={k:True for k in ('FullText','Text','MarkdownBodyLines','NER','raw_line_texts')},
            fields=dict(api_special_text_equal={k:True for k in ('title','category_no','page_no')},
                box_class_counts_reference={'table':1},box_class_counts_candidate={'table':1}),
            box_count_reference=1,box_count_candidate=1,number_symbol_sequence_equal=True,same_original_input=True,
            cpu_fallback_calls={'reference':{'det':0,'rec':0},'candidate':{'det':0,'rec':0}},cell_comparisons=[])

    def test_business_strict(self):
        r=self.row();self.assertTrue(e.row_passed(r))
        for key in r['strict_equal']:
            b=copy.deepcopy(r);b['strict_equal'][key]=False
            self.assertFalse(e.row_passed(b))

    def test_fields_counts_and_input(self):
        r=self.row()
        for key in r['fields']['api_special_text_equal']:
            b=copy.deepcopy(r);b['fields']['api_special_text_equal'][key]=False
            self.assertFalse(e.row_passed(b))
        for key,value in (('box_count_candidate',2),('same_original_input',False),('number_symbol_sequence_equal',False)):
            self.assertFalse(e.row_passed(dict(r,**{key:value})))

    def test_no_extra_fallback(self):
        r=self.row();r['cpu_fallback_calls']['candidate']['rec']=1
        self.assertFalse(e.row_passed(r))

    def test_cell_geometry_required(self):
        r=self.row();r['cell_comparisons']=[dict(count_output_exact=False)]
        self.assertFalse(e.row_passed(r))

    def test_protected_numbers_symbols(self):
        self.assertTrue(e.protected_equal('甲 1—2','乙1—2'))
        self.assertFalse(e.protected_equal('甲1—2','甲1-2'))
        self.assertFalse(e.protected_equal('甲1—2','甲12'))


if __name__=='__main__':unittest.main()
