import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import evaluate_ner_mixed as e


class Tests(unittest.TestCase):
    def test_logits_mask_and_nonfinite(self):
        x=np.zeros((1,2,43),np.float32);x[:,:,0]=1;y=x.copy();y[0,1,1]=2
        feeds={'input_ids':np.ones((1,2),np.int64),'attention_mask':np.array([[1,0]])}
        self.assertTrue(e.compare(x,y,feeds)['active_token_labels_equal'])
        feeds['attention_mask'][0,1]=1
        self.assertFalse(e.compare(x,y,feeds)['active_token_labels_equal'])
        y[0,0,0]=np.inf
        with self.assertRaises(ValueError):e.compare(x,y,feeds)

    def test_entity_scores_not_semantic_equality(self):
        a=[dict(raw_signature=[('LABEL','PRIVATE',0,7)],clean_signature=[('LABEL','PRIVATE',0,7)],threshold_signature=[('LABEL','PRIVATE',0,7)],scores=[.99])]
        b=copy.deepcopy(a);b[0]['scores']=[.991]
        r=e.trace_compare(a,b)
        self.assertTrue(r['raw_entity_spans_equal']);self.assertAlmostEqual(r['max_aligned_entity_score_delta'],.001)
        b[0]['threshold_signature']=[]
        self.assertFalse(e.trace_compare(a,b)['threshold_membership_equal'])

    def test_entity_spans_and_missing_trace(self):
        a=[dict(raw_signature=[('L','x',0,1)],clean_signature=[('L','x',0,1)],threshold_signature=[],scores=[.5])]
        b=copy.deepcopy(a);b[0]['clean_signature']=[('L','x',1,2)]
        self.assertFalse(e.trace_compare(a,b)['cleaned_entity_spans_equal'])
        self.assertFalse(e.trace_compare(a,[])['raw_entity_spans_equal'])

    def test_opaque_fusion_cannot_prove_precision(self):
        f=lambda t:{'Format/Datatype':t}
        info={'Layers':[dict(Name='target',Inputs=[f('Half')],Outputs=[f('Half')]),dict(Name='norm',Inputs=[f('Float')],Outputs=[f('Float')])]}
        with patch.object(e.n,'TARGETS',('target',)):
            self.assertTrue(e.inspect(info,['norm'])['precision_boundary_verified'])
            info['Layers'].append(dict(Name='ForeignNode[other]',Inputs=[f('Float')],Outputs=[f('Float')]))
            self.assertFalse(e.inspect(info,['norm'])['precision_boundary_verified'])
            self.assertFalse(e.inspect(info,['missing'])['precision_boundary_verified'])

    @staticmethod
    def row():
        return dict(strict_equal={k:True for k in ('FullText','Text','MarkdownBodyLines','NER','raw_line_texts')},
            same_ner_inputs=True,token_labels_equal=True,
            entities={k:True for k in ('raw_entity_spans_equal','cleaned_entity_spans_equal','threshold_membership_equal')},
            fields={'api_special_text_equal':{'title':True,'category_no':True,'page_no':True}},
            cpu_fallback_calls={'reference':{},'candidate':{}})

    def test_quality_requires_all_entities_and_fields(self):
        r=self.row();self.assertTrue(e.quality_passed(r))
        for k in r['entities']:
            b=copy.deepcopy(r);b['entities'][k]=False;self.assertFalse(e.quality_passed(b))
        r['strict_equal']['NER']=False;self.assertFalse(e.quality_passed(r))

    def test_no_error_fallback_or_input_drift(self):
        r=self.row();r['cpu_fallback_calls']['candidate']['rec']=1
        self.assertFalse(e.quality_passed(r))
        r=self.row();r['same_ner_inputs']=False;self.assertFalse(e.quality_passed(r))


if __name__=='__main__':unittest.main()
