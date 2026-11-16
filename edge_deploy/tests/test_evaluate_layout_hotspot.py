from pathlib import Path
import sys
import unittest
from types import SimpleNamespace as NS
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import evaluate_layout_hotspot as m


def fmt(kind): return [dict(Name='t', **{'Format/Datatype':kind})]


def inspector():
    rows=[dict(Name=n,ParameterType='Convolution',LayerType='Convolution',Inputs=fmt('Half'),Outputs=fmt('Half')) for n in m.build.TARGETS]
    rows += [dict(Name='GridSample.'+str(i),LayerType='PluginV2',Inputs=fmt('Float'),Outputs=fmt('Float')) for i in range(18)]
    rows += [dict(Name='restore',LayerType='Reformat',Inputs=fmt('Half'),Outputs=fmt('Float'))]
    return dict(Layers=rows)


class Tests(unittest.TestCase):
    def test_native_all_output_api_reordering(self):
        class Native:
            def run(self,names,feeds):
                if names is not None: raise ValueError('Only all outputs')
                return [np.array([2]),np.array([1])]
            def get_outputs(self): return [NS(name='b'),NS(name='a')]
        values=m.reordered_outputs(Native(),{},['a','b'])
        self.assertEqual([int(v[0]) for v in values],[1,2])
        with self.assertRaises(ValueError):m.reordered_outputs(Native(),{},['a'])
    def test_exact_target_names(self):
        self.assertTrue(m.token('fused Conv.91 + other','Conv.91'))
        self.assertFalse(m.token('Conv.910','Conv.91'))
    def test_inspector_three_hotspots_and_18_plugins(self):
        r=m.inspect(inspector()); self.assertEqual(r['fp32_grid_sample_count'],18)
        self.assertEqual(len(r['half_format_layers']),4)
    def test_additional_half_or_bad_plugin_rejected(self):
        for kind in ('half','missing','int8','target'):
            data=inspector()
            if kind=='half': data['Layers'].append(dict(Name='Softmax',LayerType='Softmax',Inputs=fmt('Half'),Outputs=fmt('Half')))
            if kind=='missing': data['Layers'].pop(4)
            if kind=='int8': data['Layers'][3]['Inputs']=fmt('Int8')
            if kind=='target': data['Layers'][0]['Outputs']=fmt('Float')
            with self.assertRaises(ValueError):m.inspect(data)
    def test_numeric_changes_are_diagnostic(self):
        a={'a':np.array([1],np.float32)};b={'a':np.array([1.5],np.float32)}
        r=m.compare(a,b)['a'];self.assertFalse(r['allclose_diagnostic']);self.assertTrue(r['all_finite'])
    def test_bad_numeric_contract_rejected(self):
        ref={'a':np.array([1],np.float32)}
        for out in ({'b':np.array([1],np.float32)}, {'a':np.array([np.nan],np.float32)},
                    {'a':np.array([1],np.float16)}, {'a':np.array([1,2],np.float32)}):
            with self.assertRaises(ValueError):m.compare(ref,out)
    def test_strict_business_and_fallback_gate(self):
        row=dict(strict_equal={k:True for k in ('FullText','Text','MarkdownBodyLines','NER','raw_line_texts')},
            fields=dict(api_special_text_equal={k:True for k in ('title','category_no','page_no')}),
            number_symbol_sequence_equal=True,same_original_input=True,
            cpu_fallback_calls=dict(reference={'rec':0},candidate={'rec':0}))
        self.assertTrue(m.business_passed(row))
        row['cpu_fallback_calls']['candidate']['rec']=1
        self.assertFalse(m.business_passed(row))
        row['cpu_fallback_calls']['candidate']['rec']=0;row['strict_equal']['Text']=False
        self.assertFalse(m.business_passed(row))
    def test_digits_symbols_not_normalized(self):
        self.assertTrue(m.protected_equal('a 1-2','b 1-2'))
        self.assertFalse(m.protected_equal('1-2','1−2'))
        self.assertFalse(m.protected_equal('１２','12'))


if __name__=='__main__':unittest.main()
