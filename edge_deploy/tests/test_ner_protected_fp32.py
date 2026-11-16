import copy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
import ner_protected_fp32 as p


class Tests(unittest.TestCase):
    @staticmethod
    def info():
        return {'Layers':[dict(Name=n,LayerType='PluginV2',Inputs=[{'Format/Datatype':'Float'}]*(3 if '_norm_' in n else 2),Outputs=[{'Format/Datatype':'Float'}]) for n in p.plugin_names()]}

    def test_all_boundaries_visible(self):
        i=p.inspect(self.info());self.assertTrue(i['precision_boundary_verified']);self.assertEqual(i['visible_protected_plugins'],37)

    def test_missing_duplicate_half_and_fused_rejected(self):
        for kind in ('missing','duplicate','half','opaque'):
            i=self.info()
            if kind=='missing':i['Layers'].pop()
            if kind=='duplicate':i['Layers'].append(copy.deepcopy(i['Layers'][0]))
            if kind=='half':i['Layers'][0]['Inputs'][0]={'Format/Datatype':'Half'}
            if kind=='opaque':i['Layers'][0]['LayerType']='Myelin'
            self.assertFalse(p.inspect(i)['precision_boundary_verified'],kind)

    def test_nonprotected_half_rejected(self):
        i=self.info();i['Layers'].append(dict(Name='other',Inputs=[{'Format/Datatype':'Half'}],Outputs=[]))
        self.assertFalse(p.inspect(i)['precision_boundary_verified'])

    def test_boundary_contract_and_coverage(self):
        rows=list(p.boundaries());self.assertEqual(len(rows),40)
        self.assertEqual(len({tuple(k.items()) for k,f in rows}),40)
        for k,f in rows:
            p.n.input_meta(f)
            if k['kind']=='none':self.assertEqual(int(f['attention_mask'].sum()),0)
            if k['kind']=='one':self.assertEqual(int(f['attention_mask'].sum()),1)
        self.assertEqual(sorted({k['length'] for k,f in rows}),[2,127,128,512])

    def test_replace_rewires_all_uses_without_public_output_change(self):
        old=SimpleNamespace(name='original');new=SimpleNamespace(name='candidate')
        class Layer:
            name='consumer';num_inputs=2
            def __init__(self):self.values=[old,old]
            def get_input(self,j):return self.values[j]
            def set_input(self,j,t):self.values[j]=t
        v=Layer();net=SimpleNamespace(num_outputs=0)
        users=p.replace_output(net,old,new,[v]);self.assertEqual(len(users),2)
        self.assertEqual(new.name,'original');self.assertEqual(old.name,'original__unused_e119')
        self.assertTrue(all(t is new for t in v.values))
        with self.assertRaises(ValueError):p.replace_output(net,old,new,[])

    def test_public_output_replacement_rejected(self):
        t=SimpleNamespace(name='output');net=SimpleNamespace(num_outputs=1,get_output=lambda i:t)
        with self.assertRaises(ValueError):p.replace_output(net,t,SimpleNamespace(name='new'),[])

    def test_full_fp32_control_not_speed_or_release(self):
        self.assertTrue(p.RULES['fp32_only']);self.assertEqual(p.RULES['benchmark_calls'],0);self.assertFalse(p.RULES['accepted_for_release'])
        self.assertEqual(len(p.PLUGINS),2)
        for path,sha in p.PLUGINS:self.assertRegex(sha,r'^[0-9a-f]{64}$')


if __name__=='__main__':unittest.main()
