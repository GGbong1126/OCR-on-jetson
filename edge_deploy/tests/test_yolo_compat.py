"""Custom checkpoint layer registration and frozen forward behavior."""
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout.yolo_compat import ensure_yolo_custom_layers


class YoloCompatTests(unittest.TestCase):
    def test_existing_custom_module_preserved(self):
        original=object()
        with patch('layout.yolo_compat.importlib.import_module',return_value=original) as load:
            self.assertIs(ensure_yolo_custom_layers(),original)
        load.assert_called_once_with('ultralytics.nn.bifpn')

    def test_missing_module_registered_for_pickle(self):
        parent=SimpleNamespace(); vendor=SimpleNamespace(Concat_BiFPN=object())
        def load(name):
            if name=='ultralytics.nn.bifpn':
                raise ModuleNotFoundError(name=name)
            return parent if name=='ultralytics.nn' else vendor
        with patch('layout.yolo_compat.importlib.import_module',side_effect=load), patch.dict(sys.modules):
            self.assertIs(ensure_yolo_custom_layers(),vendor)
            self.assertIs(sys.modules['ultralytics.nn.bifpn'],vendor)
            self.assertIs(parent.bifpn,vendor)

    def test_missing_dependency_not_masked(self):
        with patch('layout.yolo_compat.importlib.import_module',side_effect=ModuleNotFoundError(name='torch')):
            with self.assertRaises(ModuleNotFoundError): ensure_yolo_custom_layers()

    def test_forward_preserves_three_weights_two_inputs_and_dimension(self):
        # Tiny NumPy-backed torch facade: original 147 tests do not require torch installed.
        class Module:
            pass
        nn=SimpleNamespace(Module=Module,Parameter=lambda value,requires_grad: value)
        torch=SimpleNamespace(nn=nn,float32=np.float32,ones=lambda n,dtype:np.ones(n,dtype=dtype),
                              sum=lambda x,dim:np.sum(x,axis=dim),cat=lambda xs,dim:np.concatenate(xs,axis=dim))
        name='layout.yolo_layers.bifpn'
        with patch.dict(sys.modules,{'torch':torch,'torch.nn':nn}):
            spec=importlib.util.spec_from_file_location('_test_bifpn',Path(__file__).resolve().parents[2]/'layout/yolo_layers/bifpn.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        layer=module.Concat_BiFPN(dimension=1)
        layer.w=np.array([2,3,5],np.float32)
        a=np.ones((1,2,3,3),np.float32);b=np.full_like(a,2)
        expected=np.concatenate([2*a/(10+0.0001),3*b/(10+0.0001)],axis=1)
        np.testing.assert_allclose(layer.forward([a,b]),expected,rtol=1e-6)
        self.assertEqual(layer.w.shape,(3,))


if __name__=='__main__':
    unittest.main()
