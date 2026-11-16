"""CPU/fake tests only; not a substitute for native Jetson engine regression."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jetson'))
from jetson_models import OCRSession, CheckedCPU, JetsonModels
from health_jetson import ready
from source_bundle import safe_name


class AdapterTests(unittest.TestCase):
    def test_native_route_never_loads_cpu(self):
        gpu = Mock(errors=0, role='rec')
        gpu.in_profile.return_value = True
        gpu.run.return_value = [np.ones((1,2),np.float32)]
        session = OCRSession(gpu,Path('rec.onnx'))
        with patch('jetson_models.cpu_session') as factory:
            out = session.run(None,{'x':np.ones((1,3,48,32),np.float32)})
        factory.assert_not_called()
        self.assertEqual(out[0].shape,(1,2))

    def test_outside_profile_preserves_original_tensor(self):
        gpu = Mock(errors=0,role='rec')
        gpu.in_profile.return_value = False
        target = Mock()
        target.run.return_value = [np.ones((1,2),np.float32)]
        session = OCRSession(gpu,Path('rec.onnx'))
        value = np.ones((1,3,48,2000),np.float32)
        feeds = {'x':value}
        with patch('jetson_models.cpu_session',return_value=target) as factory:
            session.run(None,feeds)
            session.run(None,feeds)
        factory.assert_called_once()
        self.assertIs(target.run.call_args.args[1]['x'],value)
        gpu.run.assert_not_called()
        self.assertEqual(session.cpu.calls,2)

    def test_gpu_error_does_not_fallback(self):
        gpu = Mock(errors=0,role='rec')
        gpu.in_profile.return_value = True
        gpu.run.side_effect = RuntimeError('execution failure')
        with patch('jetson_models.cpu_session') as factory:
            with self.assertRaises(RuntimeError):
                OCRSession(gpu,Path('rec.onnx')).run(None,{'x':np.ones((1,))})
        factory.assert_not_called()

    def test_latched_gpu_error_refuses_cpu_route(self):
        gpu = Mock(errors=1,role='rec')
        with patch('jetson_models.cpu_session') as factory:
            with self.assertRaises(RuntimeError):
                OCRSession(gpu,Path('rec.onnx')).run(None,{})
        factory.assert_not_called()

    def test_cpu_nonfinite_latches(self):
        target = Mock()
        target.run.return_value = [np.array([np.nan])]
        checked = CheckedCPU(target,'layout','legacy_parser')
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                checked.run(None,{})
        target.run.assert_called_once()
        self.assertEqual(checked.errors,1)

    def test_health_requires_all_warmups_and_no_fallback_error(self):
        rows = {r:{'backend':'tensorrt82_cuda' if r in ('det','rec','yolo','enhance','ner') else 'onnxruntime_cpu',
                   'errors':0,'calls':1} for r in ('det','rec','layout','cell','yolo','ner','enhance')}
        body = {'status':'ready','scope':'full_business','deployment':'jetson-r1',
                'models_execution':rows,'warmup_execution':copy.deepcopy(rows)}
        self.assertTrue(ready(body))
        body['warmup_execution']['ner']['calls'] = 0
        self.assertFalse(ready(body))
        body['warmup_execution']['ner']['calls'] = 1
        body['models_execution']['rec']['cpu_fallback'] = {'errors':1}
        self.assertFalse(ready(body))

    def test_archive_names(self):
        self.assertTrue(safe_name('pth/model.onnx'))
        for name in ('../license.dat','/etc/passwd','C:/data','a\\b','a/../b'):
            self.assertFalse(safe_name(name))

    def test_r2_health_requires_seven_gpu_models(self):
        rows={r:{'backend':'tensorrt82_cuda','errors':0,'calls':1}
              for r in ('det','rec','layout','cell','yolo','ner','enhance')}
        body={'status':'ready','scope':'full_business','deployment':'jetson-r2',
              'models_execution':rows,'warmup_execution':copy.deepcopy(rows)}
        self.assertTrue(ready(body))
        for role in ('layout','cell','ner'):
            rows[role]['backend']='onnxruntime_cpu'
            self.assertFalse(ready(body))
            rows[role]['backend']='tensorrt82_cuda'

    def test_native_layout_restores_business_output_order(self):
        instance=object.__new__(JetsonModels)
        instance.root=Path('/release')
        instance.config={'engines':{'layout':{'path':'model.engine','sha256':'engine_hash',
                            'plugins':[{'path':'grid.so','sha256':'plugin_hash'}]}}}
        native=Mock(outputs=[SimpleNamespace(name='fetch_name_2'),SimpleNamespace(name='fetch_name_0'),SimpleNamespace(name='fetch_name_1')])
        with patch('jetson_models.Session',return_value=native) as factory:
            self.assertIs(instance.native('layout'),native)
        self.assertEqual([x.name for x in native.outputs],['fetch_name_0','fetch_name_1','fetch_name_2'])
        self.assertEqual(factory.call_args.kwargs['plugins'],[(Path('/release/grid.so'),'plugin_hash')])

    def test_native_layout_rejects_missing_business_output(self):
        instance=object.__new__(JetsonModels);instance.root=Path('/release')
        instance.config={'engines':{'layout':{'path':'model.engine','sha256':'x'}}}
        native=Mock(outputs=[SimpleNamespace(name='fetch_name_0')])
        with patch('jetson_models.Session',return_value=native):
            with self.assertRaises(ValueError):instance.native('layout')
        native.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
