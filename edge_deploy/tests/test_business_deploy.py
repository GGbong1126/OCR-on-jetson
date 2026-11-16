"""Deployment control flow, no real server import, activation or model loading."""
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'edge_deploy/business'))
sys.path.insert(0, str(ROOT/'edge_deploy/tools'))
import serve_business as serve
from preflight_business import disk_probe


class BusinessDeployTests(unittest.TestCase):
    def server(self, modelpath):
        pipeline = SimpleNamespace(_use_onnx_layout=True, _use_onnx_cell=True, _use_onnx_ocr=True,
                                   _yolo_model=object(), _ner_pipeline=object(), _imgpre_model=object(),
                                   load_models=Mock())
        factory = Mock(return_value=pipeline)
        factory.FORCE_PADDLE = False
        app = SimpleNamespace(config={}, get=lambda path: lambda fn: fn)
        cfg = SimpleNamespace(**{name: str(modelpath) for name in (
            'ONNX_LAYOUT_MODEL','ONNX_CELL_MODEL','ONNX_OCR_DET_MODEL','ONNX_OCR_REC_MODEL',
            'LAYOUT_MODEL_DIR','TABLE_CELL_MODEL_DIR','OCR_MODEL_DIR','NER_MODEL_DIR','IMGPRE_MODEL_DIR')}, YOLO_CONF=0.3)
        return SimpleNamespace(validate_license=Mock(return_value=True), OCRPipeline=factory, cfg=cfg,
                               cv2=Mock(), torch=Mock(), app_online=app, initialize_counter_file=Mock())

    def test_invalid_license_blocks_before_models(self):
        server = self.server('unused')
        server.validate_license.return_value = False
        with patch.object(serve,'load_server',return_value=server), self.assertRaisesRegex(RuntimeError,'license'):
            serve.create_app()
        server.OCRPipeline.assert_not_called()

    def test_full_factory_sets_pipeline_and_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)/'model'; model.write_bytes(b'mock')
            server = self.server(model)
            with patch.object(serve,'load_server',return_value=server):
                self.assertIs(serve.create_app(),server.app_online)
            self.assertIs(server.pipeline,server.OCRPipeline.return_value)
            server.OCRPipeline.assert_called_once_with(use_gpu=False)
            server.initialize_counter_file.assert_called_once()
            self.assertEqual(server.app_online.config['MAX_CONTENT_LENGTH'],24*1024*1024)

    def test_missing_model_and_paddle_mode_fail_before_load(self):
        for paddle in (False,True):
            server = self.server('missing')
            server.OCRPipeline.FORCE_PADDLE = paddle
            with patch.object(serve,'load_server',return_value=server), self.assertRaises(RuntimeError):
                serve.create_app()
            server.OCRPipeline.assert_not_called()

    def test_optional_model_loss_is_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)/'model'; model.write_bytes(b'mock')
            for field in ('_yolo_model','_ner_pipeline','_imgpre_model'):
                server = self.server(model)
                setattr(server.OCRPipeline.return_value,field,None)
                with patch.object(serve,'load_server',return_value=server), self.assertRaisesRegex(RuntimeError,'readiness'):
                    serve.create_app()
                server.initialize_counter_file.assert_not_called()

    def test_noninteractive_activation_does_not_generate_or_write(self):
        server = Mock(); server.validate_license.return_value=False
        with patch.object(sys.stdin,'isatty',return_value=False), self.assertRaises(RuntimeError):
            serve.activate(server)
        server.generate_auth_code.assert_not_called()
        server.create_license_file.assert_not_called()

    def test_rejected_activation_does_not_write(self):
        server=Mock(); server.validate_license.return_value=False; server.verify_activation_code.return_value=False
        with patch.object(sys.stdin,'isatty',return_value=True), patch.object(serve.getpass,'getpass',return_value='test-only'), redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            serve.activate(server)
        server.create_license_file.assert_not_called()
        server.initialize_counter_file.assert_not_called()

    def test_valid_activation_uses_existing_verifier_and_persistence(self):
        server=Mock(); server.validate_license.side_effect=[False,True]
        server.generate_auth_code.return_value='test-request';server.get_device_hash.return_value='test-hash'
        server.verify_activation_code.return_value=True
        with patch.object(sys.stdin,'isatty',return_value=True), patch.object(serve.getpass,'getpass',return_value='test-response'), redirect_stdout(io.StringIO()):
            self.assertEqual(serve.activate(server),0)
        server.verify_activation_code.assert_called_once_with('test-request','test-response')
        server.create_license_file.assert_called_once_with('test-request','test-hash')

    def test_server_paddle_imports_are_lazy_and_yolo_outside_branch(self):
        tree=ast.parse((ROOT/'server.py').read_text(encoding='utf-8'))
        self.assertFalse(any(isinstance(n,ast.ImportFrom) and n.module in (
            'layout.paddlepico_detector','layout.table_cell_postprocessor') for n in tree.body))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OCRPipeline')
        load=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='load_models')
        # YOLO path assignment must execute for ONNX and Paddle, not only in an else branch.
        self.assertTrue(any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='yolo_path' for t in n.targets) for n in load.body))

    def test_disk_probe_discards_identifier_and_rejects_write(self):
        r=disk_probe("def _get_disk_serial():\n    return 'test-device-not-for-output'\n")
        self.assertEqual(r,{'readable':True,'identifier_not_saved':True})
        with self.assertRaises(ValueError):
            disk_probe("def _get_disk_serial():\n    return open('file','w')\n")

    def test_layout_package_has_no_eager_paddle_wrapper_import(self):
        tree=ast.parse((ROOT/'layout/__init__.py').read_text(encoding='utf-8'))
        modules=[n.module for n in tree.body if isinstance(n,ast.ImportFrom)]
        self.assertNotIn('paddlepico_detector',modules)
        self.assertNotIn('table_cell_postprocessor',modules)
        self.assertTrue(any(isinstance(n,ast.FunctionDef) and n.name=='__getattr__' for n in tree.body))


if __name__=='__main__':
    unittest.main()
