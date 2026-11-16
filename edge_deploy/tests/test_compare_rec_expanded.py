"""Targeted expanded-profile regression, CPU/mock only."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_rec_expanded import select_overflow, make_sample, prepare_samples, compare_samples, summary, main
from ocr_det import sha256_file


class FakeRec:
    characters = ["blank", "A"]
    profile = [[1,3,48,w] for w in (32,160,1536)]

    def __init__(self, *args):
        self.raw = np.array([[[0.0,1.0],[1.0,0.0]]],np.float32)
        self.calls = 0
        self.closed = False

    def run_tensor(self, tensor):
        self.calls += 1
        return self.raw.copy()

    def metadata(self):
        return {"backend":"fake_test_only", "profile": self.profile}

    def close(self):
        self.closed = True


class ExpandedTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((48,1100,3),255,np.uint8)
        self.box = [[0,0],[1100,0],[1100,48],[0,48]]
        self.row = {"index":0,"box":self.box,"input_shape":[1,3,48,1100],"status":"inventoried",
                    "within_declared_rec_profile":False,"legacy_width_cap_applied":False,"input_sha256":"tensor"}
        self.baseline = {"status":"completed_consistent_not_accuracy_validated", "reference_rec_onnx_sha256":"model",
                         "dictionary_sha256":"dict", "recognizer":{"profile":[[1,3,48,w] for w in (32,160,1024)],"engine_sha256":"old"}}
        self.scan = {"status":"completed_inventory_not_rec_validated","rec_baseline_sha256":"baseline",
                     "declared_rec_profile":self.baseline['recognizer']['profile'],"declared_rec_engine_sha256":"old",
                     "expected_image_count":1, "images":[{"path":"long.png","image_sha256":"image",
                     "detection":{"ordered_boxes_exactly_equal":True},
                     "branches":{b:{"records":[deepcopy(self.row)]} for b in ('ort','trt')}}]}

    def select(self, scan=None):
        return select_overflow(scan or self.scan,self.baseline,'baseline','model','dict')

    def test_select_only_over_profile_and_preserve_shape(self):
        selected = self.select()
        self.assertEqual(len(selected),1)
        self.assertEqual(selected[0]['input_shape'][3],1100)

    def test_selection_rejects_wrong_source_profile_or_flags(self):
        for kind in ('hash','engine','profile','flag','index','branch','cap','count'):
            s=deepcopy(self.scan)
            if kind=='hash': s['rec_baseline_sha256']='wrong'
            if kind=='engine': s['declared_rec_engine_sha256']='wrong'
            if kind=='profile': s['declared_rec_profile'][2][3]=1536
            if kind=='flag': s['images'][0]['branches']['trt']['records'][0]['within_declared_rec_profile']=True
            if kind=='index': s['images'][0]['branches']['trt']['records'][0]['index']=4
            if kind=='branch': s['images'][0]['branches']['ort']['records'][0]['input_sha256']='wrong'
            if kind=='cap': s['images'][0]['branches']['trt']['records'][0]['legacy_width_cap_applied']=True
            if kind=='count': s['expected_image_count']=2
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.select(s)

    def test_sample_tensor_hash_and_shape_binding(self):
        s=make_sample(self.image,self.row,'sample')
        self.assertEqual(s['tensor'].shape,(1,3,48,1100))
        with self.assertRaises(ValueError): make_sample(self.image,self.row,'sample',expected_hash='bad')
        with self.assertRaises(ValueError): make_sample(self.image,{**self.row,'input_shape':[1,3,48,100]},'sample')

    def test_strict_text_and_no_text_in_saved_records(self):
        samples=[make_sample(self.image,self.row,'long'),make_sample(self.image,self.row,'old',expected_text='A')]
        records=[]
        with redirect_stdout(io.StringIO()): compare_samples(FakeRec(),FakeRec(),samples,records)
        s=summary(records,2)
        self.assertEqual(s['status'],'completed_strict_consistent_not_accuracy_validated')
        self.assertEqual(s['text_equal_count'],2)
        self.assertFalse(any(k in row for row in records for k in ('reference_text','candidate_text','expected_text','tensor')))

    def test_text_drift_empty_and_nonfinite_are_not_hidden(self):
        sample=make_sample(self.image,self.row,'old',expected_text='different')
        cpu,gpu=FakeRec(),FakeRec(); gpu.raw[:]=[[[1,0],[1,0]]]
        records=[]
        with redirect_stdout(io.StringIO()): compare_samples(cpu,gpu,[sample],records)
        s=summary(records,1)
        self.assertEqual(s['status'],'completed_with_text_changes_not_accepted')
        self.assertEqual(s['reference_drift_ids'],['old'])
        self.assertEqual(s['candidate_empty_ids'],['old'])
        gpu.raw[:]=np.nan; records=[]
        with redirect_stdout(io.StringIO()): compare_samples(cpu,gpu,[sample],records)
        self.assertEqual(summary(records,1)['status'],'invalid_comparison')

    def test_insufficient_profile_rejected_before_inference(self):
        cpu,gpu=FakeRec(),FakeRec(); gpu.profile=[[1,3,48,w] for w in (32,160,1024)]
        with self.assertRaises(ValueError): compare_samples(cpu,gpu,[make_sample(self.image,self.row,'x')],[])
        self.assertEqual(cpu.calls+gpu.calls,0)
        self.assertEqual(summary([],1)['status'],'invalid_comparison')

    def cli(self, root):
        imagepath=root/'long.png'; cv2.imencode('.png',self.image)[1].tofile(str(imagepath))
        model=root/'model.onnx'; model.write_bytes(b'model')
        engine=root/'new.engine'; engine.write_bytes(b'engine')
        dictionary=root/'dict.json'; dictionary.write_text('["A"]')
        baseline=deepcopy(self.baseline)
        baseline.update(reference_rec_onnx_sha256=sha256_file(model),dictionary_sha256=sha256_file(dictionary))
        basepath=root/'baseline.json'; basepath.write_text(json.dumps(baseline))
        scan=deepcopy(self.scan); scan['rec_baseline_sha256']=sha256_file(basepath)
        scan['images'][0]['image_sha256']=sha256_file(imagepath)
        tensor=make_sample(self.image,self.row,'x')['tensor']
        for branch in ('ort','trt'):
            scan['images'][0]['branches'][branch]['records'][0]['input_sha256']=hashlib.sha256(tensor.tobytes()).hexdigest()
        scanpath=root/'scan.json'; scanpath.write_text(json.dumps(scan))
        source={'status':'compared_not_accuracy_validated','image_sha256':sha256_file(imagepath),
                'recognizer':{'model_sha256':sha256_file(model),'dictionary_sha256':sha256_file(dictionary)},
                'reference':{'records':[{'index':0,'box':[[0,0],[80,0],[80,48],[0,48]],'input_shape':[1,3,48,80],
                                          'text':'A','status':'ok'}]}}
        sourcepath=root/'source.json'; sourcepath.write_text(json.dumps(source))
        pairs={'onnx':model,'engine':engine,'dict':dictionary,'images':root,'scan-report':scanpath,
               'baseline-report':basepath,'reference-report':sourcepath,'reference-image':imagepath,'output':root/'result.json'}
        return [part for key,value in pairs.items() for part in ('--'+key,str(value))]

    def test_cli_success_and_existing_output_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); args=self.cli(root)
            with patch('compare_rec_expanded.Recognizer',FakeRec),patch('compare_rec_expanded.TensorRTRecognizer',FakeRec),redirect_stdout(io.StringIO()):
                self.assertEqual(main(args),0)
            r=json.loads((root/'result.json').read_text())
            self.assertEqual(r['overflow_count'],1)
            self.assertEqual(r['summary']['compared_count'],2)
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): main(args)

    def test_cleanup_failure_invalidates_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); args=self.cli(root)
            with patch('compare_rec_expanded.Recognizer',FakeRec),patch('compare_rec_expanded.TensorRTRecognizer',FakeRec),patch.object(FakeRec,'close',side_effect=RuntimeError('cleanup')),redirect_stdout(io.StringIO()):
                self.assertEqual(main(args),2)
            r=json.loads((root/'result.json').read_text())
            self.assertEqual(r['summary']['status'],'invalid_comparison')
            self.assertIn('cleanup_error',r)

    def test_changed_image_fails_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); args=self.cli(root)
            (root/'long.png').write_bytes(b'changed')
            with patch('compare_rec_expanded.Recognizer') as cpu,redirect_stdout(io.StringIO()):
                self.assertEqual(main(args),2)
            cpu.assert_not_called()

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); self.cli(root)
            source=json.loads((root/'source.json').read_text())
            selected=self.select(); selected[0]['path']='../outside.png'
            with self.assertRaisesRegex(ValueError,'escapes'):
                prepare_samples(source,root/'long.png',source['recognizer'],root,selected)


if __name__=='__main__':
    unittest.main()
