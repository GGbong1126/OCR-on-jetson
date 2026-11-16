"""CPU/mock manifest batch scan checks, not GPU validation."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from inventory_images import inventory, discover
from scan_det_rec_shapes import check_manifest, crop_inventory, analyze_comparison, summarize, main
from ocr_det import sha256_file


class BatchScanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root/'images'
        self.images.mkdir()
        self.image = np.full((48,1100,3),255,np.uint8)
        self.path = self.images/'a.png'
        cv2.imencode('.png',self.image)[1].tofile(str(self.path))
        files, excluded = discover(self.images)
        self.manifest = inventory(self.images,files,excluded,[[1,3,32,960]]*3)
        self.row = self.manifest['records'][0]
        self.profile = [[1,3,48,w] for w in (32,160,1024)]
        self.buckets = [64,128,160,256,512,960,1024]
        self.boxes = [[[0,0],[80,0],[80,48],[0,48]]]
        self.result = {'status':'finite_outputs_compared', 'ort_output':{'all_finite':True},
                       'trt_output':{'all_finite':True}, 'image_sha256':self.row['image_sha256'],
                       'input_shape':self.row['det_input_shape'], 'ort_metadata':{'model_sha256':'onnx'},
                       'trt_metadata':{'engine_sha256':'engine'}, 'ort_boxes':self.boxes, 'trt_boxes':self.boxes,
                       'ort_box_count':1,'trt_box_count':1,'ordered_boxes_exactly_equal':True,
                       'probability_mae':0.0,'probability_max_abs_error':0.0,
                       'allclose_rtol_1e_3_atol_1e_4':True,'mask_disagreement_fraction_at_0_1':0.0}

    def analyze(self, result=None):
        return analyze_comparison(result or self.result,self.image,self.row,'onnx','engine',self.buckets,self.profile)

    def test_manifest_exact_set_and_tensor_hash(self):
        self.assertEqual(check_manifest(self.images,self.manifest),[self.row])
        bad = deepcopy(self.manifest)
        bad['records'][0]['det_input_sha256'] = 'bad'
        with self.assertRaises(ValueError): check_manifest(self.images,bad)
        (self.images/'extra.png').write_bytes(self.path.read_bytes())
        with self.assertRaises(ValueError): check_manifest(self.images,self.manifest)

    def test_changed_image_rejected(self):
        self.path.write_bytes(b'broken')
        with self.assertRaises(ValueError): check_manifest(self.images,self.manifest)

    def test_exact_branches_and_buckets(self):
        r = self.analyze()
        self.assertEqual(r['branches']['ort'],r['branches']['trt'])
        crop = r['branches']['trt']['records'][0]
        self.assertEqual(crop['input_shape'],[1,3,48,80])
        self.assertEqual(crop['bucket_width'],128)
        self.assertTrue(crop['within_declared_rec_profile'])
        self.assertEqual(summarize([r])['trt']['width_counts'],{80:1})

    def test_width_over_1024_retained_without_bucket(self):
        box = [[0,0],[1100,0],[1100,48],[0,48]]
        r = crop_inventory(self.image,[box],self.buckets,self.profile)
        row = r['records'][0]
        self.assertEqual(row['input_shape'][3],1100)
        self.assertFalse(row['within_declared_rec_profile'])
        self.assertIsNone(row['bucket_width'])
        self.assertEqual(r['summary']['no_bucket_indices'],[0])

    def test_legacy_3200_cap_and_empty_crop_are_explicit(self):
        image = np.zeros((10,1000,3),np.uint8)
        boxes = [[[0,0],[1000,0],[1000,10],[0,10]], [[1,1]]*4]
        r = crop_inventory(image,boxes,self.buckets,self.profile)
        self.assertEqual(r['records'][0]['aspect_width_before_legacy_cap'],4800)
        self.assertEqual(r['records'][0]['input_shape'][3],3200)
        self.assertEqual(r['summary']['legacy_width_cap_indices'],[0])
        self.assertEqual(r['summary']['invalid_crop_indices'],[1])

    def test_branch_differences_are_separate_not_matched_by_index(self):
        result = deepcopy(self.result)
        result.update(trt_boxes=[],trt_box_count=0,ordered_boxes_exactly_equal=False)
        r = self.analyze(result)
        summary = summarize([r])
        self.assertEqual(summary['det_box_difference_paths'],['a.png'])
        self.assertEqual(summary['trt']['zero_box_paths'],['a.png'])
        self.assertEqual(summary['ort']['box_count'],1)

    def test_nonfinite_provenance_and_box_count_fail(self):
        for kind in ('finite','hash','shape','count','equality'):
            r=deepcopy(self.result)
            if kind=='finite': r['trt_output']['all_finite']=False
            if kind=='hash': r['image_sha256']='bad'
            if kind=='shape': r['input_shape']=[1,3,960,704]
            if kind=='count': r['ort_box_count']=2
            if kind=='equality': r['ordered_boxes_exactly_equal']=False
            with self.subTest(kind=kind),self.assertRaises(ValueError): self.analyze(r)

    def setup_cli(self):
        self.inv=self.root/'inventory.json'; self.inv.write_text(json.dumps(self.manifest))
        self.base=self.root/'baseline.json'
        self.base.write_text(json.dumps({'status':'completed_consistent_not_accuracy_validated',
                                        'recognizer':{'backend':'tensorrt','profile':self.profile,'engine_sha256':'r'*64}}))
        self.onnx=self.root/'det.onnx'; self.onnx.write_bytes(b'fake onnx')
        self.engine=self.root/'det.engine'; self.engine.write_bytes(b'fake engine')
        self.out=self.root/'output'
        return ['--images',str(self.images),'--inventory',str(self.inv),'--onnx',str(self.onnx),
                '--engine',str(self.engine),'--rec-baseline',str(self.base),'--output-dir',str(self.out)]

    def fake_child(self, args, **kwargs):
        result=deepcopy(self.result)
        result['ort_metadata']['model_sha256']=sha256_file(self.onnx)
        result['trt_metadata']['engine_sha256']=sha256_file(self.engine)
        Path(args[args.index('--output')+1]).write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)

    def test_cli_fresh_child_and_exclusive_output(self):
        args=self.setup_cli()
        with patch('scan_det_rec_shapes.subprocess.run',side_effect=self.fake_child) as child, redirect_stdout(io.StringIO()):
            self.assertEqual(main(args),0)
        child.assert_called_once()
        r=json.loads((self.out/'summary.json').read_text())
        self.assertEqual(r['status'],'completed_inventory_not_rec_validated')
        self.assertEqual(r['expected_image_count'],1)
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): main(args)

    def test_child_failure_preserves_partial_report(self):
        args=self.setup_cli()
        with patch('scan_det_rec_shapes.subprocess.run',return_value=SimpleNamespace(returncode=2)), redirect_stdout(io.StringIO()):
            self.assertEqual(main(args),2)
        r=json.loads((self.out/'summary.json').read_text())
        self.assertEqual(r['status'],'invalid_scan')
        self.assertEqual(r['summary']['compared_image_count'],0)
        self.assertTrue((self.out/'det_0000.log').exists())

    def test_bad_manifest_prevents_child_and_output(self):
        args=self.setup_cli()
        (self.images/'new.txt').write_text('not inventoried')
        with patch('scan_det_rec_shapes.subprocess.run') as child,redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): main(args)
        child.assert_not_called()
        self.assertFalse(self.out.exists())


if __name__=='__main__':
    unittest.main()
