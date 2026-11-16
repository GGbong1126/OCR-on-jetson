"""CPU-only image inventory checks. No engines or OCR inference."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
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
from inventory_images import detector_profile, discover, inventory, decode, main


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.profile = [[1, 3, 960, 704]] * 3
        self.source = {"status": "completed_consistent_not_accuracy_validated", "stage": "benchmark",
                       "image_sha256": "b"*64, "det_engine_sha256": "a"*64,
                       "detector": {"backend": "tensorrt", "engine_sha256": "a"*64, "profile": self.profile}}
        self.source_path = self.root / "baseline.json"
        self.source_path.write_text(json.dumps(self.source), encoding="utf-8")

    def picture(self, name="page.png", shape=(1500, 1080, 3)):
        path = self.images / name
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(path.suffix, np.full(shape, 255, np.uint8))[1].tofile(str(path))
        return path

    def scan(self, known_files=(), known_pixels=()):
        files, excluded = discover(self.images)
        return inventory(self.images, files, excluded, self.profile, known_files, known_pixels)

    def invoke(self, *extra):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return main(["--images", str(self.images), "--det-report", str(self.source_path),
                         "--output", str(self.root / "report.json"), *extra])

    def test_real_preprocess_portrait_landscape_and_square(self):
        for name, shape in (("portrait.png", (1500,1080,3)), ("landscape.png", (1080,1500,3)),
                            ("square.png", (100,100,3))):
            self.picture(name, shape)
        r = self.scan()
        self.assertEqual(r['summary']['det_input_shape_counts'],
                         {'1x3x704x960':1, '1x3x960x704':1, '1x3x960x960':1})
        self.assertEqual(r['summary']['outside_declared_det_profile_paths'], ['landscape.png','square.png'])
        self.assertEqual(r['recognition_widths']['status'], 'not_measured')

    def test_profile_validation(self):
        self.assertEqual(detector_profile(self.source), self.profile)
        for change in ('status','sha','negative','rank','order','bool'):
            r = deepcopy(self.source)
            if change == 'status': r['stage'] = 'smoke'
            if change == 'sha': r['detector']['engine_sha256'] = 'c'*64
            if change == 'negative': r['detector']['profile'][0][2] = -1
            if change == 'rank': r['detector']['profile'] = [[1,3,960]]*3
            if change == 'order': r['detector']['profile'] = [[1,3,960,704],[1,3,800,704],[1,3,960,704]]
            if change == 'bool': r['detector']['profile'][0][0] = True
            with self.subTest(change=change), self.assertRaises(ValueError): detector_profile(r)

    def test_corrupt_and_excluded_files_not_silently_dropped(self):
        self.picture()
        (self.images/'bad.jpg').write_bytes(b'broken')
        (self.images/'empty.png').write_bytes(b'')
        (self.images/'scan.tiff').write_bytes(b'unhandled multi-page')
        r = self.scan()
        self.assertEqual(r['status'], 'invalid_inventory')
        self.assertEqual(r['summary']['error_paths'], ['bad.jpg','empty.png'])
        self.assertEqual(r['excluded_files'], [{'path':'scan.tiff','reason':'unsupported_extension'}])

    def test_duplicates_and_known_images_by_bytes_and_decoded_pixels(self):
        p = self.picture('a.png')
        (self.images/'b.png').write_bytes(p.read_bytes())
        self.picture('same_pixels.bmp')
        _, sha, pixels, _ = decode(p)
        r = self.scan([sha],[pixels])
        self.assertEqual(r['summary']['duplicate_file_groups'], [['a.png','b.png']])
        self.assertEqual(r['summary']['duplicate_decoded_pixel_groups'], [['a.png','b.png','same_pixels.bmp']])
        self.assertEqual(len(r['summary']['known_used_image_paths']), 3)

    def test_recursive_unicode_and_case_insensitive_extension(self):
        self.picture('子目录/页面.PNG')
        r = self.scan()
        self.assertEqual(r['records'][0]['path'], '子目录/页面.PNG')
        self.assertEqual(r['status'], 'completed_not_inference_validated')

    def test_empty_directory_is_invalid(self):
        self.assertEqual(self.scan()['status'], 'invalid_inventory')
        self.assertEqual(self.invoke(), 2)
        self.assertTrue((self.root/'report.json').exists())

    def test_cli_no_session_creation_and_exclusive_output(self):
        p = self.picture()
        before = p.read_bytes()
        with patch('ocr_det.ort.InferenceSession', side_effect=AssertionError('must not infer')):
            self.assertEqual(self.invoke('--used-image', str(p)), 0)
        report = json.loads((self.root/'report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['summary']['known_used_image_paths'], ['page.png'])
        self.assertEqual(p.read_bytes(), before)
        saved = (self.root/'report.json').read_bytes()
        with self.assertRaises(SystemExit): self.invoke()
        self.assertEqual((self.root/'report.json').read_bytes(), saved)

    def test_invalid_source_or_missing_used_image_no_output(self):
        self.picture()
        with self.assertRaises(SystemExit): self.invoke('--used-image', str(self.root/'missing.png'))
        self.assertFalse((self.root/'report.json').exists())
        self.source_path.write_text('{}', encoding='utf-8')
        with self.assertRaises(SystemExit): self.invoke()
        self.assertFalse((self.root/'report.json').exists())

    def test_refuse_output_inside_dataset(self):
        self.picture()
        with self.assertRaises(SystemExit): self.invoke('--output', str(self.images/'report.json'))
        self.assertFalse((self.images/'report.json').exists())

    def test_linked_files_are_listed_without_reading(self):
        p = self.picture()
        with patch.object(Path, 'is_symlink', lambda path: path == p):
            files, excluded = discover(self.images)
        self.assertEqual(files, [])
        self.assertEqual(excluded, [{'path':'page.png','reason':'linked_file'}])


if __name__ == '__main__':
    unittest.main()
