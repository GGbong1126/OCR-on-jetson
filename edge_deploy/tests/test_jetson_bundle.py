"""Offline host tests; no Jetson/CUDA execution is claimed."""
import gzip
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'prepare_trt82_bundle', Path(__file__).parents[1] / 'jetson' / 'prepare_trt82_bundle.py')
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


class BundleTests(unittest.TestCase):
    def fixture(self, root, filename='pool/main/x/example_1_arm64.deb', arch='arm64'):
        data = ('Package: example\nVersion: 1\nArchitecture: {}\n'
                'Filename: {}\nSize: 10\nInstalled-Size: 20\nSHA256: {}\n'
                .format(arch, filename, 'a' * 64)).encode()
        zipped = gzip.compress(data)
        (root / 'Packages.gz').write_bytes(zipped)
        (root / 'Release').write_text('SHA256:\n {} {} main/binary-arm64/Packages.gz\n'
            .format(hashlib.sha256(zipped).hexdigest(), len(zipped)))

    def test_pinned_selection(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(bundle, 'PINS', {'example': '1'}):
            p = Path(tmp)
            self.fixture(p)
            rows = bundle.select_packages(p)
            self.assertEqual(rows[0]['installed_kib'], 20)
            self.assertTrue(rows[0]['url'].startswith(bundle.BASE))

    def test_changed_index_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(bundle, 'PINS', {'example': '1'}):
            p = Path(tmp)
            self.fixture(p)
            with (p / 'Packages.gz').open('ab') as f:
                f.write(b'corrupted')
            with self.assertRaises(ValueError):
                bundle.select_packages(p)

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(bundle, 'PINS', {'example': '1'}):
            p = Path(tmp)
            self.fixture(p, 'pool/main/../example_1_arm64.deb')
            with self.assertRaises(ValueError):
                bundle.select_packages(p)

    def test_wrong_architecture_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(bundle, 'PINS', {'example': '1'}):
            p = Path(tmp)
            self.fixture(p, arch='amd64')
            with self.assertRaises(ValueError):
                bundle.select_packages(p)

    def test_missing_version_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(bundle, 'PINS', {'example': '2'}):
            p = Path(tmp)
            self.fixture(p)
            with self.assertRaises(ValueError):
                bundle.select_packages(p)


if __name__ == '__main__':
    unittest.main()
