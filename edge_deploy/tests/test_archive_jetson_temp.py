"""Small local fixtures only; no models or board operations."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('archive_temp', Path(__file__).resolve().parents[1] / 'tools/archive_jetson_temp.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class ArchiveTests(unittest.TestCase):
    def test_verified_copy_preserves_original_and_empty_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); source = base / 'source'; source.mkdir()
            (source / 'trial/empty').mkdir(parents=True)
            (source / 'trial/a.txt').write_bytes(b'original\r\n')
            (source / 'other.txt').write_bytes(b'excluded')
            destination = base / 'archive'
            result = module.archive(source, destination, {'trial': 'fixture'})
            self.assertEqual(result['files'], 1)
            self.assertEqual((source / 'trial/a.txt').read_bytes(), b'original\r\n')
            self.assertTrue((destination / 'payload/temp/trial/empty').is_dir())
            self.assertFalse((destination / 'payload/temp/other.txt').exists())
            self.assertEqual(json.loads((destination / 'selection.json').read_text())['excluded'][0]['name'], 'other.txt')
            with self.assertRaises(FileExistsError): module.archive(source, destination, {'trial': 'fixture'})

    def test_detects_modified_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); source = base / 'source'; source.mkdir()
            (source / 'trial').write_bytes(b'abc')
            destination = base / 'archive'
            module.archive(source, destination, {'trial': 'fixture'})
            (destination / 'payload/temp/trial').write_bytes(b'xyz')
            with self.assertRaises(ValueError): module.verify(destination, source)

    def test_nested_destination_and_unsafe_selection_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source'; source.mkdir()
            with self.assertRaises(ValueError): module.archive(source, source / 'archive', {})
            with self.assertRaises(ValueError): module.archive(source, Path(folder) / 'archive', {'../other': ''})

    def test_missing_selection_does_not_create_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source'; source.mkdir()
            destination = Path(folder) / 'archive'
            with self.assertRaises(FileNotFoundError): module.archive(source, destination, {'missing': ''})
            self.assertFalse(destination.exists())


if __name__ == '__main__': unittest.main()
