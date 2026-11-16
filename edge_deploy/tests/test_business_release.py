"""Release packaging tests with fake bytes: no imports of server/models/license logic."""
import copy
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'business'))
import release_business as release


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source, self.output = self.base / 'source', self.base / 'release'
        for name in (*release.SOURCE_FILES, release.OPTIONAL, 'pth/imgpre/only.pth'):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('fixture:' + name).encode())
        self.env = self.base / 'env.json'
        self.environment = {'schema_version': 1, 'platform': 'Linux', 'machine': 'x86_64',
                            'python': '3.10.12', 'packages': {'torch': '2.8.0+cpu'}}
        self.env.write_text(json.dumps(self.environment))

    def build(self):
        return release.build(self.source, self.output, 'fixture-r1', self.env)

    def test_explicit_payload_preserves_yaml_and_ignores_private_files(self):
        for name in ('license.dat', 'generate_12.py', 'server.log', 'pth/imgpre/private.json',
                     release.NER + 'training_args.bin', release.NER + 'checkpoint-1/private.txt'):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('DO_NOT_PACKAGE')
        result = self.build()
        self.assertEqual(result['status'], 'verified_files_not_deployed')
        self.assertEqual((self.output / release.OPTIONAL).read_bytes(), (self.source / release.OPTIONAL).read_bytes())
        self.assertFalse((self.output / 'license.dat').exists())
        self.assertFalse((self.output / release.NER / 'training_args.bin').exists())
        self.assertFalse((self.output / 'pth/imgpre/private.json').exists())

    def test_absent_yaml_keeps_json_fallback(self):
        (self.source / release.OPTIONAL).unlink()
        self.build()
        self.assertFalse((self.output / release.OPTIONAL).exists())
        self.assertTrue((self.output / 'pth/onnx_rec/char_dict.json').exists())

    def test_multiple_weights_rejected_before_output(self):
        (self.source / 'pth/imgpre/other.pth').write_bytes(b'other')
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse(self.output.exists())

    def test_uppercase_weight_suffix_matches_server_selection(self):
        path = self.source / 'pth/imgpre/only.pth'
        path.rename(self.source / 'pth/imgpre/only.PTH')
        self.build()
        self.assertTrue((self.output / 'pth/imgpre/only.PTH').exists())

    def test_mixed_case_multiple_weights_are_not_silently_omitted(self):
        (self.source / 'pth/imgpre/other.PTH').write_bytes(b'other')
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse(self.output.exists())

    def test_missing_asset_rejected_before_output(self):
        (self.source / 'server.py').unlink()
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse(self.output.exists())

    def test_existing_output_not_overwritten(self):
        self.build()
        with self.assertRaises(ValueError):
            self.build()

    def test_changed_bytes_rejected(self):
        self.build()
        (self.output / 'server.py').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            release.verify(self.output)

    def test_extra_file_and_empty_runtime_directory_rejected(self):
        self.build()
        path = self.output / 'license.dat'
        path.write_text('private fixture')
        with self.assertRaises(ValueError):
            release.verify(self.output)
        path.unlink()
        (self.output / 'runtime').mkdir()
        with self.assertRaises(ValueError):
            release.verify(self.output)

    def test_manifest_traversal_rejected(self):
        self.build()
        path = self.output / release.MANIFEST
        value = json.loads(path.read_text())
        value['files']['../outside'] = {'bytes': 1, 'sha256': 'a' * 64}
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            release.verify(self.output)

    def test_environment_rejects_urls_and_nonlinux(self):
        for field, value in (('platform', 'Windows'), ('python', '3.12.1'),
                             ('packages', {'torch': 'https://secret@example.invalid/pkg'})):
            data = copy.deepcopy(self.environment)
            data[field] = value
            with self.assertRaises(ValueError):
                release.checked_environment(data)

    def test_source_drift_leaves_no_completion_marker(self):
        original = release.fingerprint
        def drift(path):
            value = original(path)
            if Path(path) == self.output / 'server.py':
                value['sha256'] = '0' * 64
            return value
        with patch.object(release, 'fingerprint', side_effect=drift), self.assertRaises(ValueError):
            self.build()
        self.assertFalse((self.output / release.MANIFEST).exists())

    def test_archive_only_regular_allowlisted_entries_and_no_overwrite(self):
        self.build()
        output = self.base / 'release.tar.gz'
        release.archive(self.output, output)
        with tarfile.open(output) as tar:
            entries = tar.getmembers()
            self.assertTrue(all(p.isfile() and p.name.startswith('fixture-r1/') for p in entries))
            self.assertTrue(all(p.uid == 0 and p.uname == '' for p in entries))
            self.assertEqual(len(entries), release.verify(self.output)['file_count'] + 1)
        with self.assertRaises(ValueError):
            release.archive(self.output, output)

    def test_linked_payload_is_rejected(self):
        path = self.source / 'words.py'
        target = self.base / 'target'
        target.write_bytes(b'fixture')
        path.unlink()
        try:
            path.symlink_to(target)
        except OSError:
            self.skipTest('Symlinks need privileges on this Windows host; run this test in WSL')
        with self.assertRaisesRegex(ValueError, 'Linked'):
            self.build()


if __name__ == '__main__':
    unittest.main()
