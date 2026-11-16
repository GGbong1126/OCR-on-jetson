"""Synthetic hardware only. No real activation, license write, counters or models."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import hmac
import io
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import windows_hardware as hw

PROTOCOL_CONSTANTS = {'CHARSET', 'BASE', 'AUTH_CODE_LEN', 'MAX_NUM', 'APP_SEED', 'SECRET_KEY',
                      'password3', 'key29', 'COUNTER_INDICES', 'INITIAL_COUNTER',
                      'PLAIN_BYTES_LEN', 'PAYLOAD_LEN', '_positions1', '_positions2'}


def digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def pure_namespace(filename):
    tree = ast.parse((ROOT/filename).read_text(encoding='utf-8-sig'))
    allowed = {'_get_disk_serial', 'build_plain_bytes', '_mask', 'mix_number', 'unmix_number',
               'int_to_fixed_base58', 'fixed_base58_to_int', 'generate_auth_code',
               'extract_serial_from_auth', 'base58_encode_fixed', 'encrypt_auth_code',
               'verify_activation_code', 'get_device_fingerprint', 'get_device_hash', 'validate_license',
               'generate_activation_code'}
    constants = PROTOCOL_CONSTANTS | {'positions1', 'positions2'}
    nodes = [n for n in tree.body if
             (isinstance(n, ast.FunctionDef) and n.name in allowed) or
             (isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id in constants for t in n.targets))]
    env = dict(random=random.Random(), hashlib=hashlib, hmac=hmac, sha256=hashlib.sha256,
               sys=SimpleNamespace(platform='win32'), windows_hardware=hw, subprocess=subprocess,
               platform=SimpleNamespace(processor=lambda: 'TEST CPU', system=lambda: 'Windows',
                                        version=lambda: 'TEST OS', machine=lambda: 'AMD64'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), env)
    return env


class HardwareReaderTests(unittest.TestCase):
    def fallback(self, command, rows, text=False):
        with patch.object(hw.sys, 'platform', 'win32'), patch.object(hw, '_wmic_executable', return_value=None), \
                patch.object(hw, '_query_cim', return_value=rows):
            return hw.check_output(command, text=text)

    def test_existing_wmic_preserved_and_cim_not_used(self):
        with patch.object(hw.sys, 'platform', 'win32'), patch.object(hw, '_wmic_executable', return_value='wmic.exe'), \
                patch.object(hw.subprocess, 'check_output', return_value=b'legacy-exact') as call, \
                patch.object(hw, '_query_cim') as cim:
            self.assertEqual(hw.check_output('wmic diskdrive get SerialNumber'), b'legacy-exact')
        cim.assert_not_called()
        self.assertEqual(call.call_args.args[0], ['wmic.exe', 'diskdrive', 'get', 'SerialNumber'])
        self.assertNotIn('shell', call.call_args.kwargs)

    def test_present_wmic_failure_does_not_switch_identity_or_leak_output(self):
        error = subprocess.CalledProcessError(1, 'test', output=b'private-identifier')
        with patch.object(hw.sys, 'platform', 'win32'), patch.object(hw, '_wmic_executable', return_value='wmic.exe'), \
                patch.object(hw.subprocess, 'check_output', side_effect=error), patch.object(hw, '_query_cim') as cim:
            with self.assertRaises(RuntimeError) as raised: hw.check_output('wmic diskdrive get SerialNumber')
        self.assertNotIn('private', str(raised.exception))
        cim.assert_not_called()

    def test_non_windows_uses_original_subprocess(self):
        with patch.object(hw.sys, 'platform', 'linux'), patch.object(hw.subprocess, 'check_output', return_value=b'old') as call, \
                patch.object(hw, '_query_cim') as cim:
            self.assertEqual(hw.check_output('wmic cpu get Name'), b'old')
        call.assert_called_once_with('wmic cpu get Name', shell=True, text=False)
        cim.assert_not_called()

    def test_serial_case_internal_spaces_and_order_preserved(self):
        out = self.fallback('wmic diskdrive get SerialNumber',
                            [{'SerialNumber': '  zX_a B12  '}, {'SerialNumber': 'AAA'}], True)
        self.assertEqual([x.strip() for x in out.splitlines() if x.strip()], ['SerialNumber', 'zX_a B12', 'AAA'])

    def test_bytes_and_text_responses_match(self):
        rows = [{'Name': 'CPU 中文'}]
        self.assertEqual(self.fallback('wmic cpu get Name', rows).decode(),
                         self.fallback('wmic cpu get Name', rows, True))

    def test_board_table_header_width_not_single_space(self):
        out = self.fallback('wmic baseboard get Manufacturer,Product', [{'Manufacturer': 'ACME', 'Product': 'P 1'}], True)
        self.assertEqual(out.splitlines()[1].strip(), 'ACME'+' '*10+'P 1')

    def test_board_width_uses_all_rows_and_preserves_empty_values(self):
        out = self.fallback('wmic baseboard get Manufacturer,Product', [
            {'Manufacturer': 'A', 'Product': 'P'}, {'Manufacturer': 'X'*20, 'Product': None}], True)
        self.assertEqual(out.splitlines()[1].strip(), 'A'+' '*21+'P')
        self.assertEqual(out.splitlines()[2].strip(), 'X'*20)

    def test_empty_serial_does_not_become_placeholder_identity(self):
        env = pure_namespace('server.py')
        with patch.object(hw.sys, 'platform', 'win32'), patch.object(hw, '_wmic_executable', return_value=None), \
                patch.object(hw, '_query_cim', return_value=[{'SerialNumber': None}]):
            with self.assertRaises(RuntimeError): env['_get_disk_serial']()

    def test_multiline_properties_rejected(self):
        with self.assertRaises(RuntimeError):
            self.fallback('wmic diskdrive get SerialNumber', [{'SerialNumber': 'line1\nline2'}])

    def test_only_allowlisted_commands(self):
        with patch.object(hw.subprocess, 'run') as run:
            with self.assertRaises(ValueError): hw.check_output('wmic diskdrive get SerialNumber & anything')
            with self.assertRaises(ValueError): hw._query_cim('Win32_Process', ('Name',))
        run.assert_not_called()

    def test_cim_command_uses_fixed_json_and_no_shell_or_profiles(self):
        with patch.object(hw.subprocess, 'run', return_value=SimpleNamespace(stdout=b'[{"SerialNumber":"test"}]')) as run:
            self.assertEqual(hw._query_cim('Win32_DiskDrive', ('SerialNumber',)), [{'SerialNumber': 'test'}])
        argv = run.call_args.args[0]
        self.assertIn('-NoProfile', argv)
        self.assertIn('-NonInteractive', argv)
        self.assertIn('ConvertTo-Json', argv[-1])
        self.assertIn('root/cimv2', argv[-1])
        self.assertNotIn('shell', run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs['timeout'], hw.QUERY_TIMEOUT)

    def test_cim_singleton_empty_and_invalid_json(self):
        for raw, expected in ((b'{"Name":"test"}', [{'Name': 'test'}]), (b'[]', [])):
            with patch.object(hw.subprocess, 'run', return_value=SimpleNamespace(stdout=raw)):
                self.assertEqual(hw._query_cim('Win32_Processor', ('Name',)), expected)
        for raw in (b'private-invalid-json', b'null', b'[{}]', b'[{"Name":1}]'):
            with patch.object(hw.subprocess, 'run', return_value=SimpleNamespace(stdout=raw)):
                with self.assertRaises(RuntimeError) as raised: hw._query_cim('Win32_Processor', ('Name',))
            self.assertNotIn('private', str(raised.exception))

    def test_cim_timeout_sanitized(self):
        with patch.object(hw.subprocess, 'run', side_effect=subprocess.TimeoutExpired('query', 15, output=b'private')):
            with self.assertRaises(RuntimeError) as raised: hw._query_cim('Win32_Processor', ('Name',))
        self.assertNotIn('private', str(raised.exception))


class LegacyProtocolTests(unittest.TestCase):
    def test_all_other_authorization_and_counter_functions_unchanged(self):
        tree = ast.parse((ROOT/'server.py').read_text(encoding='utf-8-sig'))
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        names = [n.name for n in funcs]
        selected = [n for n in funcs[names.index('verify_disk_serial'):names.index('check_license')+1]
                    if n.name != 'get_device_fingerprint']
        selected += [n for n in tree.body if isinstance(n, ast.Assign) and
                     all(isinstance(t, ast.Name) and t.id in PROTOCOL_CONSTANTS for t in n.targets)]
        # E67 pre-patch AST, before hardware adapter (not a new algorithm baseline).
        self.assertEqual(digest(ast.Module(body=selected, type_ignores=[])),
                         'acd24faa63e82720efd974549cdd0d0e57807fc2d1b2b2bf0518e4334fa63106')

    def test_non_windows_disk_code_unchanged(self):
        tree = ast.parse((ROOT/'server.py').read_text(encoding='utf-8-sig'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_get_disk_serial')
        self.assertEqual(digest(ast.Module(body=node.body[0].orelse, type_ignores=[])),
                         '600bd07bb632993821f5db8e45806556ad68470eb0a8ea5d26941942b1da8e6f')

    def test_fingerprint_logic_only_changes_reader(self):
        tree = ast.parse((ROOT/'server.py').read_text(encoding='utf-8-sig'))
        node = deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_device_fingerprint'))
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id == 'windows_hardware': n.id = 'subprocess'
        self.assertEqual(digest(node), '149bda00dccc5e6b0f95621f5008df5d3ce2fb251d67664f9063f713d1321588')

    def test_hardware_activation_generator_not_edited(self):
        # Normalize checkout newlines while retaining the pre-patch CRLF file hash.
        raw = (ROOT/'generate_12.py').read_text(encoding='utf-8').replace('\n', '\r\n').encode('utf-8')
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         '6b225ecae009a36744af4b9400291faff91cbd0df18382f93f298f2f3c82bfb7')

    def test_legacy_request_and_activation_compatible_across_reader_paths(self):
        generator = pure_namespace('generate_12.py')
        for serial in ('TEST123', 'ABCDEFGHIJKLMNOPQRS', 'ABCDEFGHIJKLMNOPQRST', 'LONG_TEST_SERIAL_NUMBER_12345', 'Ab_123 456'):
            for seed in (0, 17, 2026):
                with self.subTest(length=len(serial), seed=seed):
                    server = pure_namespace('server.py')
                    server['random'].seed(seed)
                    legacy = f'SerialNumber\r\n  {serial}  \r\n'
                    with patch.object(hw.sys, 'platform', 'win32'), patch.object(hw, '_wmic_executable', return_value='wmic.exe'), \
                            patch.object(hw.subprocess, 'check_output', return_value=legacy):
                        old_request = server['generate_auth_code']()
                    server['random'].seed(seed)
                    with patch.object(hw.sys, 'platform', 'win32'), patch.object(hw, '_wmic_executable', return_value=None), \
                            patch.object(hw, '_query_cim', return_value=[{'SerialNumber': f'  {serial}  '}]):
                        new_request = server['generate_auth_code']()
                    self.assertEqual(old_request, new_request)
                    with redirect_stdout(io.StringIO()): code = generator['generate_activation_code'](old_request)
                    self.assertTrue(server['verify_activation_code'](new_request, code))
                    corrupt = ('A' if code[0] != 'A' else 'B') + code[1:]
                    self.assertFalse(server['verify_activation_code'](new_request, corrupt))
                    self.assertEqual(server['extract_serial_from_auth'](new_request), serial[-16:] if len(serial)>19 else serial)

    def test_different_requests_need_their_own_activation_codes(self):
        server, generator = pure_namespace('server.py'), pure_namespace('generate_12.py')
        server['_get_disk_serial'] = lambda: 'TEST-SHORT'
        server['random'].seed(1); a = server['generate_auth_code']()
        server['random'].seed(2); b = server['generate_auth_code']()
        self.assertNotEqual(a, b)
        with redirect_stdout(io.StringIO()): code = generator['generate_activation_code'](a)
        self.assertFalse(server['verify_activation_code'](b, code))

    def test_fingerprint_synthetic_wmic_and_cim_tables_match(self):
        env = pure_namespace('server.py')
        def old(command, **kwargs):
            if 'baseboard' in command: return b'Manufacturer  Product\r\nACME          P1\r\n'
            return b'SerialNumber\r\nTEST123\r\n'
        with patch.object(hw, 'check_output', side_effect=old): expected = env['get_device_fingerprint']()
        def rows(class_name, columns):
            return [{'Manufacturer': 'ACME', 'Product': 'P1'}] if class_name == 'Win32_BaseBoard' else [{'SerialNumber': 'TEST123'}]
        with patch.object(hw.sys, 'platform', 'win32'), patch.object(hw, '_wmic_executable', return_value=None), patch.object(hw, '_query_cim', side_effect=rows):
            self.assertEqual(env['get_device_fingerprint'](), expected)


if __name__ == '__main__':
    unittest.main()
