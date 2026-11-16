"""Read-only compatibility for the three legacy WMIC queries in server.py.

No license, fingerprint hash, random state, activation or counter operations.
Prefer WMIC when present; use the same local WMI properties via CIM otherwise.
Non-Windows calls deliberately keep the old subprocess behavior.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


QUERIES = {
    'wmic diskdrive get SerialNumber': ('Win32_DiskDrive', ('SerialNumber',)),
    'wmic cpu get Name': ('Win32_Processor', ('Name',)),
    'wmic baseboard get Manufacturer,Product': ('Win32_BaseBoard', ('Manufacturer', 'Product')),
}
QUERY_TIMEOUT = 15


def _wmic_executable():
    found = shutil.which('wmic')
    if found:
        return found
    path = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/wbem/WMIC.exe'
    return str(path) if path.is_file() else None


def _query_cim(class_name, columns):
    # Only constants from QUERIES may enter the script; never interpolate device data.
    if (class_name, columns) not in QUERIES.values():
        raise ValueError('Unsupported hardware query')
    properties = ','.join(columns)
    script = (
        '[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); '
        '$ErrorActionPreference = "Stop"; '
        f'$rows = @(Get-CimInstance -Namespace root/cimv2 -ClassName {class_name} '
        f'-Property {properties} | Select-Object -Property {properties}); '
        'ConvertTo-Json -InputObject $rows -Compress'
    )
    powershell = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    try:
        process = subprocess.run(
            [str(powershell), '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', script],
            capture_output=True, timeout=QUERY_TIMEOUT, check=True,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        rows = json.loads(process.stdout.decode('utf-8-sig'))
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        # Do not leak stdout/stderr (which may contain identifiers) through exception text.
        raise RuntimeError('CIM hardware query failed; no identifier substituted') from None
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        raise RuntimeError('Invalid CIM hardware response')
    for row in rows:
        if not isinstance(row, dict) or any(
            column not in row or (row[column] is not None and not isinstance(row[column], str))
            for column in columns
        ):
            raise RuntimeError('Invalid CIM hardware properties')
    return rows


def _legacy_table(rows, columns):
    """Feed the existing line parsers, including board column padding.

    Keep enumeration order and all property characters. Only the original server
    parser strips each completed row. No uppercase, internal whitespace removal,
    disk sorting, volume ID substitution or serial truncation happens here.
    This emulates the legacy default text table (max(header, values) + 2 padding);
    historical fingerprint equality must still be verified, not assumed.
    """
    values = [[row[column] if row[column] is not None else '' for column in columns] for row in rows]
    if any('\n' in value or '\r' in value or '\0' in value for row in values for value in row):
        raise RuntimeError('Invalid multiline hardware property')
    widths = [max([len(column)] + [len(row[i]) for row in values]) + 2
              for i, column in enumerate(columns)]
    lines = [''.join(value.ljust(width) for value, width in zip(row, widths))
             for row in [list(columns)] + values]
    return '\r\n'.join(lines) + '\r\n'


def check_output(command, *, shell=True, text=False):
    """Narrow substitute at existing WMIC call sites, with the same return type."""
    if command not in QUERIES or shell is not True:
        raise ValueError('Unsupported legacy hardware command')
    if not sys.platform.startswith('win'):
        return subprocess.check_output(command, shell=shell, text=text)
    executable = _wmic_executable()
    if executable:
        try:
            return subprocess.check_output(
                [executable] + command.split()[1:], text=text,
                timeout=QUERY_TIMEOUT, stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
        except (OSError, subprocess.SubprocessError):
            # A present but failing WMIC is not silently replaced with a different identity.
            raise RuntimeError('Existing WMIC hardware query failed') from None
    class_name, columns = QUERIES[command]
    output = _legacy_table(_query_cim(class_name, columns), columns)
    return output if text else output.encode('utf-8')
