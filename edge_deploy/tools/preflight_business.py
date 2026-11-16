"""Read-only full-server migration inventory. Never import server or expose license/device IDs."""
import argparse
import ast
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


PACKAGES = ('numpy', 'opencv-python', 'opencv-contrib-python', 'opencv-python-headless', 'onnxruntime', 'torch',
            'torchvision', 'Flask', 'transformers', 'paddleocr', 'paddlex',
            'paddlepaddle', 'paddlepaddle-gpu', 'ultralytics', 'pycryptodome',
            'pyclipper', 'PyYAML', 'Pillow', 'gunicorn')


def disk_probe(source):
    """Run only the server's read-only disk query; discard the identifier immediately."""
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_get_disk_serial')
    # Reject changed implementations containing calls other than this reviewed reader's operations.
    allowed = {'startswith', 'check_output', 'splitlines', 'strip', 'len', 'RuntimeError', 'Path',
               'iterdir', 'exists', 'sorted', 'read_text', 'upper', 'run'}
    for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
        name = call.func.id if isinstance(call.func, ast.Name) else call.func.attr if isinstance(call.func, ast.Attribute) else ''
        if name not in allowed:
            raise ValueError('Disk reader changed; inspect before running')
    namespace = {'sys': sys, 'subprocess': subprocess, 'Path': Path}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<disk-probe>', 'exec'), namespace)
    try:
        value = namespace['_get_disk_serial']()
        return {'readable': bool(value), 'identifier_not_saved': True}
    except Exception as error:
        return {'readable': False, 'error_type': type(error).__name__, 'identifier_not_saved': True}


def inspect(root, probe_disk=False):
    root = root.resolve()
    source = (root / 'server.py').read_text(encoding='utf-8-sig')
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    assets = {}
    names = ['server.py', 'server_config.py', 'windows_hardware.py', 'ner_postprocess.py', 'words.py',
             'train_mobilenetv3_321.py', 'layout/paddlepico_detector.py',
             'layout/table_cell_postprocessor.py', 'layout/layout_fusion.py']
    names += ['pth/onnx_rec/' + x for x in ('layout_det.onnx', 'cell_det.onnx', 'ocr_det.onnx', 'ocr_rec.onnx', 'char_dict.json')]
    names += ['pth/ner/ner_model_cadre_docs_continuous_v3/ner_model_cadre_docs_continuous_v3/' + x
              for x in ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt')]
    names += ['pth/rec/Det/Det/det_photo/best1.pt']
    for name in names:
        path = root / name
        assets[name] = {'exists': path.is_file(), 'bytes': path.stat().st_size if path.is_file() else None}
    imgpre = root / 'pth/imgpre'
    missing = [name for name, info in assets.items() if not info['exists']]
    memory = None
    meminfo = Path('/proc/meminfo')
    if meminfo.is_file():
        memory = {line.split(':')[0]: line.split(':')[1].strip()
                  for line in meminfo.read_text().splitlines()
                  if line.split(':')[0] in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree')}
    malformed = []
    requirements = root / 'requirements.txt'
    if requirements.is_file():
        # A named requirement without == is not always invalid; report suspect version-appended entries only.
        for line in requirements.read_text(encoding='utf-8-sig').splitlines():
            value = line.strip()
            if value and not value.startswith('#') and '==' not in value:
                malformed.append(value)
    return {'status': 'inventory_only_not_deployed', 'platform': platform.system(),
            'machine': platform.machine(), 'python': platform.python_version(), 'executable': sys.executable,
            'source_root': str(root), 'packages': versions, 'assets': assets, 'missing_assets': missing,
            'imgpre_weight_count': len(list(imgpre.glob('*.pth'))) if imgpre.is_dir() else 0,
            'tkinter_module_available': importlib.util.find_spec('tkinter') is not None,
            'display_environment_present': bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')),
            'source_license_file_present': (root / 'license.dat').is_file(),
            'disk_serial_probe': disk_probe(source) if probe_disk else {'not_run': True},
            'memory': memory, 'home_free_bytes': shutil.disk_usage(Path.home()).free,
            'requirements_suspect_lines': malformed,
            'note': 'No server import, model load, package installation, activation, identifiers, license contents or file writes. '
                    'Default model tree checked; environment/CLI path overrides not resolved. Package presence is not import compatibility. '
                    'No claim that a Windows license works in WSL.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--probe-disk', action='store_true', help='Linux only: run the reviewed read-only disk serial query without printing identifiers')
    args = parser.parse_args()
    if args.probe_disk and not sys.platform.startswith('linux'):
        parser.error('--probe-disk is restricted to Linux')
    print(json.dumps(inspect(args.source, args.probe_disk), ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
