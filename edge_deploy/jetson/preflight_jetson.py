"""Read-only Jetson inventory; run with system python3 before choosing dependencies.

No sudo, installs, model loading, license/counter access or hardware identifiers.
Compatible with Python 3.6+; may be sent over SSH stdin without installing files.
"""
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def read_text(path):
    try:
        return Path(path).read_bytes().decode('utf-8', errors='replace').replace('\x00', '').strip()
    except OSError:
        return None


def run(args, timeout=15):
    if not shutil.which(args[0]):
        return {'available': False}
    try:
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, timeout=timeout)
        return {'available': True, 'returncode': result.returncode,
                'stdout': result.stdout.strip()[:16000], 'stderr': result.stderr.strip()[:2000]}
    except subprocess.TimeoutExpired:
        return {'available': True, 'error': 'timeout'}


def package_versions():
    try:
        from importlib import metadata
    except ImportError:
        try:
            import importlib_metadata as metadata
        except ImportError:
            return {'status': 'metadata_module_unavailable; use dpkg inventory'}
    versions = {}
    for name in ('torch', 'torchvision', 'onnxruntime', 'onnxruntime-gpu', 'tensorrt',
                 'numpy', 'transformers', 'ultralytics', 'Flask', 'gunicorn', 'pycuda'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect():
    mem = {}
    for line in (read_text('/proc/meminfo') or '').splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            mem[key] = value.strip()
    os_info = {}
    for line in (read_text('/etc/os-release') or '').splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            if key in ('ID', 'VERSION_ID', 'PRETTY_NAME'):
                os_info[key] = value.strip('"')
    return {
        'schema_version': 1, 'status': 'inventory_only_not_deployed',
        'machine': platform.machine(), 'kernel': platform.release(), 'os': os_info,
        'board_model': read_text('/proc/device-tree/model'),
        'l4t_release': read_text('/etc/nv_tegra_release'),
        'python': platform.python_version(), 'python_executable': sys.executable,
        'home': str(Path.home()), 'uid': os.getuid(), 'memory': mem,
        'disk': run(['df', '-hP', '/', str(Path.home())]),
        'jetpack_packages': run(['dpkg-query', '-W', '-f=${Package}\t${Version}\n',
              'nvidia-jetpack', 'nvidia-l4t-core', 'libnvinfer*', 'python3-libnvinfer*',
              'libcudnn*', 'cuda-cudart-*', 'cuda-nvcc-*']),
        'python_packages': package_versions(),
        'cuda_version': read_text('/usr/local/cuda/version.json') or read_text('/usr/local/cuda/version.txt'),
        'trtexec_paths': [p for p in ('/usr/src/tensorrt/bin/trtexec', '/usr/bin/trtexec') if Path(p).is_file()],
        # On older L4T even nvpmodel -q may attempt to set the fan profile.
        # Do not invoke it from a read-only inventory.
        'power_mode': {'status': 'not_queried; old nvpmodel may write fan settings'},
        'project_service': run(['systemctl', 'show', 'autoocr-business.service',
                               '-p', 'LoadState', '-p', 'ActiveState', '-p', 'SubState']),
        'target_port_5051': run(['ss', '-H', '-ltn', 'sport = :5051']),
        'note': 'Package inventory does not prove GPU execution. No serial, MAC, license or private model data collected.'
    }


if __name__ == '__main__':
    print(json.dumps(collect(), ensure_ascii=True, indent=2))
