"""Build official TRT8.2.1 Python bindings against the existing board libraries.

Application directory only; no sudo, system Python/library replacement or nvcc.
The OSS tag is 8.2.1.8, the installed Jetson package is 8.2.1 build 9. Record both
instead of relabeling the locally built wheel as NVIDIA's binary distribution.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig


COMMIT = '6f38570b74066ef464744bc789f8512191f1cbc0'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    import pybind11
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--attempt', type=int, default=2)
    args = ap.parse_args()
    root = args.root.resolve()
    if (platform.machine() != 'aarch64' or sys.version_info[:2] != (3, 10)
            or root != Path('/home/xchq/work/autoocr')):
        raise RuntimeError('Expected isolated Python3.10 on the selected board root')
    lib = ctypes.CDLL('libnvinfer.so.8')
    lib.getInferLibVersion.restype = ctypes.c_int
    if lib.getInferLibVersion() != 8201:
        raise RuntimeError('Expected installed TensorRT runtime 8.2.1')
    src = root / 'build/trt82-source' / ('TensorRT-' + COMMIT)
    header = (src / 'include/NvInferVersion.h').read_text()
    for name, number in [('MAJOR', 8), ('MINOR', 2), ('PATCH', 1), ('BUILD', 8)]:
        if '#define NV_TENSORRT_{} {}'.format(name, number) not in header:
            raise RuntimeError('Unexpected source header version')
    if args.attempt < 2:
        raise ValueError('Preserve the initial failed configure; use attempt >= 2')
    build = root / ('build/trt82-cp310-r' + str(args.attempt))
    if build.exists():
        raise RuntimeError('Build directory exists; inspect before retry')
    build.mkdir()
    parser_include = build / 'parser-include'
    (parser_include / 'onnx').mkdir(parents=True)
    shutil.copyfile(root / 'staging/NvOnnxParser.h', parser_include / 'onnx/NvOnnxParser.h')
    links = build / 'runtime-libs'
    links.mkdir()
    library_hashes = {}
    for name in ('nvinfer', 'nvonnxparser', 'nvparsers', 'nvinfer_plugin'):
        real = (Path('/usr/lib/aarch64-linux-gnu') / ('lib' + name + '.so.8')).resolve(strict=True)
        library_hashes[real.name] = sha256(real)
        (links / ('lib' + name + '.so')).symlink_to(real)
    include = sysconfig.get_path('include')
    cmake = str(Path(sys.executable).parent / 'cmake')
    subprocess.run([cmake, '-S', str(src / 'python'), '-B', str(build),
        '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_FLAGS_RELEASE=-O2 -DNDEBUG',
        '-DVERBOSE=1', '-DTARGET=aarch64', '-DNV_TARGET_OS=linux',
        '-DEXT_PATH=' + str(root / 'build'),
        '-DPYTHON_MAJOR_VERSION=3', '-DPYTHON_MINOR_VERSION=10',
        '-DPY_INCLUDE=' + include, '-DPY_CONFIG_INCLUDE=' + include,
        '-DPYBIND11_DIR=' + pybind11.get_include(), '-DTENSORRT_ROOT=' + str(src),
        '-DTENSORRT_BUILD=' + str(links), '-DONNX_INC_DIR=' + str(parser_include),
        '-DCUDA_INCLUDE_DIRS=' + str(root / 'build/cuda-headers/usr/local/cuda-10.2/targets/aarch64-linux/include')], check=True)
    subprocess.run([cmake, '--build', str(build), '--parallel', '2'], check=True)
    version = '8.2.1.8+localcp310'
    for path in (src / 'python/packaging').rglob('*'):
        if path.is_file():
            target = build / path.relative_to(src / 'python/packaging')
            target.parent.mkdir(parents=True, exist_ok=True)
            content = path.read_text().replace('##TENSORRT_VERSION##', version).replace('##TENSORRT_MAJMINPATCH##', '8.2.1')
            # NumPy removed np.bool in 1.24; equivalent scalar dtype, no kernel change.
            content = content.replace('bool: np.bool,', 'bool: np.bool_,')
            target.write_text(content)
    subprocess.run([sys.executable, 'setup.py', '-q', 'bdist_wheel',
                    '--python-tag=cp310', '--plat-name=linux_aarch64'], cwd=str(build), check=True)
    wheels = list((build / 'dist').glob('*.whl'))
    if len(wheels) != 1:
        raise RuntimeError('Expected one wheel')
    report = {'source_commit': COMMIT, 'source_version': '8.2.1.8',
              'local_binding_version': version, 'installed_runtime_api_version': 8201,
              'installed_runtime_package': '8.2.1-1+cuda10.2',
              'python': platform.python_version(), 'pybind11': pybind11.__version__,
              'runtime_libraries_sha256': library_hashes,
              'wheel': str(wheels[0]), 'wheel_sha256': sha256(wheels[0]),
              'note': 'Local Python ABI rebuild, not a system TensorRT upgrade. GPU tests still required.'}
    with (root / ('reports/trt82_cp310_build_r' + str(args.attempt) + '.json')).open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
