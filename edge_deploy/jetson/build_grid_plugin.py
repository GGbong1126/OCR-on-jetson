"""Build only the project-local SM72 plugin using already extracted CUDA10.2."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/home/xchq/work/autoocr'))
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if platform.machine() != 'aarch64' or a.output.exists():
        p.error('Requires aarch64 and a new output path')
    cuda = a.root / 'build/cuda-headers/usr/local/cuda-10.2'
    trt = a.root / 'build/trt82-source/TensorRT-6f38570b74066ef464744bc789f8512191f1cbc0'
    command = [str(cuda / 'bin/nvcc'), '-std=c++14', '-O2', '--shared',
               '-Xcompiler', '-fPIC', '--fmad=false', '-gencode', 'arch=compute_72,code=sm_72',
               '-I'+str(cuda/'targets/aarch64-linux/include'), '-I'+str(trt/'include'),
               str(a.source), '--cudart', 'none',
               '-Xlinker', '/usr/local/cuda-10.2/targets/aarch64-linux/lib/libcudart.so.10.2',
               '-Xlinker', '/usr/lib/aarch64-linux-gnu/libnvinfer.so.8', '-o', str(a.output)]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    report = {'command': command, 'source_sha256': hashlib.sha256(a.source.read_bytes()).hexdigest(),
              'plugin_sha256': hashlib.sha256(a.output.read_bytes()).hexdigest()}
    with a.output.with_suffix('.build.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
