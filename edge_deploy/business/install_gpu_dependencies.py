"""Finish a NEW GPU venv using a reviewed baseline package inventory.

First install torch==2.8.0/torchvision==0.23.0 from the official cu126 index.
Never run in the current CPU business venv. No models, license or service access.
"""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-environment', type=Path, required=True)
    p.add_argument('--trt-wheel', type=Path, required=True)
    args = p.parse_args()
    if metadata.version('torch') != '2.8.0+cu126' or metadata.version('torchvision') != '0.23.0+cu126':
        raise RuntimeError('Only run in the newly created cu126 GPU venv')
    packages = json.loads(args.baseline_environment.read_text())['packages']
    exclude = {'torch', 'torchvision', 'onnxruntime', 'tensorrt', 'pip', 'setuptools'}
    req = [name + '==' + ver for name, ver in packages.items() if name.lower() not in exclude]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple',
                    *req, 'onnxruntime-gpu==1.19.2'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                    str(args.trt_wheel)], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)


if __name__ == '__main__':
    main()
