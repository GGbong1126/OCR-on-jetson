"""Allowlisted CPU or native-TRT OCR business release. Never activate/touch counters.

Build a new directory; verify exact files and hashes; optionally archive it.
The environment snapshot records installed versions, not an offline dependency lock.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import tarfile


NER = 'pth/ner/ner_model_cadre_docs_continuous_v3/ner_model_cadre_docs_continuous_v3/'
BUSINESS = 'edge_deploy/business/'
SOURCE_FILES = (
    'server.py', 'server_config.py', 'windows_hardware.py', 'ner_postprocess.py',
    'words.py', 'train_mobilenetv3_321.py',
    'layout/__init__.py', 'layout/layout_fusion.py', 'layout/yolo_compat.py',
    'layout/yolo_layers/__init__.py', 'layout/yolo_layers/bifpn.py',
    'layout/paddlepico_detector.py', 'layout/table_cell_postprocessor.py',
    'layout/structure_analyzer.py',
) + tuple(BUSINESS + x for x in (
    'serve_business.py', 'start.sh', 'service_health.py', 'smoke_business.py',
    'acceptance_business.py', 'compare_business.py', 'requirements.txt',
    'release_business.py', 'start_release.sh', 'RELEASE.md', 'API_USAGE.md',
)) + tuple('pth/onnx_rec/' + x for x in (
    'layout_det.onnx', 'cell_det.onnx', 'ocr_det.onnx', 'ocr_rec.onnx', 'char_dict.json',
)) + tuple(NER + x for x in (
    'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
    'special_tokens_map.json', 'vocab.txt',
)) + ('pth/rec/Det/Det/det_photo/best1.pt',)
OPTIONAL = 'pth/ocr/server/rec/inference.yml'
MANIFEST = 'release_manifest.json'
ENVIRONMENT = 'deployment_environment.json'
TRT_CONFIG = BUSINESS + 'trt_models.json'
TRT_ENGINES = ('pth/tensorrt/ocr_det_fp32_h960_w672_736_r1.engine',
               'pth/tensorrt/ocr_rec_fp32_w32_1536_r1.engine')
TRT_FILES = (BUSINESS + 'trt_sessions.py', BUSINESS + 'GPU_DEPLOYMENT.md', TRT_CONFIG,
             'edge_deploy/tools/trt_backend.py', 'edge_deploy/tools/ocr_det.py', *TRT_ENGINES)
GPU_FILES = (BUSINESS + 'gpu_models.py',)


def plain(path):
    """Reject symlinks/junctions in all existing path components, including Windows."""
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if not part.exists() and not part.is_symlink():
            continue
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Linked/reparse paths are not allowed')
    return path


def fingerprint(path):
    path = plain(path)
    if not path.is_file():
        raise ValueError('Required regular file missing: ' + path.name)
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            size += len(block)
            digest.update(block)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def checked_environment(value):
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'platform', 'machine', 'python', 'packages'}
            or value['schema_version'] != 1 or value['platform'] != 'Linux'
            or value['machine'] != 'x86_64' or not re.fullmatch(r'3\.10\.\d+', value['python'])):
        raise ValueError('Expected the verified Linux x86_64 Python 3.10 environment')
    packages = value['packages']
    if not isinstance(packages, dict) or not packages:
        raise ValueError('Package inventory required')
    for name, version in packages.items():
        if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name)
                or not isinstance(version, str) or not re.fullmatch(r'[A-Za-z0-9.!+_-]+', version)):
            raise ValueError('Invalid package/version; URLs and paths are not recorded')
    return value


def capture_environment(output):
    value = {'schema_version': 1, 'platform': platform.system(), 'machine': platform.machine(),
             'python': platform.python_version(),
             'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))}
    write_new(plain(output), checked_environment(value))
    return {'status': 'environment_inventory_not_lockfile', 'package_count': len(value['packages'])}


def collect(root):
    root = plain(root)
    names = list(SOURCE_FILES)
    # Match server.load_models' case-insensitive suffix check, including mixed .pth/.PTH.
    weights = [p for p in (root / 'pth/imgpre').iterdir() if p.name.lower().endswith('.pth')]
    if len(weights) != 1:
        raise ValueError('Expected exactly one enhancement .pth; do not silently choose a new model')
    names.append(weights[0].relative_to(root).as_posix())
    if (root / OPTIONAL).exists():
        names.append(OPTIONAL)  # Preserve the native YAML-before-JSON dictionary choice.
    return {name: fingerprint(root / name) for name in sorted(names)}


def check_trt_assets(root, files):
    spec = json.loads(plain(root / TRT_CONFIG).read_text(encoding='utf-8'))
    if (spec.get('schema_version') != 1 or spec.get('precision') != 'fp32_no_tf32'
            or spec.get('tensorrt_version') != '10.7.0' or set(spec['models']) != {'det', 'rec'}):
        raise ValueError('Unsupported TRT configuration')
    for index, role in enumerate(('det', 'rec')):
        row = spec['models'][role]
        if row['engine'] != TRT_ENGINES[index] or row['onnx'] != 'pth/onnx_rec/ocr_' + role + '.onnx':
            raise ValueError('Unexpected TRT asset path')
        for key in ('engine', 'onnx'):
            if files[row[key]]['sha256'] != row[key + '_sha256']:
                raise ValueError('TRT engine/source hash differs from approved build: ' + role)


def check_gpu_environment(env):
    packages = env['packages']
    if (packages.get('torch') != '2.8.0+cu126' or packages.get('torchvision') != '0.23.0+cu126'
            or packages.get('onnxruntime-gpu') != '1.19.2' or 'onnxruntime' in packages):
        raise ValueError('Expected isolated cu126 PyTorch and ORT GPU environment')


def build(source, output, release_id, environment, trt_engine_dir=None, gpu_models=False):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', release_id):
        raise ValueError('Use a simple release ID, not a path')
    source, output = plain(source), plain(output)
    if output.exists() or source == output or source.is_relative_to(output):
        raise ValueError('Output must be a new directory, not a source ancestor')
    env = checked_environment(json.loads(Path(environment).read_text(encoding='utf-8')))
    files = collect(source)
    sources = {name: source / name for name in files}
    if gpu_models:
        if trt_engine_dir is None:
            raise ValueError('GPU models require TRT engines')
        check_gpu_environment(env)
        for name in GPU_FILES:
            sources[name] = source / name
            files[name] = fingerprint(sources[name])
    if trt_engine_dir is not None:
        engine_dir = plain(trt_engine_dir)
        for name in TRT_FILES:
            sources[name] = engine_dir / Path(name).name if name in TRT_ENGINES else source / name
            files[name] = fingerprint(sources[name])
        check_trt_assets(source, files)
        if env['packages'].get('tensorrt') != '10.7.0':
            raise ValueError('TRT release requires the verified TensorRT 10.7.0 environment')
    # No recursive copy: excluded files are never read, even when they are under model directories.
    output.mkdir(parents=True, exist_ok=False)
    for name, expected in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with plain(sources[name]).open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        if fingerprint(target) != expected:
            raise ValueError('Source changed while copying; incomplete directory retained')
    write_new(output / ENVIRONMENT, env)
    files[ENVIRONMENT] = fingerprint(output / ENVIRONMENT)
    kind = 'trt_business_source_models' if trt_engine_dir is not None else 'cpu_business_source_models'
    if gpu_models:
        kind = 'gpu_business_source_models'
    manifest = {'schema_version': 1, 'release_id': release_id, 'kind': kind,
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'files': files,
                'note': 'Internal source/model bundle, no venv/wheels or machine runtime state. '
                        'Hashes detect drift, not provenance/authenticity. Not a deployed/accuracy-validated release.'}
    write_new(output / MANIFEST, manifest)  # Completion marker written only after all payload hashes pass.
    return verify(output)


def verify(root):
    root = plain(root)
    plain(root / MANIFEST)
    manifest = json.loads((root / MANIFEST).read_text(encoding='utf-8'))
    if (manifest.get('schema_version') != 1 or manifest.get('kind') not in ('cpu_business_source_models', 'trt_business_source_models', 'gpu_business_source_models')
            or not isinstance(manifest.get('release_id'), str)
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', manifest['release_id'])):
        raise ValueError('Invalid manifest identity')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Manifest files missing')
    fixed = set(SOURCE_FILES) | {ENVIRONMENT}
    if manifest['kind'] in ('trt_business_source_models', 'gpu_business_source_models'):
        fixed |= set(TRT_FILES)
    if manifest['kind'] == 'gpu_business_source_models':
        fixed |= set(GPU_FILES)
    if not fixed <= set(files):
        raise ValueError('Missing required release entries')
    weights = []
    for name in files:
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or str(path) != name:
            raise ValueError('Unsafe manifest path')
        if name not in fixed | {OPTIONAL}:
            if path.parent.as_posix() != 'pth/imgpre' or path.suffix.lower() != '.pth':
                raise ValueError('Unexpected file in release manifest')
            weights.append(name)
    if len(weights) != 1 or len({name.casefold() for name in files}) != len(files):
        raise ValueError('Ambiguous enhancement weights or filename collision')
    actual = set()
    allowed_dirs = {str(p) for name in set(files) | {MANIFEST} for p in PurePosixPath(name).parents if str(p) != '.'}
    for current, directories, filenames in os.walk(root, followlinks=False):
        for name in directories + filenames:
            path = plain(Path(current) / name)
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                if relative not in allowed_dirs:
                    raise ValueError('Unexpected directory in release: ' + relative)
            else:
                if not path.is_file():
                    raise ValueError('Non-regular release entry')
                actual.add(relative)
    if actual != set(files) | {MANIFEST}:
        raise ValueError('Release contains extra or missing files')
    for name, expected in files.items():
        if fingerprint(root / name) != expected:
            raise ValueError('Hash/size mismatch: ' + name)
    env = checked_environment(json.loads((root / ENVIRONMENT).read_text(encoding='utf-8')))
    if manifest['kind'] == 'gpu_business_source_models':
        check_gpu_environment(env)
    if manifest['kind'] in ('trt_business_source_models', 'gpu_business_source_models'):
        check_trt_assets(root, files)
        if env['packages'].get('tensorrt') != '10.7.0':
            raise ValueError('Missing pinned TensorRT environment record')
    return {'status': 'verified_files_not_deployed', 'release_id': manifest['release_id'],
            'file_count': len(files), 'payload_bytes': sum(row['bytes'] for row in files.values()),
            'manifest_sha256': fingerprint(root / MANIFEST)['sha256']}


def archive(root, output):
    root, output = plain(root), plain(output)
    verified = verify(root)
    if output.exists() or output.is_relative_to(root):
        raise ValueError('Archive must be new and outside release directory')
    manifest = json.loads((root / MANIFEST).read_text(encoding='utf-8'))
    with output.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz', compresslevel=1) as tar:
        for name in sorted(set(manifest['files']) | {MANIFEST}):
            path = plain(root / name)
            expected = manifest['files'].get(name, fingerprint(path))
            if fingerprint(path) != expected:
                raise ValueError('Release changed before archiving; archive incomplete')
            info = tar.gettarinfo(str(path), arcname=verified['release_id'] + '/' + name)
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mode, info.mtime = 0o600, 0
            with path.open('rb') as content:
                tar.addfile(info, content)
    return {'status': 'archived_not_deployed', **fingerprint(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='command', required=True)
    env = modes.add_parser('environment'); env.add_argument('--output', type=Path, required=True)
    make = modes.add_parser('build')
    for name in ('source', 'output', 'environment'):
        make.add_argument('--' + name, type=Path, required=True)
    make.add_argument('--release-id', required=True)
    make.add_argument('--trt-engine-dir', type=Path, help='Copy only the two approved FP32 engines into a new TRT release')
    make.add_argument('--gpu-models', action='store_true', help='Include validated non-OCR CUDA adapter')
    check = modes.add_parser('verify'); check.add_argument('--root', type=Path, required=True)
    pack = modes.add_parser('archive')
    pack.add_argument('--root', type=Path, required=True); pack.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'environment':
        result = capture_environment(args.output)
    elif args.command == 'build':
        result = build(args.source, args.output, args.release_id, args.environment, args.trt_engine_dir, args.gpu_models)
    elif args.command == 'verify':
        result = verify(args.root)
    else:
        result = archive(args.root, args.output)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
