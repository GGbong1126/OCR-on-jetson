"""Allowlisted source/model transfer; no environment, license or counter files.

Build on the development host. Install into a NEW board directory, verifying all
hashes before exposing it to a service. This is not a deployed release marker.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def fingerprint(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def safe_name(name):
    p = PurePosixPath(name)
    return (bool(name) and not p.is_absolute() and '..' not in p.parts
            and '\\' not in name and ':' not in name and str(p) == name)


def build(source, output):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'business_release', source / 'edge_deploy/business/release_business.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = module.collect(source)
    manifest = {'kind': 'jetson_source_models_not_deployed', 'files': files}
    blob = json.dumps(manifest, indent=2).encode('utf-8')
    with output.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz', compresslevel=1) as tar:
        for name, expected in sorted(files.items()):
            path = module.plain(source / name)
            if fingerprint(path) != expected:
                raise ValueError('Source changed: ' + name)
            info = tar.gettarinfo(str(path), arcname=name)
            info.mode, info.uid, info.gid, info.mtime = 0o600, 0, 0, 0
            info.uname = info.gname = ''
            with path.open('rb') as f:
                tar.addfile(info, f)
        info = tarfile.TarInfo('jetson_source_manifest.json')
        info.size, info.mode = len(blob), 0o600
        tar.addfile(info, io.BytesIO(blob))
    print(json.dumps({'archive': fingerprint(output), 'files': len(files),
                      'payload_bytes': sum(r['bytes'] for r in files.values())}))


def install(archive, output, expected_sha):
    if fingerprint(archive)['sha256'] != expected_sha:
        raise ValueError('Archive hash mismatch')
    if output.exists() or output.is_symlink():
        raise ValueError('Install destination must not exist')
    for parent in output.parents:
        if parent.is_symlink():
            raise ValueError('Linked destination parent')
    with tarfile.open(str(archive), 'r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() or not safe_name(m.name) for m in members):
            raise ValueError('Unsafe or duplicate archive entry')
        manifest = json.load(tar.extractfile('jetson_source_manifest.json'))
        files = manifest['files']
        if (manifest['kind'] != 'jetson_source_models_not_deployed'
                or set(names) != set(files) | {'jetson_source_manifest.json'}):
            raise ValueError('Manifest mismatch')
        needed = sum(m.size for m in members)
        if shutil.disk_usage(str(output.parent)).free < needed + 1024**3:
            raise ValueError('Keep at least 1GiB free after extraction')
        output.mkdir(mode=0o700)
        for member in members:
            target = output / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            target.chmod(0o600)
            if member.name in files and fingerprint(target) != files[member.name]:
                raise ValueError('Extracted hash mismatch: ' + member.name)
    print(json.dumps({'status': 'source_verified_not_deployed', 'files': len(files),
                      'directory': str(output)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('build', 'install'))
    ap.add_argument('--source', type=Path)
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--destination', type=Path)
    ap.add_argument('--sha256')
    args = ap.parse_args()
    if args.mode == 'build':
        if args.source is None:
            ap.error('--source is required')
        build(args.source.resolve(), args.archive)
    else:
        if args.destination is None or args.sha256 is None:
            ap.error('--destination and --sha256 are required')
        install(args.archive, args.destination.absolute(), args.sha256)


if __name__ == '__main__':
    main()
