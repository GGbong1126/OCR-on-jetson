"""Prepare a pinned JP4.6.1 ARM64 runtime bundle on the connected host.

Does not install anything or resolve arbitrary dependencies. Before using the
bundle, verify Release.gpg with the target's existing NVIDIA OTA public key,
then run apt-get --simulate on the target. Hash checks alone are not signatures.
Python 3.6+; uses curl for verified HTTPS downloads (including official redirects).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess


BASE = 'https://repo.download.nvidia.com/jetson/common/'
PINS = {
    'cuda-cudart-10-2': '10.2.300-1',
    'cuda-nvrtc-10-2': '10.2.300-1',
    'libcublas10': '10.2.3.300-1',
    'libcudnn8': '8.2.1.32-1+cuda10.2',
    'libnvinfer8': '8.2.1-1+cuda10.2',
    'libnvinfer-plugin8': '8.2.1-1+cuda10.2',
    'libnvparsers8': '8.2.1-1+cuda10.2',
    'libnvonnxparsers8': '8.2.1-1+cuda10.2',
    'libnvinfer-bin': '8.2.1-1+cuda10.2',
    'python3-libnvinfer': '8.2.1-1+cuda10.2',
}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def select_packages(metadata):
    release = (metadata / 'Release').read_text(encoding='utf-8')
    hashes = release.split('SHA256:\n', 1)[1]
    row = next(line.split() for line in hashes.splitlines()
               if line.endswith(' main/binary-arm64/Packages.gz'))
    index = metadata / 'Packages.gz'
    if index.stat().st_size != int(row[1]) or sha256(index) != row[0]:
        raise ValueError('Package index differs from Release SHA256/size')
    paragraphs = gzip.decompress(index.read_bytes()).decode('utf-8').split('\n\n')
    chosen = {}
    for paragraph in paragraphs:
        fields = dict(line.split(': ', 1) for line in paragraph.splitlines()
                      if line and not line[0].isspace() and ': ' in line)
        name = fields.get('Package')
        if name not in PINS or fields.get('Version') != PINS[name]:
            continue
        if fields.get('Architecture') != 'arm64' or name in chosen:
            raise ValueError('Unexpected architecture or duplicate: ' + name)
        filename = PurePosixPath(fields['Filename'])
        if (filename.is_absolute() or '..' in filename.parts
                or filename.parts[:2] != ('pool', 'main')
                or not filename.name.endswith('_arm64.deb')):
            raise ValueError('Unsafe repository filename')
        chosen[name] = {
            'package': name, 'version': fields['Version'], 'architecture': 'arm64',
            'filename': filename.name, 'url': BASE + filename.as_posix(),
            'bytes': int(fields['Size']), 'installed_kib': int(fields['Installed-Size']),
            'sha256': fields['SHA256'], 'depends': fields.get('Depends', ''),
        }
    if set(chosen) != set(PINS):
        raise ValueError('Missing pinned packages: ' + str(sorted(set(PINS) - set(chosen))))
    return [chosen[name] for name in PINS]


def download(row, directory):
    path = directory / row['filename']
    if not path.exists():
        part = path.with_suffix('.deb.part')
        subprocess.run([
            'curl', '--fail', '--location', '--silent', '--show-error',
            '--proto', '=https', '--proto-redir', '=https',
            '--connect-timeout', '15', '--max-time', '600', '--retry', '2',
            '--output', str(part), row['url'],
        ], check=True)
        if part.stat().st_size != row['bytes'] or sha256(part) != row['sha256']:
            raise ValueError('Downloaded package hash/size mismatch: ' + row['package'])
        part.rename(path)
    if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
        raise ValueError('Existing package hash/size mismatch: ' + row['package'])
    print('Verified ' + row['package'], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', type=Path, required=True)
    ap.add_argument('--download', action='store_true')
    args = ap.parse_args()
    root = args.directory.resolve()
    rows = select_packages(root / 'metadata')
    total = sum(row['bytes'] for row in rows)
    installed = sum(row['installed_kib'] for row in rows) * 1024
    print(json.dumps({'package_count': len(rows), 'download_bytes': total,
                      'installed_bytes_metadata': installed, 'packages': rows}, indent=2))
    if not args.download:
        return
    if shutil.disk_usage(str(root)).free < total + 512 * 1024**2:
        raise ValueError('Insufficient host space including 512 MiB reserve')
    packages = root / 'packages'
    packages.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda row: download(row, packages), rows))
    manifest = root / 'bundle.json'
    with manifest.open('x', encoding='utf-8') as f:
        json.dump({'status': 'hash_verified_not_installed', 'packages': rows,
                   'download_bytes': total, 'installed_bytes_metadata': installed,
                   'note': 'Requires external signature verification and target apt simulation.'}, f, indent=2)
    with (root / 'SHA256SUMS').open('x', encoding='ascii') as f:
        for row in rows:
            f.write(row['sha256'] + '  packages/' + row['filename'] + '\n')
    print('Bundle prepared; no software installed. ' + str(manifest), flush=True)


if __name__ == '__main__':
    main()
