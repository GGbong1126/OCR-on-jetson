"""Copy explicitly selected Jetson evidence; never move, delete or unpack it."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat


SELECTION = {
    'jetson_jp461_r1': 'bootstrap, offline dependencies, model trials and build evidence',
    'jetson_codex_deploy': 'board development tooling and deployment support',
    'jetson_codex_deploy.tar.gz': 'board development tooling payload',
    'batch_ocr_e143_r1': 'batch OCR correctness and baseline evidence',
    'batch_ocr_e144_r1': 'batch OCR geometry and GPU execution evidence',
    'kmalloc_e147_r1': 'board kernel allocation investigation',
    'display_memory_e150_r1': 'board display event memory investigation',
    'det_batch_e151_recovered': 'recovered DET batch build and scripts',
    'det_batch_e153_r1': 'DET batch and cross-image validation',
    'ipc_e154_r1': 'C++ and Python IPC probe',
    'scheduler_e155_r1': 'bounded scheduler and credit reservation',
    'lifecycle_e156_r1': 'registry lifecycle and failure evidence',
    'json_ipc_e157_r1': 'JSON schema, parser and integration evidence',
    'channels_e158_r1': 'four-channel I/O integration',
    'workers_e159_r1': 'CPU/GPU worker bridge and recovered sources',
    'ocr_workers_e160_r1': 'full OCR stage bridge and source snapshots',
    'cpp_app_e161_r1': 'C++ batch application, hardware JPEG and full source snapshots',
    'batch_defaults_e162_r1': 'hardware JPEG and DET N2 defaults and rollback evidence',
    'multistream_e163_r1': 'Plan2 E163/E164 implementation, validation and profiling',
    'ablation_e165_r1': 'throughput ablations and frozen controls',
    'rec_concurrency_e166_r1': 'REC physical kernel concurrency experiments',
    'threaded_e167_r1': 'production OCR owner threads, throughput selection and acceptance',
    'http_rec_trace_e168_r2': 'actual HTTP REC kernel trace and final verification',
    'http_rec_trace_e168_r1_payload.tgz': 'initial trace payload, including failed attempt history',
    'http_rec_trace_e168_r2_payload.tgz': 'intermediate trace payload',
    'http_rec_trace_e168_r2b_payload.tgz': 'executed trace payload',
    'evaluate_ner_mixed_e115_executed.py': 'executed TRT8.2 NER mixed precision evaluator',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def regular(path):
    info = path.lstat()
    if path.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400:
        raise ValueError('Refusing symlink/reparse point: ' + str(path))
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise ValueError('Refusing special file: ' + str(path))
    return info


def scan(path):
    regular(path)
    yield path
    if path.is_dir():
        for child in sorted(path.iterdir(), key=lambda p: p.name):
            yield from scan(child)


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def verify(destination, source):
    manifest = json.loads((destination / 'manifest.json').read_text(encoding='utf-8'))
    expected = {row['path']: row for row in manifest['files']}
    payload = destination / 'payload' / 'temp'
    actual = {p.relative_to(payload).as_posix() for p in scan(payload) if p.is_file()}
    if actual != set(expected):
        raise ValueError('Archive has extra or missing files')
    for name, row in expected.items():
        for path in (source / name, payload / name):
            regular(path)
            if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
                raise ValueError('Source/archive hash mismatch: ' + name)
    return {'status': 'verified_source_and_copy', 'files': len(expected),
            'bytes': sum(row['bytes'] for row in expected.values()),
            'manifest_sha256': digest(destination / 'manifest.json'),
            'source_preserved': True, 'board_contacted': False}


def archive(source, destination, selection=SELECTION):
    regular(source)
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be disjoint')
    if destination.exists():
        raise FileExistsError('Use a new archive directory')
    for parent in destination.parents:
        if parent.exists():
            regular(parent)
    items = []
    for name in selection:
        if name in ('.', '..') or any(c in name for c in '/\\:'):
            raise ValueError('Selection must contain plain top-level names')
        items.extend(scan(source / name))
    total = sum(p.stat().st_size for p in items if p.is_file())
    existing_parent = next(p for p in destination.parents if p.exists())
    if shutil.disk_usage(existing_parent).free < total + 256 * 1024 * 1024:
        raise ValueError('Not enough space for a verified copy')
    destination.mkdir(parents=True, exist_ok=False)
    payload = destination / 'payload' / 'temp'
    payload.mkdir(parents=True)
    rows, copied, last_progress = [], 0, 0
    print('Copying selected entries: %d; bytes: %d' % (len(selection), total), flush=True)
    for path in items:
        relative = path.relative_to(source)
        target = payload / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        before = regular(path)
        h = hashlib.sha256()
        with path.open('rb') as src, target.open('xb') as dst:
            for chunk in iter(lambda: src.read(1024 * 1024), b''):
                dst.write(chunk)
                h.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Source changed during copy: ' + str(relative))
        shutil.copystat(path, target)
        rows.append({'path': relative.as_posix(), 'bytes': before.st_size,
                     'sha256': h.hexdigest(), 'source_mtime_ns': before.st_mtime_ns})
        copied += before.st_size
        if copied - last_progress >= 512 * 1024 * 1024:
            print('Copied bytes:', copied, flush=True)
            last_progress = copied
    excluded = [{'name': p.name, 'reason': 'Not selected as board-specific evidence; retained at source'}
                for p in sorted(source.iterdir(), key=lambda p: p.name) if p.name not in selection]
    write_json(destination / 'selection.json', {
        'included': [{'name': n, 'reason': reason} for n, reason in selection.items()],
        'excluded': excluded,
        'note': 'Only a copy of selected temp entries, not a complete board backup or install bundle.'})
    write_json(destination / 'manifest.json', {
        'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_root': str(source), 'files': rows,
        'note': 'Original bytes retained, including failed/intermediate experiments. No extraction or execution.'})
    print('Verifying source and destination hashes...', flush=True)
    result = verify(destination, source)
    write_json(destination / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    archive(args.source, args.destination)


if __name__ == '__main__':
    main()
