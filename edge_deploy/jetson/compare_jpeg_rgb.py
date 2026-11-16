"""M1a RGB/trace diagnostics only; never load models or call Extract."""
import argparse
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
import baseline_batch_ocr as baseline


def metrics(reference, candidate):
    if (reference.dtype != np.uint8 or candidate.dtype != np.uint8 or reference.shape != candidate.shape
            or reference.ndim != 3 or reference.shape[2] != 3 or not reference.size):
        raise ValueError('Matching nonempty RGB uint8 arrays required')
    total = equal = maximum = 0
    # Keep diagnostic memory bounded even on the low-memory board.
    for start in range(0, reference.shape[0], 128):
        diff = np.abs(reference[start:start + 128].astype(np.int16)
                      - candidate[start:start + 128].astype(np.int16))
        total += int(diff.sum(dtype=np.int64))
        equal += int((diff == 0).sum())
        maximum = max(maximum, int(diff.max()))
    return dict(mae=total / reference.size, max_abs_error=maximum,
                identical_components_fraction=equal / reference.size)


def device_calls(text, *, include_commands=False):
    fds, nodes = {}, {}
    for line in text.splitlines():
        if '<unfinished ...>' in line or 'resumed>' in line:
            raise ValueError('Interleaved trace needs explicit reassembly')
        opened = re.search(r'open(?:at)?\(.*?"([^"\n]+)".*?\)\s+=\s+(\d+)', line)
        if opened:
            path, fd = opened.group(1), int(opened.group(2))
            fds[fd] = path
            if path.startswith('/dev/'):
                nodes.setdefault(path, dict(opens=0, ioctl_success=0, ioctl_failure=0))['opens'] += 1
        invoked = re.search(r'ioctl\((0x[\da-fA-F]+|\d+),.*\)\s+=\s+(-?\d+)', line)
        if invoked:
            path = fds.get(int(invoked.group(1), 0), '')
            if path in nodes:
                key = 'ioctl_success' if int(invoked.group(2)) >= 0 else 'ioctl_failure'
                nodes[path][key] += 1
                if include_commands and int(invoked.group(2))>=0:
                    command=re.search(r'ioctl\([^,]+,\s*(0x[\da-fA-F]+|\d+),',line)
                    if command:
                        code=hex(int(command.group(1),0)&0xffffffff)
                        commands=nodes[path].setdefault('successful_commands',{})
                        commands[code]=commands.get(code,0)+1
        closed = re.search(r'close\((\d+)(?:<[^>]*>)?\)\s+=\s+0', line)
        if closed:
            fds.pop(int(closed.group(1)), None)
    return nodes


def memory_guard():
    rows = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(rows['MemAvailable'].split()[0])
    if available < 384 * 1024:
        raise RuntimeError('Less than 384 MiB available; no extra image diagnostic started')
    return available


def select_cases():
    from PIL import Image
    from PIL.JpegImagePlugin import get_sampling
    rows = []
    root = baseline.BASE / 'testpic'
    for p in sorted(root.rglob('*')):
        if p.is_symlink() or not p.is_file() or p.suffix.lower() not in ('.jpg', '.jpeg'):
            continue
        if p.stat().st_size > 16 * 1024 * 1024:
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
                valid = (im.format == 'JPEG' and im.mode == 'RGB' and w > 0 and h > 0
                         and w % 2 == 0 and h % 2 == 0 and w * h <= 8_000_000
                         and not im.info.get('progressive') and get_sampling(im) == 2)
                if valid:
                    im.verify()
            if valid:
                rows.append(dict(image_sha256=baseline.sha(p), size=[w, h], path=p))
        except (OSError, ValueError):
            continue
    rows.sort(key=lambda r: r['image_sha256'])
    unique = {r['image_sha256']: r for r in reversed(rows)}
    selected = [unique[h] for h in sorted(unique)[:2]]
    if len(selected) != 2:
        raise ValueError('Two existing even-size baseline420 JPEG controls required')
    return selected


def invoke(probe, path, raw, expected=0):
    result = subprocess.run([str(probe), str(path), str(raw)], capture_output=True, text=True, timeout=30)
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    if result.returncode != expected or len(rows) != 1:
        raise ValueError('Unexpected RGB probe result')
    row = rows[0]
    if expected == 0:
        count = row['width'] * row['height'] * 3
        if (row.get('rgb_format') not in ('RGB24_full_range_candidate','RGB24_fancy_full_range_candidate') or row.get('rgb_written_bytes') != count
                or not raw.is_file() or raw.stat().st_size != count or raw.stat().st_mode & 0o777 != 0o600):
            raise ValueError('Raw RGB contract/permission mismatch')
    return row


def pixels(path):
    from PIL import Image
    with Image.open(path) as im:
        im.load()
        rgb = im.convert('RGB')
        try:
            w, h = rgb.size
            return np.frombuffer(rgb.tobytes(), dtype=np.uint8).reshape(h, w, 3)
        finally:
            rgb.close()


def resized_pixels(image):
    from PIL import Image
    image.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=90)
    stream.seek(0)
    with Image.open(stream) as decoded:
        rgb = decoded.convert('RGB')
        try:
            return np.frombuffer(rgb.tobytes(), dtype=np.uint8).reshape(rgb.height, rgb.width, 3)
        finally:
            rgb.close()


def run(probe, trace, output):
    from PIL import Image, __version__ as pillow_version
    stage = Path(__file__).resolve().parent
    plan = output.with_name(output.stem + '_plan.json')
    if (output.exists() or plan.exists() or output.parent.resolve() != stage
            or probe.parent.resolve() != stage or trace.parent.resolve() != stage):
        raise ValueError('Fresh isolated stage output/probe/trace required')
    memory_guard()
    nodes = device_calls(trace.read_text())
    if nodes.get('/dev/nvhost-nvjpg', {}).get('ioctl_success', 0) <= 0:
        raise ValueError('Trace lacks successful NVJPG device ioctls')
    cases = select_cases()
    before = baseline.fingerprints()
    frozen = dict(source_sha256=baseline.sha(__file__), probe_sha256=baseline.sha(probe),
                  trace_sha256=baseline.sha(trace), cases=[{k: v for k, v in r.items() if k != 'path'} for r in cases],
                  max_case_pixels=8_000_000, minimum_available_kib=384 * 1024,
                  uniform_color_gate=dict(mae_lte=3, max_abs_error_lte=3),
                  acceptance='Uniform color gate only; real-image metrics diagnostic. No OCR quality acceptance or speed claim.')
    with plan.open('x') as f:
        json.dump(frozen, f, indent=2)
    report = dict(status='running', plan_sha256=baseline.sha(plan), pillow=pillow_version,
                  device_calls=nodes, synthetic=[], real=[], fingerprints_before=before,
                  note='No new model sessions, HTTP changes or Extract calls. RGB conversion version is recorded per probe metadata; no OCR quality claim.')
    with output.open('x') as stream:
        try:
            with tempfile.TemporaryDirectory(dir=str(stage), prefix='rgb_') as tmp:
                folder = Path(tmp)
                for i, color in enumerate(((15, 85, 180), (240, 30, 60), (30, 220, 60))):
                    memory_guard()
                    p, raw = folder / f'color_{i}.jpg', folder / f'color_{i}.rgb'
                    Image.new('RGB', (512, 256), color).save(p, subsampling=2, quality=95)
                    meta = invoke(probe, p, raw)
                    if (meta['width'], meta['height']) != (512, 256):
                        raise ValueError('Synthetic RGB geometry mismatch')
                    diff = metrics(pixels(p), np.frombuffer(raw.read_bytes(), dtype=np.uint8).reshape(256, 512, 3))
                    report['synthetic'].append(dict(case=f'color_{i}', metadata=meta, metrics=diff))
                    if diff['mae'] > 3 or diff['max_abs_error'] > 3:
                        raise ValueError('Uniform color gate failed')
                odd = folder / 'odd.jpg'
                Image.new('RGB', (513, 257)).save(odd, subsampling=2)
                rejected = folder / 'odd.rgb'
                report['synthetic'].append(dict(case='odd_rgb', metadata=invoke(probe, odd, rejected, 3)))
                if rejected.exists():
                    raise ValueError('Rejected odd input produced pixels')
                protected = folder / 'protected.rgb'
                protected.write_bytes(b'unchanged')
                invoke(probe, p, protected, 2)
                if protected.read_bytes() != b'unchanged':
                    raise ValueError('Existing output overwritten')
                report['synthetic'].append(dict(case='existing_output_preserved', passed=True))
                for i, case in enumerate(cases):
                    available = memory_guard()
                    p = case['path']
                    if baseline.sha(p) != case['image_sha256']:
                        raise ValueError('Frozen image changed')
                    raw = folder / f'real_{i}.rgb'
                    meta = invoke(probe, p, raw)
                    w, h = case['size']
                    if (meta['width'], meta['height']) != (w, h):
                        raise ValueError('Real RGB geometry mismatch')
                    candidate = np.frombuffer(raw.read_bytes(), dtype=np.uint8).reshape(h, w, 3)
                    raw_diff = metrics(pixels(p), candidate)
                    image = Image.frombytes('RGB', (w, h), candidate.tobytes())
                    del candidate
                    try:
                        cand_resized = resized_pixels(image)
                    finally:
                        image.close()
                    # Keep the legacy JPEG decoder's draft/thumbnail behavior on the reference side.
                    with Image.open(p) as original:
                        ref_resized = resized_pixels(original)
                    processed_diff = metrics(ref_resized, cand_resized)
                    report['real'].append(dict(image_sha256=case['image_sha256'], size=case['size'],
                        available_before_kib=available, metadata=meta, raw_rgb=raw_diff,
                        after_thumbnail_jpeg90=processed_diff, processed_shape=list(ref_resized.shape)))
                    del ref_resized, cand_resized
                    raw.unlink()
                    print('RGB diagnostic completed', i + 1, '/', len(cases), flush=True)
            report['fingerprints_after'] = baseline.fingerprints()
            if before != report['fingerprints_after'] or not baseline.health.ready(baseline.call('/healthz')):
                raise ValueError('Original deployment changed or service unhealthy')
            report['status'] = 'pixel_and_device_diagnostic_completed_not_ocr_validated'
        except Exception as exc:
            report.update(status='failed', error=type(exc).__name__)
            raise
        finally:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in ('status', 'synthetic', 'real', 'device_calls')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.probe.resolve(), args.trace.resolve(), args.output)
