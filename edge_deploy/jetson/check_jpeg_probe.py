"""Isolated probe checks: geometry and input rejection only, no pixel/OCR claims."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import baseline_batch_ocr as baseline


def check(probe, path, expected_code, size=None):
    result = subprocess.run([str(probe), str(path)], capture_output=True, text=True, timeout=30)
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    if result.returncode != expected_code or len(rows) != 1:
        raise ValueError('Probe returned an unexpected status')
    row = rows[0]
    if expected_code == 0 and ((row.get('width'), row.get('height')) != size
                              or row.get('status') != 'decoded_geometry_checked_not_ocr_validated'
                              or len(row.get('planes', [])) != 3):
        raise ValueError('Probe geometry differs from image header')
    return dict(returncode=result.returncode, result=row)


def run(probe, source, output):
    from PIL import Image
    from PIL.JpegImagePlugin import get_sampling
    stage = Path(__file__).resolve().parent
    if output.exists() or output.parent.resolve() != stage or probe.parent.resolve() != stage:
        raise ValueError('Fresh report and probe must be in the isolated stage')
    prior = json.loads(source.read_text())
    if prior['status'] != 'serial_ocr_controls_recorded_not_accuracy_validated':
        raise ValueError('A completed M0 control report is required')
    report = dict(status='running', probe_sha256=baseline.sha(probe),
                  source_sha256=baseline.sha(__file__), baseline_sha256=baseline.sha(source),
                  synthetic=[], real=[], note='Geometry/rejection checks only; no RGB pixels, OCR accuracy or speed benchmark.')
    with output.open('x', encoding='utf-8') as stream:
        try:
            with tempfile.TemporaryDirectory(prefix='jpeg_probe_', dir=str(stage)) as tmp:
                folder = Path(tmp)
                image = Image.new('RGB', (512, 256), (30, 90, 170))
                cases = []
                for name, options, code in (
                    ('baseline.jpg', dict(subsampling=2), 0),
                    ('progressive.jpg', dict(subsampling=2, progressive=True), 3),
                    ('444.jpg', dict(subsampling=0), 3),
                    ('image.png', {}, 3),
                ):
                    p = folder / name
                    image.save(p, **options)
                    cases.append((name, code, image.size if code == 0 else None))
                odd = folder / 'odd.jpg'
                Image.new('RGB', (513, 257), (90, 120, 160)).save(odd, subsampling=2)
                cases.append((odd.name, 0, (513, 257)))
                grey = folder / 'grey.jpg'
                image.convert('L').save(grey)
                cases.append((grey.name, 3, None))
                (folder / 'invalid.jpg').write_bytes(b'\xff\xd8\x00\x00')
                cases.append(('invalid.jpg', 2, None))
                # Oversized dimensions are rejected from the header before decoding/allocation.
                data = bytearray((folder / 'baseline.jpg').read_bytes())
                marker = data.index(b'\xff\xc0')
                data[marker + 5:marker + 9] = b'\x1f\x40\x1f\x40'  # 8000 x 8000
                (folder / 'pixel_limit.jpg').write_bytes(data)
                cases.append(('pixel_limit.jpg', 2, None))
                with (folder / 'byte_limit.jpg').open('xb') as large:
                    large.truncate(16 * 1024 * 1024 + 1)
                cases.append(('byte_limit.jpg', 2, None))
                for name, code, size in cases:
                    report['synthetic'].append(dict(name=name, **check(probe, folder / name, code, size)))
                wanted = {r['image_sha256'] for r in prior['rows']}
                for p in sorted((baseline.BASE / 'testpic').rglob('*')):
                    if not p.is_file() or p.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                        continue
                    digest = baseline.sha(p)
                    if digest not in wanted:
                        continue
                    with Image.open(p) as im:
                        im.verify()
                    with Image.open(p) as im:
                        size = im.size
                        supported = (im.format == 'JPEG' and im.mode == 'RGB'
                                     and not im.info.get('progressive') and get_sampling(im) == 2)
                    report['real'].append(dict(image_sha256=digest, **check(probe, p, 0 if supported else 3,
                                                                         size if supported else None)))
                    wanted.remove(digest)
                if wanted or not any(r['returncode'] == 0 for r in report['real']):
                    raise ValueError('Real JPEG control coverage incomplete')
            report['status'] = 'probe_geometry_and_rejection_checked_not_ocr_validated'
        except Exception as exc:
            report.update(status='failed', error=type(exc).__name__)
            raise
        finally:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.probe.resolve(), args.baseline, args.output)
