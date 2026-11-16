"""Quality-only JPEG/RGB paired OCR via existing loopback service; no Extract.

Lossless PNG carries candidate pixels for testing only, not production IPC or a
decode/pipeline performance comparison. No additional model process is loaded.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import tempfile
import time

import baseline_batch_ocr as baseline
import compare_jpeg_rgb as rgb


def differences(reference, candidate):
    a, b = reference['msg']['pages'][0], candidate['msg']['pages'][0]
    coords = lambda page: [[row[k] for k in ('x1', 'y1', 'x2', 'y2')] for row in page['blocks']]
    texts = lambda page: [row.get('text', '') for row in page['blocks']]
    return dict(strict_response_equal=baseline.signature(reference) == baseline.signature(candidate),
                ordered_texts_equal=texts(a) == texts(b), markdown_equal=a.get('markdown') == b.get('markdown'),
                ordered_coordinates_equal=coords(a) == coords(b),
                dimensions_equal=(a['width'], a['height']) == (b['width'], b['height']))


def call_ocr(raw, image_type):
    rgb.memory_guard()
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError('Quality transport exceeds original client limit')
    before = baseline.call('/healthz')
    if not baseline.health.ready(before):
        raise ValueError('Existing service not ready')
    start = time.perf_counter()
    result = baseline.call('/OCR', dict(img=base64.b64encode(raw).decode('ascii'), imageType=str(image_type)))
    wall = time.perf_counter() - start
    stats = baseline.ocr_statistics(result)
    after = baseline.call('/healthz')
    calls = baseline.delta(before, after, image_type)
    for role in ('det', 'rec'):
        if after['models_execution'][role]['cpu_fallback'] != before['models_execution'][role]['cpu_fallback']:
            raise RuntimeError('Unexpected CPU fallback; stopping quality diagnostic')
    return result, dict(statistics=stats, model_calls=calls, single_request_wall_s=wall)


def controls():
    from PIL import Image
    from PIL.JpegImagePlugin import get_sampling
    prior = baseline.BASE / 'staging/rec_soak_e126_r1/plan.json'
    known = {row['image_sha256'] for row in json.loads(prior.read_text())['cases']}
    found = {}
    for p in sorted((baseline.BASE / 'testpic').rglob('*')):
        if p.is_symlink() or not p.is_file() or p.suffix.lower() not in ('.jpg', '.jpeg'):
            continue
        if p.stat().st_size > 16 * 1024 * 1024:
            continue
        digest = baseline.sha(p)
        if digest not in known:
            continue
        with Image.open(p) as im:
            w, h = im.size
            if (im.mode != 'RGB' or im.info.get('progressive') or get_sampling(im) != 2
                    or w % 2 or h % 2 or w * h > 8_000_000):
                continue
            im.verify()
        found[digest] = dict(image_sha256=digest, size=[w, h], path=p)
    if len(found) < 2:
        raise ValueError('Two previously verified even-size controls required')
    return [found[h] for h in sorted(found)[:2]], prior


def run(probe, output):
    from PIL import Image
    stage = Path(__file__).resolve().parent
    plan = output.with_name(output.stem + '_plan.json')
    if (probe.parent.resolve() != stage or output.parent.resolve() != stage
            or plan.exists() or output.exists()):
        raise ValueError('Fresh isolated report/probe required')
    rgb.memory_guard()
    cases, prior = controls()
    initial_health = baseline.call('/healthz')
    if not baseline.health.ready(initial_health):
        raise ValueError('Existing service not ready')
    before = baseline.fingerprints()
    frozen = dict(source_sha256=baseline.sha(__file__), probe_sha256=baseline.sha(probe),
                  prior_plan_sha256=baseline.sha(prior), image_types=[3, 1],
                  cases=[{k: v for k, v in row.items() if k != 'path'} for row in cases],
                  acceptance='Strict complete response equality; text/Markdown/coordinates also diagnostic, not GT.',
                  transport='Candidate RGB -> lossless PNG -> existing /OCR, quality-only; not deployment IPC or performance.',
                  fingerprints_before=before)
    with plan.open('x') as stream:
        json.dump(frozen, stream, indent=2)
    report = dict(status='running', plan_sha256=baseline.sha(plan), rows=[], extract_requests=0,
                  extra_model_processes=0, before_worker_pid=initial_health['worker_pid'])
    with output.open('x') as stream:
        try:
            with tempfile.TemporaryDirectory(prefix='ocr_rgb_', dir=str(stage)) as tmp:
                for index, case in enumerate(cases):
                    rgb.memory_guard()
                    p = case['path']
                    if baseline.sha(p) != case['image_sha256']:
                        raise ValueError('Frozen quality image changed')
                    raw_path = Path(tmp) / f'{index}.rgb'
                    metadata = rgb.invoke(probe, p, raw_path)
                    if [metadata['width'], metadata['height']] != case['size']:
                        raise ValueError('Candidate dimensions differ')
                    pixels = raw_path.read_bytes()
                    png = io.BytesIO()
                    with Image.frombytes('RGB', tuple(case['size']), pixels) as im:
                        im.save(png, format='PNG', compress_level=3)
                    payload = png.getvalue()
                    with Image.open(io.BytesIO(payload)) as checked:
                        if checked.mode != 'RGB' or checked.size != tuple(case['size']) or checked.tobytes() != pixels:
                            raise ValueError('Lossless test transport changed RGB pixels')
                    del pixels
                    raw_path.unlink()
                    original = p.read_bytes()
                    for image_type in (3, 1):
                        reference, first = call_ocr(original, image_type)
                        candidate, second = call_ocr(payload, image_type)
                        report['rows'].append(dict(image_sha256=case['image_sha256'], image_type=image_type,
                            reference_sha256=baseline.signature(reference), candidate_sha256=baseline.signature(candidate),
                            reference=first, candidate=second, comparison=differences(reference, candidate)))
                        del reference, candidate
                        print('OCR RGB quality pair completed', len(report['rows']), '/ 4', flush=True)
            after = baseline.call('/healthz')
            report['fingerprints_after'] = baseline.fingerprints()
            if (report['fingerprints_after'] != before or not baseline.health.ready(after)
                    or after['worker_pid'] != initial_health['worker_pid']):
                raise ValueError('Original deployment changed or worker restarted/unhealthy')
            report['after_worker_pid'] = after['worker_pid']
            report['all_strict_equal'] = all(r['comparison']['strict_response_equal'] for r in report['rows'])
            report['status'] = ('paired_http_consistent_not_accuracy_validated' if report['all_strict_equal']
                                else 'paired_http_differences_not_accepted')
        except Exception as exc:
            report.update(status='failed', error=type(exc).__name__)
            raise
        finally:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], comparisons=[r['comparison'] for r in report['rows']]), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.probe.resolve(), args.output)
