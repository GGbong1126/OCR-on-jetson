"""M0: freeze four serial OCR controls; no Extract, text dumps or service changes."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener

import health_jetson_dev as health
import health_rec_wide as core

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'business'))
from smoke_business import ocr_statistics

BASE = Path('/home/xchq/work/autoocr')
RELEASE = BASE / 'releases/jetson-r2'
DEV = BASE / 'overlays/jetson-dev-e129-r1'


def signature(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def controls(cases):
    if len(cases) != 26 or len({r['image_sha256'] for r in cases}) != 13:
        raise ValueError('Expected the original thirteen-image control set')
    hashes = [cases[0]['image_sha256']]
    wide = next(r['image_sha256'] for r in cases if r['wide_calls'] > 0)
    if wide == hashes[0]:
        hashes.insert(0, next(r['image_sha256'] for r in cases if r['image_sha256'] != wide))
    else:
        hashes.append(wide)
    # Freeze both ordinary and enhanced processing before making any request.
    return [dict(image_sha256=h, image_type=t) for h in hashes for t in (3, 1)]


def call(endpoint, payload=None):
    if endpoint not in ('/healthz', '/OCR'):
        raise ValueError('Only OCR and readiness are in scope')
    raw = None if payload is None else json.dumps(payload).encode()
    request = Request('http://127.0.0.1:5051' + endpoint, data=raw,
                      headers={'Content-Type': 'application/json'})
    with build_opener(ProxyHandler({}), core.NoRedirect()).open(request, timeout=240) as response:
        data = response.read(24 * 1024 * 1024 + 1)
        if (response.status != 200 or len(data) > 24 * 1024 * 1024
                or response.headers.get('X-AutoOCR-Mode') != health.MODE):
            raise ValueError('Unexpected development service response')
        return json.loads(data)


def delta(before, after, image_type):
    if (not health.ready(before) or not health.ready(after)
            or before['worker_pid'] != after['worker_pid']):
        raise ValueError('Service unhealthy or restarted during control')
    calls = {k: after['models_execution'][k]['calls'] - before['models_execution'][k]['calls']
             for k in core.ROLES}
    if (calls['ner'] != 0 or any(calls[k] != 1 for k in ('det', 'layout', 'yolo'))
            or calls['rec'] <= 0 or calls['cell'] < 0
            or (calls['enhance'] <= 0 if image_type == 1 else calls['enhance'] != 0)):
        raise ValueError('Unexpected OCR-only execution counts or concurrent caller')
    return calls


def fingerprints():
    config = RELEASE / 'edge_deploy/jetson/deployment.json'
    spec = json.loads(config.read_text())
    paths = [config, RELEASE / 'server.py', RELEASE / 'server_config.py',
             RELEASE / 'jetson_source_manifest.json', DEV / 'manifest.json']
    paths += [RELEASE / row['path'] for row in spec['engines'].values()]
    for row in spec['engines'].values():
        paths += [RELEASE / plugin['path'] for plugin in row.get('plugins', [])]
    for overlay in (DEV, BASE / 'overlays/rec-wide-e124-r1'):
        paths += [p for p in overlay.iterdir() if p.is_file()]
    return {str(p.relative_to(BASE)): sha(p) for p in sorted(set(paths))}


def run(output):
    if output.exists() or output.parent.resolve() != Path(__file__).resolve().parent:
        raise ValueError('Use a fresh report in the isolated script directory')
    if BASE.resolve() != BASE or not (BASE / 'testpic').is_dir():
        raise ValueError('Expected existing project and test images')
    prior = BASE / 'staging/rec_soak_e126_r1/plan.json'
    cases = controls(json.loads(prior.read_text())['cases'][:26])
    wanted = {c['image_sha256'] for c in cases}
    files = {}
    for p in sorted((BASE / 'testpic').rglob('*')):
        if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            digest = sha(p)
            if digest in wanted:
                files[digest] = p
    if set(files) != wanted:
        raise ValueError('Existing control images missing')
    before = fingerprints()
    report = dict(status='running', schema_version=1, source_sha256=sha(__file__),
                  prior_plan_sha256=sha(prior), cases=cases, fingerprints_before=before, rows=[],
                  note='Serial single-pass controls, not GT or a speed benchmark. No Extract requests.')
    with output.open('x', encoding='utf-8') as stream:
        try:
            for case in cases:
                raw = files[case['image_sha256']].read_bytes()
                if len(raw) > 16 * 1024 * 1024:
                    raise ValueError('Control image exceeds input limit')
                first = call('/healthz')
                if not health.ready(first):
                    raise ValueError('Development service is not ready')
                start = time.perf_counter()
                body = call('/OCR', dict(img=base64.b64encode(raw).decode('ascii'),
                                         imageType=str(case['image_type'])))
                wall = time.perf_counter() - start
                stats = ocr_statistics(body)
                after = call('/healthz')
                row = dict(case, response_sha256=signature(body), statistics=stats,
                           model_calls=delta(first, after, case['image_type']), request_wall_s=wall)
                report['rows'].append(row)
                print('OCR control completed', len(report['rows']), '/', len(cases), flush=True)
            report['fingerprints_after'] = fingerprints()
            if before != report['fingerprints_after']:
                raise ValueError('Baseline deployment changed')
            report['status'] = 'serial_ocr_controls_recorded_not_accuracy_validated'
        except Exception as exc:
            report.update(status='failed', error=type(exc).__name__)
            raise
        finally:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], rows=report['rows']), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
