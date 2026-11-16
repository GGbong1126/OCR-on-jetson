"""Readiness for the opt-in E124 factory; never imports models or license code.

Accept the declared on/off mode only. A normal out-of-profile CPU call is not
a backend failure, but an error, closed context or incomplete warmup is.
"""
import argparse
import json
import time
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

ROLES = ('det', 'rec', 'layout', 'cell', 'yolo', 'ner', 'enhance')
ORIGINAL = {'x': [[1, 3, 48, 32], [1, 3, 48, 160], [1, 3, 48, 1536]]}
WIDE = {'x': [[1, 3, 48, 1537], [1, 3, 48, 1696], [1, 3, 48, 2304]]}
URL = 'http://127.0.0.1:5051/healthz'


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def native(row, minimum=1):
    return (isinstance(row, dict) and row.get('backend') == 'tensorrt82_cuda'
            and row.get('precision') == 'fp32_no_tf32' and row.get('closed') is False
            and type(row.get('errors')) is int and row['errors'] == 0
            and integer(row.get('calls'), minimum))


def fallback(row):
    return (isinstance(row, dict) and integer(row.get('calls'))
            and type(row.get('errors')) is int and row['errors'] == 0)


def wide(row):
    if (not isinstance(row, dict) or row.get('backend') != 'tensorrt82_fp32_wide_overlay'
            or row.get('precision') != 'fp32_no_tf32' or row.get('closed') is not False
            or type(row.get('errors')) is not int or row['errors'] != 0
            or type(row.get('route_errors')) is not int or row['route_errors'] != 0
            or not integer(row.get('calls'), 4)):
        return False
    base, extra = row.get('original_execution'), row.get('wide_execution')
    return (native(base) and native(extra, 3) and base.get('profiles') == ORIGINAL
            and extra.get('profiles') == WIDE and row.get('profiles') == ORIGINAL
            and row.get('wide_profiles') == WIDE and row['calls'] == base['calls'] + extra['calls'])


def ready(body, mode='on'):
    if mode not in ('on', 'off') or not isinstance(body, dict):
        return False
    backend = 'tensorrt82_fp32_wide_with_ort_fallback' if mode == 'on' else 'tensorrt82_fp32_with_ort_fallback'
    if (body.get('status') != 'ready' or body.get('scope') != 'full_business'
            or body.get('deployment') != 'jetson-r2' or body.get('ocr_backend') != backend
            or body.get('cpu_modules') != {}
            or body.get('out_of_profile_policy') != 'onnxruntime_cpu_original_tensor_no_resize'
            or any(body.get(k) is not True for k in ('layout', 'cell', 'yolo', 'ner', 'image_enhancement'))):
        return False
    current, warm = body.get('models_execution'), body.get('warmup_execution')
    if not isinstance(current, dict) or not isinstance(warm, dict) or set(current) != set(ROLES) or set(warm) != set(ROLES):
        return False
    for role in ROLES:
        validate = wide if role == 'rec' and mode == 'on' else native
        row, first = current[role], warm[role]
        if not validate(row) or not validate(first) or row['calls'] < first['calls']:
            return False
        if role in ('det', 'rec'):
            if not fallback(row.get('cpu_fallback')) or not fallback(first.get('cpu_fallback')):
                return False
            if row['cpu_fallback']['calls'] < first['cpu_fallback']['calls']:
                return False
        if role == 'rec':
            if row.get('profiles') != ORIGINAL or first.get('profiles') != ORIGINAL:
                return False
            if mode == 'off' and any(k in r for r in (row, first) for k in ('wide_execution', 'wide_profiles')):
                return False
            if mode == 'on' and any(row[k]['calls'] < first[k]['calls'] for k in ('original_execution', 'wide_execution')):
                return False
    return True


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def wait_ready(timeout, mode, opener=None):
    if type(timeout) is not int or timeout <= 0 or mode not in ('on', 'off'):
        raise ValueError('Positive integer timeout and explicit on/off mode required')
    opener = opener or build_opener(ProxyHandler({}), NoRedirect())
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with opener.open(URL, timeout=min(3.0, max(.001, deadline - time.monotonic()))) as response:
                raw = response.read(65537)
                if response.status == 200 and len(raw) <= 65536 and ready(json.loads(raw), mode):
                    return
        except (OSError, ValueError, TypeError):
            pass
        left = deadline - time.monotonic()
        if left > 0:
            time.sleep(min(2.0, left))
    raise RuntimeError('Jetson readiness deadline exceeded; no response contents logged')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('on', 'off'))
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    wait_ready(args.timeout, args.mode)
    print('Jetson readiness passed: seven FP32 models, expected wide mode and warmups')


if __name__ == '__main__':
    main()
