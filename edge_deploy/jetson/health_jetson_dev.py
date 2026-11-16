"""Fixed-loopback development readiness; no inference, quota or body logging."""
import argparse
import json
import time
from urllib.request import build_opener, ProxyHandler
import health_rec_wide as core

MODE = 'jetson_development_no_license'
AUTHORIZATION = dict(hardware_license='disabled', usage_limit='disabled', scope='isolated_board_test')


def ready(body):
    return (isinstance(body, dict) and body.get('service_mode') == MODE
            and body.get('authorization') == AUTHORIZATION and core.ready(body, 'on'))


def wait(timeout=300):
    if type(timeout) is not int or timeout <= 0: raise ValueError('Positive timeout required')
    opener = build_opener(ProxyHandler({}), core.NoRedirect())
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        try:
            with opener.open(core.URL, timeout=min(3, max(.001, deadline-time.monotonic()))) as response:
                raw = response.read(65537)
                if response.status == 200 and len(raw) <= 65536 and ready(json.loads(raw)): return
        except (OSError, ValueError, TypeError): pass
        time.sleep(min(1, max(0, deadline-time.monotonic())))
    raise RuntimeError('Development service readiness timeout')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=int, default=300)
    wait(parser.parse_args().timeout)
    print('Development service ready; seven FP32 GPU models, wide rec, authorization limits disabled.')
