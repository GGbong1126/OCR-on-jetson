"""Poll readiness without business requests or license count consumption."""
import argparse
import json
import time
from urllib.request import build_opener, ProxyHandler


def ready(body):
    if (body.get('status') != 'ready' or body.get('scope') != 'full_business'
            or body.get('deployment') not in ('jetson-r1','jetson-r2')):
        return False
    rows = body.get('models_execution',{})
    warmup = body.get('warmup_execution',{})
    for role in ('det','rec','layout','cell','yolo','ner','enhance'):
        row = rows.get(role,{})
        if row.get('errors') != 0 or warmup.get(role,{}).get('calls',0) < 1:
            return False
        if row.get('cpu_fallback',{}).get('errors',0):
            return False
    required = ('det','rec','yolo','enhance') if body['deployment']=='jetson-r1' else (
        'det','rec','layout','cell','yolo','ner','enhance')
    return all(rows[r].get('backend') == 'tensorrt82_cuda' for r in required)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--timeout',type=int,default=300)
    args = ap.parse_args()
    opener, end = build_opener(ProxyHandler({})), time.monotonic()+args.timeout
    while time.monotonic() < end:
        try:
            with opener.open('http://127.0.0.1:5051/healthz',timeout=3) as response:
                data = response.read(65537)
                if response.status == 200 and len(data) <= 65536 and ready(json.loads(data)):
                    print('Jetson full-business readiness passed, all model warmups recorded.')
                    return
        except (OSError,ValueError,TypeError):
            pass
        time.sleep(2)
    raise SystemExit('Jetson readiness deadline exceeded')


if __name__ == '__main__':
    main()
