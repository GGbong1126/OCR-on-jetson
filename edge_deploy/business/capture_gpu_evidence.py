"""Read-only deployment snapshot. No OCR/Extract, counters or private log access."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from urllib.request import ProxyHandler, build_opener

from release_business import verify, fingerprint
from service_health import is_ready


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a new report path')
    info = verify(a.release)
    result = subprocess.run(['systemctl', 'show', 'autoocr-business.service',
                 '-p', 'MainPID', '-p', 'ActiveState', '-p', 'SubState', '-p', 'NRestarts',
                 '-p', 'WorkingDirectory', '-p', 'UnitFileState'], check=True, capture_output=True, text=True)
    unit = dict(line.split('=', 1) for line in result.stdout.splitlines())
    if unit['WorkingDirectory'] != str(a.release.resolve() / 'edge_deploy/business'):
        raise RuntimeError('Release is not the current systemd service')
    with build_opener(ProxyHandler({})).open('http://127.0.0.1:5051/healthz', timeout=15) as r:
        body = json.load(r)
        if not is_ready(r.status, body, 'tensorrt_fp32_with_ort_fallback', 'cuda'):
            raise RuntimeError('GPU service not ready')
    memory = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    value = {'captured_at_utc': datetime.now(timezone.utc).isoformat(), 'release': info,
             'systemd': unit, 'unit_sha256': fingerprint('/etc/systemd/system/autoocr-business.service')['sha256'],
             'health': body, 'gpu_memory_total_used_free_mib_snapshot': memory.stdout.strip(),
             'note': 'Actual health read after execution; not GT, benchmark, VRAM peak or license balance.'}
    with a.output.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
