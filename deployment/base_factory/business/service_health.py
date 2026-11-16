"""Bounded startup checks; loopback only, no inference, license or counter access."""
import argparse
import json
import socket
import time
from urllib.request import ProxyHandler, build_opener


URL = 'http://127.0.0.1:5051/healthz'
MODELS = ('layout', 'yolo', 'cell', 'ner', 'image_enhancement')


def is_ready(status, body, backend='onnxruntime_cpu', other_backend=None):
    if other_backend is not None:
        if not isinstance(body, dict) or body.get('other_models_backend') != other_backend:
            return False
        if other_backend == 'cuda':
            rows = body.get('models_execution', {})
            if not isinstance(rows, dict):
                return False
            for name in MODELS:
                row = rows.get(name, {})
                if (not isinstance(row, dict) or row.get('device') != 'cuda:0'
                        or row.get('errors') != 0 or type(row.get('warmup_calls')) is not int
                        or row['warmup_calls'] < 1):
                    return False
                if name in ('layout', 'cell') and row.get('warmup_node_providers', {}).get('CUDAExecutionProvider', 0) < 1:
                    return False
    if backend == 'tensorrt_fp32_with_ort_fallback':
        execution = body.get('ocr_execution', {}) if isinstance(body, dict) else {}
        if not isinstance(execution, dict):
            return False
        for role in ('det', 'rec'):
            row = execution.get(role, {})
            if (not isinstance(row, dict) or row.get('healthy') is not True
                    or type(row.get('warmup_trt_calls')) is not int or row['warmup_trt_calls'] < 1
                    or row.get('errors') != 0):
                return False
    return (status == 200 and isinstance(body, dict)
            and body.get('status') == 'ready'
            and body.get('scope') == 'full_business'
            and body.get('ocr_backend') == backend
            and all(body.get(name) is True for name in MODELS))


def check_port():
    # Also prevents accepting an older process's health response during startup.
    # Not an atomic reservation: Gunicorn must still bind the socket itself.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('127.0.0.1', 5051))
        except OSError as exc:
            raise RuntimeError('Loopback port 5051 unavailable; do not stop an unknown owner.') from exc


def wait_ready(timeout, backend='onnxruntime_cpu', other_backend=None):
    if timeout <= 0:
        raise ValueError('timeout must be positive')
    deadline = time.monotonic() + timeout
    opener = build_opener(ProxyHandler({}))
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            with opener.open(URL, timeout=min(3.0, remaining)) as response:
                raw = response.read(16385)
                if len(raw) <= 16384 and is_ready(response.status, json.loads(raw), backend, other_backend):
                    return
        except (OSError, ValueError, TypeError):
            # No response bodies, model logs or licensing details in diagnostics.
            pass
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(2.0, remaining))
    raise RuntimeError('Full-business readiness deadline exceeded; inspect local service logs.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-port', action='store_true')
    mode.add_argument('--wait-ready', action='store_true')
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--backend', choices=('onnxruntime_cpu', 'tensorrt_fp32_with_ort_fallback'),
                        default='onnxruntime_cpu')
    parser.add_argument('--other-backend', choices=('cpu', 'cuda'))
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if args.check_port:
        check_port()
        print('Loopback port 5051 available.', flush=True)
    else:
        wait_ready(args.timeout, args.backend, args.other_backend)
        print('Full-business readiness passed (all models, '+args.backend+', loopback).', flush=True)


if __name__ == '__main__':
    main()
