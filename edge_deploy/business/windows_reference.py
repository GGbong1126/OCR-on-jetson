"""Temporary loopback reference: run native server.main, expose read-only runtime info.

No OCR/device/threshold overrides. Does not activate, bypass or reset a license.
Separate runtime logs/temp, original source-root license and existing counter state.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys


def fingerprint_file(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            digest.update(data)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def runtime_info(server, source):
    p, cfg = server.pipeline, server.cfg
    sessions = {'layout': p._layout_detector, 'cell': p._table_cell_processor,
                'ocr_det': getattr(p._ocr_engine, 'det_session', None),
                'ocr_rec': getattr(p._ocr_engine, 'rec_session', None)}
    providers = {name: s.get_providers() if hasattr(s, 'get_providers') else None for name, s in sessions.items()}
    assets = {name: getattr(cfg, name) for name in ('ONNX_LAYOUT_MODEL', 'ONNX_CELL_MODEL',
                                                 'ONNX_OCR_DET_MODEL', 'ONNX_OCR_REC_MODEL')}
    assets['yolo'] = str(Path(cfg.LAYOUT_MODEL_DIR)/'det_photo/best1.pt')
    ready = all((p._models_loaded, p._use_onnx_layout, p._use_onnx_cell, p._use_onnx_ocr,
                 p._yolo_model is not None, p._ner_pipeline is not None, p._imgpre_model is not None))
    return {'status': 'ready' if ready else 'incomplete', 'platform': platform.system(), 'pid': os.getpid(),
            'entry': 'native_server_main_default_device', 'use_gpu': p.use_gpu,
            'image_enhancement_device': str(p._imgpre_device),
            'ner_device': str(getattr(p._ner_pipeline, 'device', 'unknown')),
            'providers': providers, 'yolo_conf': p.yolo_conf, 'ner_threshold': cfg.NER_SCORE_THRESHOLD,
            'torch_threads': server.torch.get_num_threads(), 'opencv_threads': server.cv2.getNumThreads(),
            'packages': {name: importlib.metadata.version(name) for name in
                         ('torch', 'onnxruntime', 'ultralytics', 'transformers', 'numpy', 'Flask')},
            'source_files': {name: fingerprint_file(source/name) for name in
                             ('server.py', 'server_config.py', 'ner_postprocess.py', 'windows_hardware.py')},
            'assets': {name: {'path': str(Path(path).resolve()), **fingerprint_file(path)} for name, path in assets.items()},
            'note': 'CPU/GPU labels describe this process, not a controlled CPU-only comparison or benchmark.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--port', type=int, default=5052)
    args = parser.parse_args()
    if sys.platform != 'win32':
        parser.error('This reference must run in native Windows Python')
    if not 1024 <= args.port <= 65535 or args.port == 5051:
        parser.error('Choose an unprivileged port separate from WSL5051')
    source = Path(__file__).resolve().parents[2]
    runtime = args.runtime.resolve()
    runtime.mkdir(parents=True, exist_ok=False)
    os.chdir(source)
    sys.path.insert(0, str(source))
    os.environ['AUTOOCR_LOG_FILE'] = str(runtime/'server.log')
    os.environ['AUTOOCR_TEMP_DIR'] = str(runtime/'temp')
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    server = importlib.import_module('server')
    if not server.validate_license():
        raise RuntimeError('Windows license invalid; no GUI activation attempted')
    original_run = server.app_online.run
    info = {}

    @server.app_online.get('/__reference_info')
    def reference_info():
        return info

    def run_with_info(*a, **kw):
        info.update(runtime_info(server, source))
        (runtime/'runtime_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        if info['status'] != 'ready':
            raise RuntimeError('Native reference did not load the full business path')
        return original_run(*a, **kw)

    # Observe only the final serving boundary; native main still owns license/models/arguments.
    server.app_online.run = run_with_info
    sys.argv = [str(source/'server.py'), '--host', '127.0.0.1', '--port-online', str(args.port)]
    multiprocessing.freeze_support()
    server.main()


if __name__ == '__main__':
    main()
