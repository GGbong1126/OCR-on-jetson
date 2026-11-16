"""Explicit board-only development HTTP, authorized without license/usage limits.

Immutable production server/functions remain intact. Only the private Extract
function binding uses an in-memory no-quota policy; its business bytecode is
unchanged. No fake license, activation, counter reads/writes or production flag.
"""
import atexit
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import FunctionType

BASE = Path('/home/xchq/work/autoocr')
PACKAGE = BASE/'overlays/jetson-dev-e129-r1'
OVERLAY = BASE/'overlays/rec-wide-e124-r1'
RELEASE = BASE/'releases/jetson-r2'
RUNTIME = BASE/'runtime/jetson-dev-e129-r1'
MODE = 'jetson_development_no_license'
AUTHORIZATION = dict(hardware_license='disabled', usage_limit='disabled', scope='isolated_board_test')
PINS = {'serve_rec_wide.py': '2f252ecc666c24b7996f1e80c31ab5142e061c79fb675ccd98dd102e387822bd',
        'rec_wide_adapter.py': '1bf3e7ef8a371f9d1f803679b036759b4ac8a7133a15e467be4bba36bc624090'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def preflight():
    expected = dict(AUTOOCR_ENABLE_DEV_NO_AUTH='1', AUTOOCR_SOURCE_ROOT=str(RELEASE),
        AUTOOCR_RUNTIME_DIR=str(RUNTIME), AUTOOCR_LOG_FILE=os.devnull,
        AUTOOCR_REC_WIDE_CONFIG=str(OVERLAY/'config.json'), AUTOOCR_HOST='127.0.0.1',
        AUTOOCR_PORT_ONLINE='5051')
    if Path(__file__).resolve().parent != PACKAGE or any(os.environ.get(k) != v for k,v in expected.items()):
        raise ValueError('Explicit isolated Jetson development environment required')
    if RUNTIME.is_symlink() or (RUNTIME.exists() and RUNTIME.resolve() != RUNTIME):
        raise ValueError('Development runtime must be isolated')
    manifest = json.loads((PACKAGE/'manifest.json').read_text())
    if manifest['kind'] != MODE:
        raise ValueError('Wrong delivery manifest')
    for name, digest in manifest['files'].items():
        if Path(name).name != name or (PACKAGE/name).is_symlink() or sha(PACKAGE/name) != digest:
            raise ValueError('Development delivery changed')
    for name, digest in PINS.items():
        if sha(OVERLAY/name) != digest: raise ValueError('Immutable wide adapter changed')
    return manifest


def unmetered_usage(amount):
    """User-authorized dev policy. Does not simulate credit or access storage."""
    if type(amount) is not int or amount != 1:
        raise ValueError('Unexpected usage operation')


def dev_extract(original):
    if original.__closure__ or original.__name__ != 'api_extract' or 'deal_counter' not in original.__code__.co_names:
        raise ValueError('Unexpected original Extract binding')
    namespace = original.__globals__.copy()
    namespace['deal_counter'] = unmetered_usage
    clone = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
    clone.__kwdefaults__ = original.__kwdefaults__
    return clone


def assemble(server, pipeline, models, config, adapter):
    from flask import Flask, request
    server.pipeline = pipeline
    # Legacy logs may contain recognized document headers. Do not persist them.
    server.logger.disabled = True
    app = Flask('autoocr_jetson_development', static_folder=None)
    app.config['MAX_CONTENT_LENGTH'] = 24*1024*1024

    def quiet_handler(fn):
        def handler():
            with open(os.devnull, 'w') as quiet:
                with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                    return fn()
        return handler

    app.add_url_rule('/OCR', 'api_ocr', quiet_handler(server.api_ocr), methods=['POST'])
    app.add_url_rule('/Extract', 'api_extract', quiet_handler(dev_extract(server.api_extract)), methods=['POST'])

    @app.before_request
    def guard():
        # This is not an internet-facing unauthenticated service.
        if request.remote_addr not in ('127.0.0.1', '::1'):
            return {'status': 'error', 'message': 'Loopback development access only'}, 403
        if not adapter.healthy(models):
            return {'status': 'error', 'message': 'Model backend unhealthy'}, 503

    @app.after_request
    def reject_partial(response):
        if request.remote_addr in ('127.0.0.1', '::1') and not adapter.healthy(models):
            response = app.make_response(({'status': 'error', 'message': 'Model backend unhealthy'}, 503))
        response.headers['X-AutoOCR-Mode'] = MODE
        return response

    @app.get('/healthz')
    def healthz():
        return dict(status='ready', scope='full_business', deployment='jetson-r2',
            service_mode=MODE, authorization=dict(AUTHORIZATION), worker_pid=os.getpid(),
            ocr_backend='tensorrt82_fp32_wide_with_ort_fallback', layout=True, cell=True,
            yolo=True, ner=True, image_enhancement=True, models_execution=models.stats(),
            warmup_execution=models.warmup_execution, cpu_modules=config['cpu_modules'],
            out_of_profile_policy='onnxruntime_cpu_original_tensor_no_resize')

    app.extensions['autoocr_dev_models'] = models
    return app


def create_app():
    preflight()
    sys.path.insert(0, str(OVERLAY))
    factory = importlib.import_module('serve_rec_wide')
    adapter = importlib.import_module('rec_wide_adapter')
    if any(Path(m.__file__).resolve().parent != OVERLAY for m in (factory, adapter)):
        raise ValueError('Unexpected wide module location')
    models = None
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, config = factory.load_business()
        app = assemble(server, pipeline, models, config, adapter)
        atexit.register(adapter.close_models, models)
        return app
    except Exception:
        if models is not None: adapter.close_models(models)
        raise


if __name__ == '__main__':
    preflight()
    print('Isolated development package verified; authorization limits disabled by user request.')
