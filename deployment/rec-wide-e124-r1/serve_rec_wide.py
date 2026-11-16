"""Optional business factory over immutable jetson-r2; unset config to roll back.

This module does not launch a listener. create_app keeps the original hardware
license/counter rules. Offline load_business is separate, like --probe-models.
"""
import importlib
import json
import os
from pathlib import Path
import sys
import rec_wide_adapter as adapter

CONFIG_ENV = 'AUTOOCR_REC_WIDE_CONFIG'


def original_factory():
    root = Path(os.environ['AUTOOCR_SOURCE_ROOT']).resolve()
    if root != adapter.RELEASE:
        raise ValueError('Wide adapter requires immutable jetson-r2')
    directory = root / 'edge_deploy/jetson'
    if adapter.digest(directory / 'deployment.json') != adapter.DEPLOYMENT_SHA:
        raise ValueError('Original deployment changed')
    for name, expected in adapter.BASE_ADAPTERS.items():
        if adapter.digest(directory / name) != expected:
            raise ValueError('Original factory/backend changed')
    sys.path.insert(0, str(directory))
    base = importlib.import_module('serve_jetson')
    if Path(base.__file__).resolve() != directory / 'serve_jetson.py':
        raise ValueError('Unexpected original factory import')
    return base


def selected_config():
    value = os.environ.get(CONFIG_ENV)
    if value is None:
        return None
    if not value.strip():
        raise ValueError('Empty wide config is invalid; unset variable for rollback')
    path = Path(value).resolve()
    config, _ = adapter.validate_config(path)
    if adapter.digest(__file__) != config['adapters']['serve_rec_wide.py']:
        raise ValueError('Loaded factory differs from selected package')
    return path


def load_models(base, server, root, config, wide_config):
    pipeline, models = base.load_models(server, root, config)
    try:
        if wide_config is not None:
            adapter.install(models, wide_config)
        if not adapter.healthy(models):
            raise RuntimeError('Business backend not ready')
        return pipeline, models
    except Exception:
        adapter.close_models(models)
        raise


def load_business():
    """Offline only: no license, counter, HTTP creation or authorization mutation."""
    wide_config = selected_config()
    base = original_factory()
    server, root, config = base.deployment()
    pipeline, models = load_models(base, server, root, config, wide_config)
    return server, pipeline, models, config


def create_app():
    wide_config = selected_config()
    base = original_factory()
    if wide_config is None:
        return base.create_app()
    server, root, config = base.deployment()
    if not server.validate_license():
        raise RuntimeError('Jetson hardware activation required; no models or HTTP started')
    pipeline, models = load_models(base, server, root, config, wide_config)
    try:
        server.initialize_counter_file()
        server.pipeline = pipeline
        app = server.app_online
        app.config['MAX_CONTENT_LENGTH'] = 24 * 1024 * 1024

        @app.before_request
        def require_healthy():
            if not adapter.healthy(models):
                return {'status': 'error', 'message': 'Model backend unhealthy'}, 503

        @app.after_request
        def reject_partial(response):
            if not adapter.healthy(models):
                return app.make_response(({'status': 'error', 'message': 'Model backend unhealthy'}, 503))
            return response

        @app.get('/healthz')
        def healthz():
            body = dict(status='ready' if adapter.healthy(models) else 'error', scope='full_business',
                        deployment=root.name, ocr_backend='tensorrt82_fp32_wide_with_ort_fallback',
                        layout=True, cell=True, yolo=True, ner=True, image_enhancement=True,
                        models_execution=models.stats(), warmup_execution=models.warmup_execution,
                        cpu_modules=config['cpu_modules'],
                        out_of_profile_policy='onnxruntime_cpu_original_tensor_no_resize')
            return body, 200 if adapter.healthy(models) else 503
        # Explicit lifetime handle for a supervising process; no automatic hot swap.
        app.extensions['autoocr_wide_models'] = models
        return app
    except Exception:
        adapter.close_models(models)
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-models', type=Path, required=True)
    args = parser.parse_args()
    if args.probe_models.exists():
        parser.error('Report exists')
    models = None
    try:
        _, _, models, _ = load_business()
        with args.probe_models.open('x', encoding='utf-8') as f:
            json.dump(dict(status='offline_synthetic_ready_not_http_validated',
                           wide_enabled=os.environ.get(CONFIG_ENV) is not None,
                           execution=models.stats()), f, indent=2, allow_nan=False)
    finally:
        if models is not None:
            adapter.close_models(models)
