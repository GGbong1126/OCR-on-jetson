"""Opt-in E123 FP32 recognition coverage. No experiment imports or input changes."""
import hashlib
import json
from pathlib import Path
import numpy as np

BASE = Path('/home/xchq/work/autoocr')
RELEASE = BASE / 'releases/jetson-r2'
DEPLOYMENT_SHA = '2ced56d7d35e429199d51282eaaf961d1408bf2bc0f2c5784fcfb2b0fecba6d8'
ENGINE_SHA = '8e3f9167d1496011c8bf528df0c5eff006bc8a0d4f254e2d42a9a47009e7c95b'
SOURCE_SHA = 'c93ea0ae08479f3b9a3d8b3549c7a45776f4830a10a430e3d355570d1daec5f4'
ORIGINAL_PROFILE = [[1, 3, 48, 32], [1, 3, 48, 160], [1, 3, 48, 1536]]
WIDE_PROFILE = [[1, 3, 48, 1537], [1, 3, 48, 1696], [1, 3, 48, 2304]]
ADAPTERS = ('rec_wide_adapter.py', 'serve_rec_wide.py')
BASE_ADAPTERS = {
    'serve_jetson.py': '5857f4cf88db30198db8db8670b5878c056ea5d865cd78b007062b1362024d3d',
    'jetson_models.py': '1f8cff5c7fc850fea75e45583c6fcb0baa6d320f8c386424ae92339086989b70',
    'trt82_backend.py': '259c49a140006f8fccc93a94632e0f5abbc30883e53760e90f688eb4a7f86925',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def configuration(directory):
    """Deterministic package manifest; generation does not enable the service."""
    directory = Path(directory)
    return dict(schema_version=1, kind='rec_wide_fp32_e123',
                base_release=str(RELEASE), deployment_sha256=DEPLOYMENT_SHA,
                source_sha256=SOURCE_SHA, engine='rec_wide_fp32.engine',
                engine_sha256=ENGINE_SHA, profile=WIDE_PROFILE,
                adapters={name: digest(directory / name) for name in ADAPTERS},
                policy='original_tensor_cpu_outside_union_no_error_fallback')


def validate_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    if config != configuration(path.parent):
        raise ValueError('Wide rec configuration or package integrity differs')
    # The loaded implementation must be the one named in the selected package.
    if digest(__file__) != config['adapters']['rec_wide_adapter.py']:
        raise ValueError('Loaded adapter differs from selected package')
    engine = path.parent / config['engine']
    if engine.is_symlink() or digest(engine) != ENGINE_SHA:
        raise ValueError('Wide rec engine integrity differs')
    return config, engine


class WideRoute:
    """Two separately owned GPU contexts inside the original OCRSession.

    Installation/removal is startup-only, never a live hot swap. Inference
    errors are latched, including errors swallowed by legacy business code.
    """
    def __init__(self, base, wide):
        self.base, self.wide = base, wide
        self.route_errors = 0
        self._closed = False

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def closed(self):
        return self._closed or self.base.closed or self.wide.closed

    @property
    def errors(self):
        return self.route_errors + self.base.errors + self.wide.errors

    @property
    def calls(self):
        return self.base.calls + self.wide.calls

    def require_healthy(self):
        if self.closed or self.errors:
            raise RuntimeError('Wide recognition route unavailable')

    def in_profile(self, feeds):
        self.require_healthy()
        try:
            return self.base.in_profile(feeds) or self.wide.in_profile(feeds)
        except Exception:
            self.route_errors += 1
            raise

    def run(self, names, feeds):
        self.require_healthy()
        try:
            if self.base.in_profile(feeds):
                return self.base.run(names, feeds)
            if self.wide.in_profile(feeds):
                return self.wide.run(names, feeds)
            raise ValueError('Out-of-profile inputs belong to original outer CPU policy')
        except Exception:
            self.route_errors += 1
            raise

    def stats(self):
        return dict(self.base.stats(), backend='tensorrt82_fp32_wide_overlay',
                    calls=self.calls, errors=self.errors, closed=self.closed,
                    original_execution=self.base.stats(), wide_execution=self.wide.stats(),
                    wide_profiles=self.wide.profiles, route_errors=self.route_errors)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.wide.close()
        finally:
            self.base.close()


def install(models, config_path):
    """Validate and warm up all wide-profile boundaries before publishing route."""
    _, engine = validate_config(config_path)
    from trt82_backend import Session
    rec = models.models['rec']
    base = rec.gpu
    if isinstance(base, WideRoute) or base.profiles != {'x': ORIGINAL_PROFILE}:
        raise ValueError('Expected original, unwrapped recognition session')
    if base.errors or base.closed or (rec.cpu and rec.cpu.errors):
        raise RuntimeError('Cannot install over unhealthy recognition session')
    if digest(rec.onnx) != SOURCE_SHA:
        raise ValueError('Recognition source changed')
    wide = Session(engine, ENGINE_SHA, 'rec_wide_fp32')
    try:
        if (wide.profiles != {'x': WIDE_PROFILE}
                or [v.name for v in wide.inputs] != [v.name for v in base.inputs]
                or [v.name for v in wide.outputs] != [v.name for v in base.outputs]
                or any(v != np.dtype('float32') for v in wide.dtypes.values())):
            raise ValueError('Wide recognition interface/profile differs')
        for shape in WIDE_PROFILE:
            wide.run(None, {'x': np.zeros(shape, dtype=np.float32)})
        route = WideRoute(base, wide)
        route.require_healthy()
    except Exception:
        wide.close()
        raise
    rec.gpu = route
    models.warmup_execution = models.stats()
    return route


def close_models(models):
    """Attempt every GPU cleanup even when one session reports a close error."""
    errors = []
    for model in models.models.values():
        try:
            getattr(model, 'gpu', model).close()
        except Exception as err:
            errors.append(type(err).__name__)
    if errors:
        raise RuntimeError('Model cleanup failed: ' + ','.join(errors))


def healthy(models):
    return models.healthy and all(not row.get('closed', False) for row in models.stats().values())
