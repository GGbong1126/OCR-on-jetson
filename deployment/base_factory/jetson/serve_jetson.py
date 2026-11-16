"""Jetson-only full server factory; original hardware activation/counters remain."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent


def deployment():
    root = Path(os.environ['AUTOOCR_SOURCE_ROOT']).resolve()
    if platform.machine() != 'aarch64' or platform.system() != 'Linux':
        raise RuntimeError('This deployment is for the audited Jetson only')
    if root not in (Path('/home/xchq/work/autoocr/releases/jetson-r1'),
                    Path('/home/xchq/work/autoocr/releases/jetson-r2')):
        raise ValueError('Unexpected release root')
    config = json.loads((HERE / 'deployment.json').read_text(encoding='utf-8'))
    if config['schema_version'] != 1 or config['platform'] != 'L4T32.7.1_TRT8.2.1_CUDA10.2':
        raise ValueError('Unexpected configuration')
    from trt82_backend import sha256
    manifest = json.loads((root / 'jetson_source_manifest.json').read_text(encoding='utf-8'))
    for name, row in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != row['sha256']:
            raise ValueError('Source asset changed: ' + name)
    for relative, expected in config['adapter_sha256'].items():
        if Path(relative).name != relative or sha256(HERE / relative) != expected:
            raise ValueError('Deployment adapter changed')
    for row in config['runtime_libraries']:
        if sha256(row['path']) != row['sha256']:
            raise ValueError('Audited runtime library changed')
    sys.path.insert(0, str(root / 'edge_deploy/business'))
    from serve_business import load_server
    return load_server(), root, config


def load_models(server, root, config):
    from jetson_models import JetsonModels
    server.cv2.setNumThreads(1)
    server.torch.set_num_threads(2)
    if server.OCRPipeline.FORCE_PADDLE:
        raise RuntimeError('Paddle fallback is not part of the audited deployment')
    models = JetsonModels(root, server.cfg, config)
    pipeline = server.OCRPipeline(use_gpu=False, onnx_session_factory=models.session_factory,
                                  imgpre_fp16=False, imgpre_batch=1)
    cfg = server.cfg
    pipeline.load_models(layout_dir=cfg.LAYOUT_MODEL_DIR, cell_dir=cfg.TABLE_CELL_MODEL_DIR,
                         ocr_dir=cfg.OCR_MODEL_DIR, ner_dir=cfg.NER_MODEL_DIR,
                         imgpre_dir=cfg.IMGPRE_MODEL_DIR, yolo_conf=cfg.YOLO_CONF)
    if not all((pipeline._use_onnx_layout,pipeline._use_onnx_cell,pipeline._use_onnx_ocr,
                pipeline._yolo_model is not None,pipeline._ner_pipeline is not None,
                pipeline._imgpre_model is not None)):
        raise RuntimeError('Full business model readiness failed')
    models.attach(pipeline)
    models.warmup(pipeline)
    return pipeline, models


def create_app():
    server, root, config = deployment()
    if not server.validate_license():
        raise RuntimeError('Jetson hardware activation required; no models or HTTP started')
    pipeline, models = load_models(server, root, config)
    server.initialize_counter_file()
    server.pipeline = pipeline
    app = server.app_online
    app.config['MAX_CONTENT_LENGTH'] = 24 * 1024 * 1024

    @app.before_request
    def require_healthy():
        if not models.healthy:
            return {'status': 'error','message':'Model backend unhealthy'}, 503

    @app.after_request
    def reject_partial(response):
        if not models.healthy:
            return app.make_response(({'status':'error','message':'Model backend unhealthy'},503))
        return response

    @app.get('/healthz')
    def healthz():
        body = {'status':'ready' if models.healthy else 'error','scope':'full_business',
                'deployment':root.name,'ocr_backend':'tensorrt82_fp32_with_ort_fallback',
                'layout':True,'cell':True,'yolo':True,'ner':True,'image_enhancement':True,
                'models_execution':models.stats(), 'warmup_execution':models.warmup_execution,
                'cpu_modules':config['cpu_modules'],
                'out_of_profile_policy':'onnxruntime_cpu_original_tensor_no_resize'}
        return body, 200 if models.healthy else 503
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-license',action='store_true')
    mode.add_argument('--activate',action='store_true')
    mode.add_argument('--probe-models',type=Path,
                      help='Synthetic model checks only; does not expose HTTP, extract fields or touch counters')
    args = ap.parse_args()
    server, root, config = deployment()
    if args.probe_models:
        if args.probe_models.exists():
            ap.error('Report exists')
        _, models = load_models(server,root,config)
        result = {'status':'synthetic_models_ready_not_http_validated', 'models':models.stats()}
        with args.probe_models.open('x',encoding='utf-8') as f:
            json.dump(result,f,indent=2)
        print(json.dumps(result,indent=2))
    elif args.activate:
        from serve_business import activate
        activate(server)
    elif not server.validate_license():
        raise RuntimeError('Jetson hardware activation required')
    else:
        print('Existing Jetson license validated; counters unchanged.')


if __name__ == '__main__':
    main()
