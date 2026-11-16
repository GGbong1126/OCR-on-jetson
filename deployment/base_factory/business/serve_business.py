"""Full /OCR and /Extract WSGI factory plus explicit terminal activation.

No license bypass or fork-preloaded models. Optional native TRT OCR; one sync worker.
"""
import argparse
import getpass
import importlib
import os
from pathlib import Path
import sys


def load_server():
    source = Path(os.environ['AUTOOCR_SOURCE_ROOT']).resolve()
    runtime = Path(os.environ['AUTOOCR_RUNTIME_DIR']).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    os.chdir(runtime)
    os.environ.setdefault('AUTOOCR_BASE_DIR', str(source))
    os.environ.setdefault('AUTOOCR_LICENSE_FILE', str(runtime / 'license.dat'))
    os.environ.setdefault('AUTOOCR_LOG_FILE', str(runtime / 'server.log'))
    os.environ.setdefault('AUTOOCR_TEMP_DIR', str(runtime / 'temp'))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('YOLO_CONFIG_DIR', str(runtime / 'ultralytics'))
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    sys.path.insert(0, str(source))
    return importlib.import_module('server')


def ensure_license(server):
    if not server.validate_license():
        raise RuntimeError('Valid WSL license required. Run serve_business.py --activate in a terminal first.')


def activate(server):
    if server.validate_license():
        print('Existing license validated.')
        return 0
    if not sys.stdin.isatty():
        raise RuntimeError('Activation requires an interactive terminal; no GUI or automatic bypass.')
    code = server.generate_auth_code()
    print('Send this device request code to your authorized license administrator (do not put it in reports):')
    print(code)
    activation = getpass.getpass('Activation code (hidden input): ').strip()
    if not server.verify_activation_code(code, activation):
        raise RuntimeError('Activation verification failed; no license written.')
    # Same verification and persistence path as check_license; retain current counter behavior.
    server.create_license_file(code, server.get_device_hash())
    server.initialize_counter_file()
    ensure_license(server)
    print('Activation complete. License stays in the runtime directory.')
    return 0


def create_app():
    backend = os.environ.get('AUTOOCR_OCR_BACKEND', 'cpu')
    if backend not in ('cpu', 'trt_fp32'):
        raise RuntimeError('Unsupported AUTOOCR_OCR_BACKEND')
    other_backend = os.environ.get('AUTOOCR_OTHER_BACKEND', 'cpu')
    if other_backend not in ('cpu', 'cuda') or (other_backend == 'cuda' and backend != 'trt_fp32'):
        raise RuntimeError('CUDA business models require the verified TRT OCR deployment')
    server = load_server()
    ensure_license(server)
    if server.OCRPipeline.FORCE_PADDLE:
        raise RuntimeError('This deployment requires the existing ONNX business path, not FORCE_PADDLE.')
    cfg = server.cfg
    paths = (cfg.ONNX_LAYOUT_MODEL, cfg.ONNX_CELL_MODEL, cfg.ONNX_OCR_DET_MODEL, cfg.ONNX_OCR_REC_MODEL)
    if not all(Path(p).is_file() for p in paths):
        raise RuntimeError('All four business ONNX files are required; no implicit Paddle fallback.')
    server.cv2.setNumThreads(1)
    server.torch.set_num_threads(2)
    gpu_models = None
    if other_backend == 'cuda':
        from gpu_models import GPUModels
        gpu_models = GPUModels(cfg)
        pipeline = server.OCRPipeline(use_gpu=True, onnx_session_factory=gpu_models.session_factory,
                                      imgpre_fp16=False, imgpre_batch=1)
    else:
        pipeline = server.OCRPipeline(use_gpu=False)
    pipeline.load_models(layout_dir=cfg.LAYOUT_MODEL_DIR, cell_dir=cfg.TABLE_CELL_MODEL_DIR,
                         ocr_dir=cfg.OCR_MODEL_DIR, ner_dir=cfg.NER_MODEL_DIR,
                         imgpre_dir=cfg.IMGPRE_MODEL_DIR, yolo_conf=cfg.YOLO_CONF)
    # Existing loader catches optional model errors. Full-service bootstrap refuses silent loss.
    if not all((pipeline._use_onnx_layout, pipeline._use_onnx_cell, pipeline._use_onnx_ocr,
                pipeline._yolo_model is not None, pipeline._ner_pipeline is not None,
                pipeline._imgpre_model is not None)):
        raise RuntimeError('Full-service model readiness failed; inspect runtime/server.log.')
    trt_ocr = None
    if backend == 'trt_fp32':
        from trt_sessions import attach
        trt_ocr = attach(pipeline._ocr_engine, os.environ['AUTOOCR_SOURCE_ROOT'])
        pipeline._ocr_engine = trt_ocr
    if gpu_models is not None:
        gpu_models.attach(pipeline)
        # Original optional-stage catches must not turn GPU errors into HTTP success.
        @server.app_online.before_request
        def require_gpu_health():
            if not gpu_models.healthy:
                return {'status': 'error', 'message': 'GPU models unhealthy'}, 503

        @server.app_online.after_request
        def reject_gpu_partial(response):
            if not gpu_models.healthy:
                return server.app_online.make_response(({'status': 'error', 'message': 'GPU models unhealthy'}, 503))
            return response
    server.initialize_counter_file()
    server.pipeline = pipeline
    server.app_online.config['MAX_CONTENT_LENGTH'] = 24 * 1024 * 1024

    @server.app_online.get('/healthz')
    def healthz():
        body = {'status': 'ready', 'scope': 'full_business', 'ocr_backend': 'onnxruntime_cpu',
                'layout': True, 'yolo': True, 'cell': True, 'ner': True, 'image_enhancement': True}
        if trt_ocr is not None:
            from trt_sessions import BACKEND
            body.update(ocr_backend=BACKEND, ocr_execution=trt_ocr.stats(),
                        other_models_backend='cpu', out_of_profile_policy='onnxruntime_cpu_no_resize')
            if gpu_models is not None:
                body.update(other_models_backend='cuda', models_execution=gpu_models.stats())
            if not trt_ocr.healthy:
                body['status'] = 'error'
                return body, 503
        return body

    return server.app_online


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--check-license', action='store_true')
    args = parser.parse_args()
    if args.activate == args.check_license:
        parser.error('Choose exactly one of --activate or --check-license')
    server = load_server()
    if args.activate:
        return activate(server)
    ensure_license(server)
    print('License validated; models not loaded.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
