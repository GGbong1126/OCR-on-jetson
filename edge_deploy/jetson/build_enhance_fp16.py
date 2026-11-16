"""E101 isolated fixed-shape enhancement FP16 build; never edits a release."""
import argparse
import json
from pathlib import Path
import platform
import time

import quantize_yolo_ptq as q

SOURCE = q.BASE / 'staging/enhance_onnx_r1/enhance.onnx'
SOURCE_SHA = '647444586c2444f646e899c0774c14dcf81bd21db8c5cca37e958636a1c01661'
WEIGHTS_SHA = '4310deb4cd00f99710a8a7bf4faf004394a344c5360b14ff6d4f81a391c1b4b3'
DEPLOYMENT_SHA = '2ced56d7d35e429199d51282eaaf961d1408bf2bc0f2c5784fcfb2b0fecba6d8'
INPUT_SHAPE = (1, 3, 512, 512)
OUTPUT_SHAPE = (1, 1, 512, 512)


def provenance():
    cfg_path = q.RELEASE / 'edge_deploy/jetson/deployment.json'
    cfg = json.loads(cfg_path.read_text())
    exported = json.loads(SOURCE.with_name('export_report.json').read_text())
    if (q.digest(cfg_path) != DEPLOYMENT_SHA or q.digest(SOURCE) != SOURCE_SHA
            or exported['onnx_sha256'] != SOURCE_SHA or exported['source_sha256'] != WEIGHTS_SHA
            or cfg['engines']['enhance']['source_sha256'] != WEIGHTS_SHA):
        raise ValueError('Original enhancement provenance changed')
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, required=True)
    a = p.parse_args()
    a.stage = a.stage.resolve()
    if (platform.machine() != 'aarch64' or not a.stage.is_relative_to(q.BASE / 'staging')
            or not a.stage.is_dir()):
        p.error('Existing board project staging directory required')
    target, engine_path = a.stage / 'build.json', a.stage / 'enhance_fp16.engine'
    if target.exists() or engine_path.exists() or (a.stage / 'timing.cache').exists():
        raise ValueError('Refuse to overwrite an experiment')
    cfg = provenance()
    import tensorrt as trt
    if not trt.__version__.startswith('8.2.1'):
        raise ValueError('Expected audited TensorRT8.2.1')
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    if not builder.platform_has_fast_fp16:
        raise RuntimeError('Native FP16 unavailable')
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(SOURCE.read_bytes()):
        raise RuntimeError('; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if (network.num_inputs != 1 or network.num_outputs != 1
            or network.get_input(0).name != 'images' or tuple(network.get_input(0).shape) != INPUT_SHAPE
            or tuple(network.get_output(0).shape) != OUTPUT_SHAPE
            or network.get_input(0).dtype != trt.float32 or network.get_output(0).dtype != trt.float32):
        raise ValueError('Unexpected enhancement contract')
    config = builder.create_builder_config()
    config.max_workspace_size = 256 * 1024**2
    config.avg_timing_iterations = 8
    config.set_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.clear_flag(trt.BuilderFlag.INT8)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cache = config.create_timing_cache(b'')
    config.set_timing_cache(cache, False)
    report = {'status': 'building', 'script_sha256': q.digest(__file__), 'onnx_sha256': SOURCE_SHA,
              'weights_sha256': WEIGHTS_SHA, 'deployment_sha256': DEPLOYMENT_SHA,
              'baseline_engine_sha256': cfg['engines']['enhance']['sha256'],
              'input_shape': INPUT_SHAPE, 'output_shape': OUTPUT_SHAPE, 'trt_version': trt.__version__,
              'parsed_layers': network.num_layers, 'workspace_bytes': config.max_workspace_size,
              'avg_timing_iterations': config.avg_timing_iterations, 'flags': int(config.flags),
              'tactic_sources': config.get_tactic_sources(),
              'note': 'FP16 enabled, FP32 fallback/I-O, TF32/INT8 disabled, no pruning or calibration. Same workspace256MiB/avgTiming8 as deployed FP32; DETAILED inspection differs.'}
    start = time.perf_counter()
    try:
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            raise RuntimeError('Enhancement engine build failed')
        with engine_path.open('xb') as f:
            f.write(bytes(blob))
        with (a.stage / 'timing.cache').open('xb') as f:
            f.write(bytes(config.get_timing_cache().serialize()))
        report.update(status='built_not_validated', engine_sha256=q.digest(engine_path),
                      engine_bytes=engine_path.stat().st_size)
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        report['build_s'] = time.perf_counter() - start
        q.save(target, report)
    print(json.dumps({k: report[k] for k in ('status', 'engine_bytes', 'engine_sha256', 'build_s')}), flush=True)


if __name__ == '__main__':
    main()
