"""E108 bounded FP32 workspace trial; original fixed tiles/weights, no release edits."""
import argparse
import json
from pathlib import Path
import platform
import time

import build_enhance_fp16 as source

q = source.q
WORKSPACE = 1024 * 1024**2


def validate_contract(inputs, outputs):
    if inputs != [('images', [1, 3, 512, 512], 'DataType.FLOAT')]:
        raise ValueError('Original fixed input required')
    if outputs != [('output', [1, 1, 512, 512], 'DataType.FLOAT')]:
        raise ValueError('Original fixed output required')


def provenance():
    cfg = source.provenance()
    spec = cfg['engines']['enhance']
    if q.digest(q.RELEASE / spec['path']) != spec['sha256']:
        raise ValueError('Original engine changed')
    return dict(script_sha256=q.digest(__file__), source_helper_sha256=q.digest(source.__file__),
                onnx_sha256=source.SOURCE_SHA, weights_sha256=source.WEIGHTS_SHA,
                baseline_engine_sha256=spec['sha256'], deployment_sha256=source.DEPLOYMENT_SHA,
                plan_sha256=q.digest(q.BASE / 'staging/quant_yolo_e92_r1/plan.json'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, type=Path)
    a = p.parse_args(); stage = a.stage.resolve()
    if platform.machine() != 'aarch64' or not stage.is_relative_to(q.BASE/'staging') or not stage.is_dir():
        p.error('Existing isolated board stage required')
    if any((stage/n).exists() for n in ('build.json', 'enhance_fp32.engine', 'timing.cache')):
        raise ValueError('Refuse to overwrite experiment')
    report = dict(provenance(), status='building')
    import tensorrt as trt
    if trt.__version__ != '8.2.1.8+localcp310':
        raise ValueError('Expected audited board binding')
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(source.SOURCE.read_bytes()):
        raise RuntimeError('; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    def tensor(t): return (t.name, list(t.shape), str(t.dtype))
    inputs = [tensor(network.get_input(i)) for i in range(network.num_inputs)]
    outputs = [tensor(network.get_output(i)) for i in range(network.num_outputs)]
    validate_contract(inputs, outputs)
    config = builder.create_builder_config()
    config.max_workspace_size = WORKSPACE
    config.avg_timing_iterations = 8
    for flag in (trt.BuilderFlag.FP16, trt.BuilderFlag.INT8, trt.BuilderFlag.TF32):
        config.clear_flag(flag)
    if int(config.flags) != 0:
        raise ValueError('Unexpected precision flags')
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_timing_cache(config.create_timing_cache(b''), False)
    report.update(inputs=inputs, outputs=outputs, workspace_bytes=config.max_workspace_size,
                  flags=int(config.flags), avg_timing_iterations=config.avg_timing_iterations,
                  tactic_sources=config.get_tactic_sources(), trt_version=trt.__version__,
                  parsed_layers=network.num_layers,
                  note='FP32/noTF32, same graph and batch1 tiles. Workspace 256->1024MiB, fresh timing cache and detailed inspector. Rebuild variation is not isolated workspace causation; no speed or memory saving assumed.')
    start = time.perf_counter()
    try:
        blob = builder.build_serialized_network(network, config)
        if blob is None: raise RuntimeError('No engine returned')
        engine = stage/'enhance_fp32.engine'
        with engine.open('xb') as f: f.write(bytes(blob))
        with (stage/'timing.cache').open('xb') as f:
            f.write(bytes(config.get_timing_cache().serialize()))
        report.update(status='built_not_validated', engine_sha256=q.digest(engine), engine_bytes=engine.stat().st_size)
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        report['build_s'] = time.perf_counter()-start
        q.save(stage/'build.json', report)
    print(json.dumps({k:report[k] for k in ('status','engine_bytes','engine_sha256','build_s')}), flush=True)


if __name__ == '__main__': main()
