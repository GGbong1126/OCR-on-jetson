"""Parse/build derived FP32 models with the project GridSample plugin on TRT8.2."""
import argparse
import ctypes
import json
from pathlib import Path
from time import perf_counter

from trt82_backend import sha256


def build(model, plugin, engine=None):
    import tensorrt as trt
    if not trt.__version__.startswith('8.2.1'): raise RuntimeError('TRT8.2.1 required')
    handle = ctypes.CDLL(str(plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, '')
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    start = perf_counter()
    if not parser.parse(model.read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    info = {'onnx_sha256': sha256(model), 'plugin_sha256': sha256(plugin),
            'layers': network.num_layers,
            'inputs': {network.get_input(i).name: list(network.get_input(i).shape) for i in range(network.num_inputs)},
            'outputs': {network.get_output(i).name: list(network.get_output(i).shape) for i in range(network.num_outputs)}}
    if engine:
        if engine.exists(): raise ValueError('Engine exists')
        config = builder.create_builder_config()
        config.max_workspace_size = 512 * 1024**2
        config.clear_flag(trt.BuilderFlag.TF32)
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            item = network.get_input(i)
            if any(d < 1 for d in item.shape): raise ValueError('Expected fixed business input shapes')
            profile.set_shape(item.name, item.shape, item.shape, item.shape)
            if [list(s) for s in profile.get_shape(item.name)] != [list(item.shape)] * 3:
                raise ValueError('Cannot set profile')
        config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None: raise RuntimeError('Engine build failed')
        with engine.open('xb') as f: f.write(bytes(serialized))
        info['engine_sha256'] = sha256(engine)
    info['elapsed_s'] = perf_counter() - start
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--plugin', type=Path, required=True)
    p.add_argument('--engine', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('Report exists')
    report = {'status': 'running'}
    try:
        report.update(build(a.model, a.plugin, a.engine))
        report['status'] = 'built_not_execution_validated' if a.engine else 'parsed_not_built'
    except Exception as e:
        report.update(status='failed', error=str(e))
        raise
    finally:
        with a.output.open('x') as f: json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
