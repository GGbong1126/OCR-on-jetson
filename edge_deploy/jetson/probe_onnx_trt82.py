"""Parse original ONNX models with Jetson TensorRT 8.2; do not alter/build them."""
import argparse
import hashlib
import json
from pathlib import Path
import platform


def main():
    import tensorrt as trt
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Report exists; choose a new path')
    if platform.machine() != 'aarch64' or not trt.__version__.startswith('8.2.1'):
        raise RuntimeError('Expected Jetson aarch64 TensorRT 8.2.1')
    report = {'status': 'running', 'tensorrt': trt.__version__, 'models': [],
              'note': 'Parser check only; no engine build, inference or accuracy validation.'}
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    try:
        for path in args.models:
            data = path.read_bytes()
            with trt.Builder(logger) as builder:
                with builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network:
                    with trt.OnnxParser(network, logger) as parser:
                        ok = parser.parse(data)
                        row = {'name': path.name, 'sha256': hashlib.sha256(data).hexdigest(),
                               'bytes': len(data), 'parsed': bool(ok),
                               'errors': [str(parser.get_error(i)) for i in range(parser.num_errors)]}
                        if ok:
                            row['layers'] = network.num_layers
                            row['inputs'] = [{'name': network.get_input(i).name,
                                             'shape': list(network.get_input(i).shape)}
                                            for i in range(network.num_inputs)]
                            row['outputs'] = [{'name': network.get_output(i).name,
                                              'shape': list(network.get_output(i).shape)}
                                             for i in range(network.num_outputs)]
                        report['models'].append(row)
                        print(path.name + ': ' + ('parsed' if ok else 'FAILED'), flush=True)
        report['status'] = ('all_parsed_not_execution_validated'
                            if all(r['parsed'] for r in report['models']) else 'parser_failure')
    except Exception as exc:
        report['status'] = 'execution_error'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if report['status'] != 'all_parsed_not_execution_validated':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
