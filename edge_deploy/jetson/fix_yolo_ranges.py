"""Lower only proven integer shape-derived YOLO anchor ranges for TRT8.2.

Range(0.0, float(Shape[axis]), 1.0) -> Cast(Range(0, Shape[axis], 1), FLOAT).
No weights, boxes, thresholds or padding policy change. Keep original ONNX.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort


def lower(model):
    producers = {v: n for n in model.graph.node for v in n.output}
    result, names = [], []
    for node in model.graph.node:
        if node.op_type != 'Range':
            result.append(node)
            continue
        for name, expected in ((node.input[0], 0), (node.input[2], 1)):
            constant = producers[name]
            if constant.op_type != 'Constant':
                raise ValueError('Range endpoints not constant')
            value = numpy_helper.to_array(next(a.t for a in constant.attribute if a.name == 'value'))
            if value.shape != () or value.dtype != np.float32 or value.item() != expected:
                raise ValueError('Only float32 zero/one shape ranges are supported')
        cast = producers[node.input[1]]
        if cast.op_type != 'Cast' or next(a.i for a in cast.attribute if a.name == 'to') != TensorProto.FLOAT:
            raise ValueError('Range limit not float cast')
        gather = producers[cast.input[0]]
        if gather.op_type != 'Gather' or producers[gather.input[0]].op_type != 'Shape':
            raise ValueError('Range limit not a shape dimension')
        prefix = node.name + '_trt82'
        inputs = []
        for i, old in enumerate((node.input[0], cast.input[0], node.input[2])):
            new = prefix + '_i' + str(i)
            if i in (0, 2):
                # TRT8.2 shape symbolic evaluation cannot always fold FLOAT->INT
                # constants here. Supply genuine INT32 constants, not a cast.
                result.append(helper.make_node('Constant', [], [new], name=new,
                    value=numpy_helper.from_array(np.array(0 if i == 0 else 1, dtype=np.int32))))
            else:
                result.append(helper.make_node('Cast', [old], [new], name=new, to=TensorProto.INT32))
            inputs.append(new)
        result.append(helper.make_node('Range', inputs, [prefix + '_out'], name=prefix))
        result.append(helper.make_node('Cast', [prefix + '_out'], list(node.output),
                                       name=prefix + '_float', to=TensorProto.FLOAT))
        names.append(node.name)
    if not names:
        raise ValueError('No matching ranges')
    del model.graph.node[:]
    model.graph.node.extend(result)
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Output exists')
    args.output.mkdir(parents=True)
    model = onnx.load(str(args.source))
    changed = lower(model)
    onnx.checker.check_model(model)
    target = args.output / 'yolo.onnx'
    onnx.save(model, str(target))
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 2
    sessions = [ort.InferenceSession(str(p), sess_options=opt, providers=['CPUExecutionProvider'])
                for p in (args.source, target)]
    rng, rows = np.random.default_rng(0), []
    for height, width in ((640, 640), (640, 448), (448, 640), (64, 64)):
        tensor = rng.random((1, 3, height, width), dtype=np.float32)
        a, b = [s.run(None, {'images': tensor})[0] for s in sessions]
        if not np.isfinite(a).all() or not np.isfinite(b).all() or a.shape != b.shape:
            raise ValueError('Invalid comparison')
        row = {'input_shape': list(tensor.shape), 'output_shape': list(a.shape),
               'exactly_equal': bool(np.array_equal(a, b)),
               'max_abs_error': float(np.max(np.abs(a - b)))}
        rows.append(row)
        if not row['exactly_equal']:
            raise ValueError('Expected exact equality for integer anchor-range lowering')
    report = {'status': 'cpu_four_shape_exact_not_task_validated', 'changed_range_nodes': changed,
              'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
              'onnx_sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'comparisons': rows}
    with (args.output / 'lowering_report.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
