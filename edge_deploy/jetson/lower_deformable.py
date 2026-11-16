"""Prepare derived batch-one layout/cell graphs, preserving all original outputs.

Keep a portable standard-ONNX lowered graph for ORT equivalence testing, then
replace ONLY checked GridSample nodes in a second TensorRT-specific graph.
im_shape and scale_factor remain runtime inputs; never freeze their values.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import onnxruntime as ort


def attributes(n):
    return {a.name: h.get_attribute_value(a) for a in n.attribute}


def lower_norm(model):
    inferred = onnx.shape_inference.infer_shapes(model)
    types = {v.name: v.type.tensor_type for v in
             list(inferred.graph.input)+list(inferred.graph.value_info)+list(inferred.graph.output)}
    nodes, changed = [], []
    for n in model.graph.node:
        if n.op_type != 'LayerNormalization':
            nodes.append(n)
            continue
        a = attributes(n)
        tensor_type = types.get(n.input[0])
        if (a.get('axis', -1) != 2 or len(n.output) != 1 or len(n.input) != 3
                or a.get('stash_type', 1) != 1 or tensor_type is None
                or tensor_type.elem_type != T.FLOAT or len(tensor_type.shape.dim) != 3):
            raise ValueError('Unexpected LayerNormalization contract')
        base = n.name + '_autoocr'
        def op(kind, inputs, suffix, **kw):
            name = base + suffix
            nodes.append(h.make_node(kind, inputs, [name], name=name, **kw))
            return name
        eps = base + '_epsilon'
        model.graph.initializer.append(nh.from_array(np.array(a.get('epsilon', 1e-5), np.float32), eps))
        mean = op('ReduceMean', [n.input[0]], '_mean', axes=[2], keepdims=1)
        center = op('Sub', [n.input[0], mean], '_center')
        sq = op('Mul', [center, center], '_square')
        var = op('ReduceMean', [sq], '_var', axes=[2], keepdims=1)
        denom = op('Sqrt', [op('Add', [var, eps], '_vareps')], '_denom')
        normalized = op('Div', [center, denom], '_normalized')
        scaled = op('Mul', [normalized, n.input[1]], '_scaled')
        nodes.append(h.make_node('Add', [scaled, n.input[2]], list(n.output), name=base+'_bias'))
        changed.append(n.name)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return changed


def lower_mod(model):
    producers = {o: n for n in model.graph.node for o in n.output}
    constants = {i.name: nh.to_array(i) for i in model.graph.initializer}
    nodes, changes = [], []
    for n in model.graph.node:
        if n.op_type != 'Mod':
            nodes.append(n)
            continue
        source = n.input[0]
        while source in producers and producers[source].op_type == 'Identity':
            source = producers[source].input[0]
        topk = producers.get(source)
        divisor = constants.get(n.input[1])
        if (topk is None or topk.op_type != 'TopK' or topk.output[1] != source
                or divisor is None or divisor.size != 1 or int(divisor.item()) not in (1, 25)
                or attributes(n).get('fmod', 0) != 0):
            raise ValueError('Mod is not the expected nonnegative TopK-index remainder')
        # Integer quotient is exact for nonnegative indices (small fixed class/query axes).
        base = n.name + '_autoocr'
        nodes.extend([
            h.make_node('Div', list(n.input), [base+'_q'], name=base+'_div'),
            h.make_node('Mul', [base+'_q', n.input[1]], [base+'_p'], name=base+'_mul'),
            h.make_node('Sub', [n.input[0], base+'_p'], list(n.output), name=base+'_sub')])
        changes.append(n.name)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return changes


def plugin_graph(model):
    count = 0
    for n in model.graph.node:
        if n.op_type == 'GridSample':
            a = attributes(n)
            if a != {'align_corners': 0, 'mode': b'bilinear', 'padding_mode': b'zeros'}:
                raise ValueError('Unsupported GridSample attributes: '+str(a))
            n.op_type = 'AutoOCRGridSample'
            n.domain = 'autoocr'
            del n.attribute[:]
            count += 1
    model.opset_import.append(h.make_opsetid('autoocr', 1))
    return count


def remove_empty_concat(model):
    constants = {i.name: nh.to_array(i) for i in model.graph.initializer}
    changed = []
    for n in model.graph.node:
        if n.op_type != 'Concat': continue
        # Concatenating a constant length-zero vector to a shape vector is identity.
        keep = [v for v in n.input if not (v in constants and constants[v].shape == (0,))]
        if len(keep) != len(n.input):
            if attributes(n).get('axis') != 0 or not keep:
                raise ValueError('Unexpected empty Concat')
            del n.input[:]; n.input.extend(keep)
            if len(keep) == 1:
                n.op_type = 'Identity'; del n.attribute[:]
            changed.append(n.name)
    return changed


def compact_uniform_constants(model):
    initializers, nodes, changed = [], [], []
    for i in model.graph.initializer:
        a = nh.to_array(i)
        if a.nbytes >= 1024**2 and np.all(a == a.flat[0]):
            scalar, shape = i.name+'_scalar', i.name+'_shape'
            initializers.extend([nh.from_array(np.array(a.flat[0],dtype=a.dtype),scalar),
                                 nh.from_array(np.array(a.shape,dtype=np.int64),shape)])
            nodes.append(h.make_node('Expand',[scalar,shape],[i.name],name=i.name+'_compact'))
            changed.append({'name':i.name,'saved_bytes':a.nbytes})
        else: initializers.append(i)
    nodes.extend(model.graph.node)
    del model.graph.node[:]; model.graph.node.extend(nodes)
    del model.graph.initializer[:]; model.graph.initializer.extend(initializers)
    return changed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('Output directory exists')
    a.output.mkdir(parents=True)
    m = onnx.load(str(a.source))
    report = {'source_sha256': hashlib.sha256(a.source.read_bytes()).hexdigest(),
              'original_ops': dict(Counter(n.op_type for n in m.graph.node))}
    if {v.name for v in m.graph.input} != {'image', 'im_shape', 'scale_factor'}:
        raise ValueError('Unexpected inputs')
    for v in m.graph.input:
        v.type.tensor_type.shape.dim[0].dim_value = 1
    report['lowered_norm'] = lower_norm(m)
    # Old exported value_info includes obsolete symbolic batch/shape metadata.
    del m.graph.value_info[:]
    m = onnx.shape_inference.infer_shapes(m, strict_mode=True, data_prop=True)
    temp = a.output / 'batch1.onnx'
    onnx.save(m, str(temp))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    folded = a.output / 'folded.onnx'
    opts.optimized_model_filepath = str(folded)
    session = ort.InferenceSession(str(temp), opts, providers=['CPUExecutionProvider'])
    del session
    m = onnx.load(str(folded))
    report['empty_concat'] = remove_empty_concat(m)
    report['lowered_mod'] = lower_mod(m)
    report['compact_uniform_constants'] = compact_uniform_constants(m)
    onnx.checker.check_model(m)
    portable = a.output / 'portable.onnx'
    onnx.save(m, str(portable))
    report['grid_plugins'] = plugin_graph(m)
    if report['grid_plugins'] != 18: raise ValueError('Expected 18 GridSample nodes')
    candidate = a.output / 'trt.onnx'
    onnx.checker.check_model(m)
    onnx.save(m, str(candidate))
    report['derived_ops'] = dict(Counter(n.op_type for n in m.graph.node))
    report['hashes'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in (portable, candidate)}
    report['status'] = 'prepared_not_execution_validated'
    with (a.output/'conversion.json').open('x') as f: json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
