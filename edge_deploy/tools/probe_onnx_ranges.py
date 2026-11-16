"""Inspect FP32 ranges around BatchNorm parameters beyond FP16 range.

This instruments an in-memory ONNX copy and disables ORT graph optimization
to observe the original graph's values. It does not locate the first failing
TensorRT kernel, reproduce its fusions, or prove the cause of its NaNs.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from compare_trt import tensor_summary
from ocr_det import preprocess, read_image, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="reports/fp16_range_probe.json")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    model = onnx.load(args.onnx)
    weights = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    producers = {out: n for n in model.graph.node for out in n.output}
    probe_names = []
    suspects = []
    for node in model.graph.node:
        if node.op_type != "BatchNormalization" or any(k not in weights for k in node.input[1:5]):
            continue
        params = [weights[k] for k in node.input[1:5]]
        if not any(np.any(np.abs(a) > 65504) or not np.isfinite(a).all() for a in params):
            continue
        names = [node.input[0], node.output[0]]
        upstream = producers.get(node.input[0])
        if upstream is not None and upstream.op_type == "Conv":
            names.insert(0, upstream.input[0])
        suspects.append({"batchnorm": node.name,
                         "upstream": upstream.name if upstream is not None else None,
                         "parameters": {name: tensor_summary(weights[name]) for name in node.input[1:5]},
                         "probe_tensors": names})
        for name in names:
            if name not in probe_names:
                probe_names.append(name)
    result = {
        "model_sha256": sha256_file(args.onnx), "image_sha256": sha256_file(args.image),
        "operator_counts": dict(Counter(n.op_type for n in model.graph.node)),
        "suspects": suspects, "activation_ranges": {},
        "note": "Original FP32 graph with ORT optimizations disabled. Out-of-range values are risk evidence, not proof of the first TensorRT overflow. Large BN constants alone can be harmless if folded. No model file was modified.",
    }
    if probe_names:
        # Retain input/output original contract separately; expose only probes.
        del model.graph.output[:]
        model.graph.output.extend(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
                                  for name in probe_names)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(model.SerializeToString(), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        del model, weights
        tensor = preprocess(read_image(args.image))
        result["input_shape"] = list(tensor.shape)
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("Expected one input")
        outputs = session.run(probe_names, {inputs[0].name: tensor})
        result["activation_ranges"] = {name: tensor_summary(a) for name, a in zip(probe_names, outputs)}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"suspect_layers": [s["batchnorm"] for s in suspects],
                      "activation_ranges": result["activation_ranges"]}, indent=2))
    print(f"Saved: {destination.resolve()}")


if __name__ == "__main__":
    main()
