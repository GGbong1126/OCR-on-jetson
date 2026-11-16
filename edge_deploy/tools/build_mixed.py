"""TensorRT 10.7 conservative FP32-prefix / FP16-allowed-tail experiment.

Build success is NOT numerical validation. Run compare_trt.py before benchmarking.
The original ONNX and existing engines/reports are never overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


BOUNDARY = "p2o.pd_op.batch_norm_.84.0"
COMPUTE_TYPES = {
    "CONVOLUTION", "DECONVOLUTION", "SCALE", "ACTIVATION", "POOLING",
    "ELEMENTWISE", "UNARY", "REDUCE", "MATRIX_MULTIPLY", "SOFTMAX",
    "NORMALIZATION", "RESIZE",
}
DATA_TYPES = {"IDENTITY", "SHUFFLE", "CONCATENATION", "SLICE", "GATHER"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_prefix(layers, boundary):
    """Select through boundary, then close over dependencies (not order alone)."""
    producers = {}
    for index, layer in enumerate(layers):
        for output in layer["outputs"]:
            name = output["name"]
            if name in producers:
                raise ValueError(f"Duplicate parsed tensor name: {name}")
            producers[name] = index
    if boundary not in producers:
        raise ValueError(f"Boundary tensor not found: {boundary}; inspect manifest layers")
    selected = set(range(producers[boundary] + 1))
    pending = list(selected)
    while pending:
        for name in layers[pending.pop()]["inputs"]:
            producer = producers.get(name)
            if producer is not None and producer not in selected:
                selected.add(producer)
                pending.append(producer)
    return selected


def apply_constraints(network, layers, selected, trt):
    for index in sorted(selected):
        entry = layers[index]
        layer = network.get_layer(index)
        outputs = [i for i, out in enumerate(entry["outputs"])
                   if out["floating"] and not out["shape_tensor"]]
        if not outputs:
            entry["constraint_note"] = "No floating execution output; shape/integer path unchanged"
            continue
        kind = entry["type"]
        if kind == "CONSTANT":
            entry["constraint_note"] = "Original constant dtype retained; consumers constrained"
            continue
        if kind not in COMPUTE_TYPES | DATA_TYPES:
            raise ValueError(f"Unreviewed floating layer type {kind}: {entry['name']}")
        if kind in COMPUTE_TYPES:
            layer.precision = trt.float32
            entry["requested_precision"] = "float32"
        for output_index in outputs:
            layer.set_output_type(output_index, trt.float32)
        entry["requested_fp32_output_indices"] = outputs


def snapshot(network, trt):
    layers = []
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        outputs = []
        for i in range(layer.num_outputs):
            tensor = layer.get_output(i)
            if tensor is None:
                raise ValueError(f"Unsupported optional output in layer {layer.name}")
            outputs.append({
                "name": tensor.name, "dtype": str(tensor.dtype),
                "shape": list(tensor.shape), "shape_tensor": bool(tensor.is_shape_tensor),
                "floating": tensor.dtype in (trt.float32, trt.float16),
            })
        layers.append({
            "index": index, "name": layer.name, "type": layer.type.name,
            "inputs": [layer.get_input(i).name for i in range(layer.num_inputs)
                       if layer.get_input(i) is not None],
            "outputs": outputs,
        })
    return layers


def add_fixed_profile(builder, config, input_name, shape):
    """set_shape returns None in TRT 10.7 Python, not a success boolean."""
    shape = tuple(shape)
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, shape, shape, shape)  # Raises on invalid setters.
    if not profile:
        raise ValueError(f"Invalid input profile for {input_name}: {shape}")
    actual = [list(dims) for dims in profile.get_shape(input_name)]
    if actual != [list(shape)] * 3:
        raise ValueError(f"Input profile readback mismatch for {input_name}: {actual}")
    index = config.add_optimization_profile(profile)
    if index < 0:  # Zero is the valid first profile index.
        raise ValueError(f"Could not add input profile for {input_name}")
    return {"index": index, "input_name": input_name, "min_opt_max": actual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--fp32-until-tensor", default=BOUNDARY)
    parser.add_argument("--shape", type=int, nargs=4, default=[1, 3, 960, 704],
                        metavar=("N", "C", "H", "W"))
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--inspect-only", action="store_true",
                        help="Parse, select and constrain layers without building an engine")
    args = parser.parse_args()
    if args.shape[:2] != [1, 3] or any(x <= 0 or x % 32 for x in args.shape[2:]):
        parser.error("Expected shape 1 3 H W with positive H/W divisible by 32")
    if args.workspace_mib <= 0:
        parser.error("workspace-mib must be positive")
    if len({p.resolve() for p in (args.onnx, args.engine, args.report)}) != 3:
        parser.error("ONNX, engine and report must be different paths")
    for path in (args.engine, args.report):
        if path.exists():
            parser.error(f"Refusing to overwrite {path}; use a new experiment name")
        path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "starting", "created_utc": datetime.now(timezone.utc).isoformat(),
        "onnx": str(args.onnx.resolve()), "engine": str(args.engine.resolve()),
        "profile_min_opt_max": args.shape, "workspace_mib": args.workspace_mib,
        "max_aux_streams": 0, "tf32": False, "fp16_allowed": True,
        "precision_constraints": "OBEY", "boundary_tensor": args.fp32_until_tensor,
        "selection": "parsed prefix inclusive + recursive producer closure",
        "note": "Requested constraints, not measured fused-layer precision. No inference or accuracy validation performed.",
    }
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    def save():
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        report["onnx_sha256"] = sha256(args.onnx)
        import tensorrt as trt
        report["tensorrt_version"] = trt.__version__
        if not trt.__version__.startswith("10.7."):
            raise RuntimeError("This experiment targets TensorRT 10.7.x; review APIs for other versions")
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, "")
        builder = trt.Builder(logger)
        network = builder.create_network(0)  # Weakly typed; explicit batch in TRT 10.
        onnx_parser = trt.OnnxParser(network, logger)
        if not onnx_parser.parse_from_file(str(args.onnx.resolve())):
            raise RuntimeError("\n".join(str(onnx_parser.get_error(i))
                                         for i in range(onnx_parser.num_errors)))
        report["layers"] = snapshot(network, trt)
        save()  # Retain parsed tensor names even when boundary matching fails.
        if network.num_inputs != 1 or network.num_outputs != 1:
            raise ValueError("Expected one input and one output")
        inp, out = network.get_input(0), network.get_output(0)
        if inp.dtype != trt.float32 or out.dtype != trt.float32 or inp.is_shape_tensor:
            raise ValueError("Expected FP32 execution input/output")
        selected = select_prefix(report["layers"], args.fp32_until_tensor)
        for i, entry in enumerate(report["layers"]):
            entry["selected_fp32_prefix"] = i in selected
        apply_constraints(network, report["layers"], selected, trt)
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.clear_flag(trt.BuilderFlag.TF32)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib * 1024**2)
        config.max_aux_streams = 0
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        report["verified_profile"] = add_fixed_profile(builder, config, inp.name, args.shape)
        # Keep original FP32 I/O and require linear buffers for trt_backend.py.
        inp.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        out.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        report.update({"input_name": inp.name, "output_name": out.name,
                       "selected_layer_count": len(selected), "parsed_layer_count": network.num_layers,
                       "status": "planned" if args.inspect_only else "building"})
        save()
        print(f"Protected prefix: {len(selected)}/{network.num_layers} parsed layers (includes constants/shape layers)", flush=True)
        print(f"Manifest: {args.report.resolve()}", flush=True)
        if args.inspect_only:
            return
        started = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        report["build_seconds"] = time.perf_counter() - started
        if serialized is None:
            raise RuntimeError("TensorRT build failed; inspect console/build log. Constraints were NOT relaxed.")
        with args.engine.open("xb") as stream:
            stream.write(serialized)
        report.update({"status": "built_unvalidated", "engine_sha256": sha256(args.engine),
                       "engine_bytes": args.engine.stat().st_size})
        save()
        print(f"Saved engine: {args.engine.resolve()}. Run compare_trt.py next; not yet validated.")
    except Exception as error:
        report.update({"status": "failed", "error": str(error)})
        save()
        raise


if __name__ == "__main__":
    main()
