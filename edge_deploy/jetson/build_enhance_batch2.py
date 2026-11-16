"""E109 independent static batch2 export/build; same FP32 weights, no release edits."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import build_enhance_fp16 as source

q = source.q
ARCH_SHA = '463e6fd3ba27671d09daf1b73786ff8b58055aa3911396679a4dd46d18e29070'
WORKSPACE = 256 * 1024**2


def contract(inputs, outputs):
    if inputs != [('images', [2, 3, 512, 512], 'DataType.FLOAT')] or outputs != [('output', [2, 1, 512, 512], 'DataType.FLOAT')]:
        raise ValueError('Static FP32 batch2 contract required')


def provenance():
    cfg = source.provenance()
    spec = cfg['engines']['enhance']
    if q.digest(q.RELEASE/spec['path']) != spec['sha256'] or q.digest(q.RELEASE/'train_mobilenetv3_321.py') != ARCH_SHA:
        raise ValueError('Original engine/architecture changed')
    return dict(script_sha256=q.digest(__file__), source_helper_sha256=q.digest(source.__file__),
                architecture_sha256=ARCH_SHA, weights_sha256=source.WEIGHTS_SHA,
                original_onnx_sha256=source.SOURCE_SHA, baseline_engine_sha256=spec['sha256'],
                deployment_sha256=source.DEPLOYMENT_SHA,
                plan_sha256=q.digest(q.BASE/'staging/quant_yolo_e92_r1/plan.json'))


def compare(ref, out):
    if ref.shape != out.shape or ref.dtype != np.float32 or out.dtype != np.float32 or not np.isfinite(ref).all() or not np.isfinite(out).all():
        raise ValueError('Invalid synthetic outputs')
    diff = np.abs(ref.astype(np.float64)-out.astype(np.float64))
    return dict(shape=list(out.shape), mae=float(diff.mean()), max_abs_error=float(diff.max()),
                allclose=bool(np.allclose(ref, out, rtol=1e-3, atol=1e-4)))


def export(stage):
    for n in ('export.json', 'enhance_batch2.onnx'):
        if (stage/n).exists(): raise ValueError('Refuse overwrite')
    report = dict(provenance(), status='exporting')
    start = time.perf_counter()
    try:
        import torch
        import onnx
        import onnxruntime as ort
        sys.path.insert(0, str(q.RELEASE))
        from train_mobilenetv3_321 import MobileNetV3_Unet
        weights = list((q.RELEASE/'pth/imgpre').glob('*.pth'))
        if len(weights) != 1 or q.digest(weights[0]) != source.WEIGHTS_SHA:
            raise ValueError('Weights differ')
        torch.set_num_threads(2); torch.manual_seed(109)
        model = MobileNetV3_Unet(pretrained=False).float().cpu().eval()
        checkpoint = torch.load(str(weights[0]), map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint.get('G_model', checkpoint), strict=True)
        del checkpoint
        x = torch.rand(2, 3, 512, 512)
        with torch.inference_mode():
            ref = torch.cat([model(x[i:i+1]) for i in range(2)]).numpy()
            batched = model(x).numpy()
        path = stage/'enhance_batch2.onnx'
        torch.onnx.export(model, (x,), str(path), input_names=['images'], output_names=['output'],
                          opset_version=13, dynamic_axes={}, do_constant_folding=True,
                          dynamo=False, external_data=False)
        onnx.checker.check_model(str(path))
        options = ort.SessionOptions(); options.intra_op_num_threads = 2
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        cpu = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        original = ort.InferenceSession(str(source.SOURCE), sess_options=options, providers=['CPUExecutionProvider'])
        feeds = x.numpy(); out = cpu.run(None, {'images': feeds})[0]
        old = np.concatenate([original.run(None, {'images': feeds[i:i+1]})[0] for i in range(2)])
        swapped = cpu.run(None, {'images': feeds[::-1].copy()})[0][::-1].copy()
        checks = dict(torch_batch_vs_single=compare(ref, batched), ort_vs_torch_batch=compare(batched, out),
                      ort_batch_vs_original_single=compare(old, out), swapped_order=compare(out, swapped))
        report.update(checks=checks, torch=torch.__version__, onnx=onnx.__version__, ort=ort.__version__,
                      opset=13, onnx_sha256=q.digest(path), onnx_bytes=path.stat().st_size,
                      note='CPU synthetic checks only. Same weights/architecture, independent static batch2 export; no real inputs saved or batch metadata-only patch.')
        if not all(v['allclose'] for v in checks.values()): raise ValueError('Synthetic export regression')
        report['status'] = 'exported_synthetic_consistent_not_business_validated'
    except Exception as exc:
        report.update(status='failed', error=type(exc).__name__); raise
    finally:
        report['export_s'] = time.perf_counter()-start; q.save(stage/'export.json', report)
    print(json.dumps(dict(status=report['status'], checks=checks)), flush=True)


def build(stage):
    for n in ('build.json', 'enhance_batch2.engine', 'timing.cache'):
        if (stage/n).exists(): raise ValueError('Refuse overwrite')
    ex = json.loads((stage/'export.json').read_text())
    if ex['status'] != 'exported_synthetic_consistent_not_business_validated' or any(ex[k] != v for k,v in provenance().items()) or q.digest(stage/'enhance_batch2.onnx') != ex['onnx_sha256']:
        raise ValueError('Export gate/provenance differs')
    import tensorrt as trt
    if trt.__version__ != '8.2.1.8+localcp310': raise ValueError('Audited board binding required')
    logger = trt.Logger(trt.Logger.INFO); builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse((stage/'enhance_batch2.onnx').read_bytes()):
        raise RuntimeError('; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    desc = lambda t: (t.name, list(t.shape), str(t.dtype))
    inputs = [desc(network.get_input(i)) for i in range(network.num_inputs)]
    outputs = [desc(network.get_output(i)) for i in range(network.num_outputs)]
    contract(inputs, outputs)
    config = builder.create_builder_config(); config.max_workspace_size = WORKSPACE
    config.avg_timing_iterations = 8
    for flag in (trt.BuilderFlag.FP16, trt.BuilderFlag.INT8, trt.BuilderFlag.TF32): config.clear_flag(flag)
    if int(config.flags) != 0: raise ValueError('Unexpected flags')
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_timing_cache(config.create_timing_cache(b''), False)
    report = dict(provenance(), status='building', inputs=inputs, outputs=outputs,
        export_sha256=q.digest(stage/'export.json'), onnx_sha256=ex['onnx_sha256'],
        workspace_bytes=WORKSPACE, avg_timing_iterations=8, flags=int(config.flags),
        tactic_sources=config.get_tactic_sources(), trt_version=trt.__version__, parsed_layers=network.num_layers)
    start = time.perf_counter()
    try:
        blob = builder.build_serialized_network(network, config)
        if blob is None: raise RuntimeError('No engine returned')
        path = stage/'enhance_batch2.engine'
        with path.open('xb') as f: f.write(bytes(blob))
        with (stage/'timing.cache').open('xb') as f: f.write(bytes(config.get_timing_cache().serialize()))
        report.update(status='built_not_validated', engine_sha256=q.digest(path), engine_bytes=path.stat().st_size)
    except Exception as exc:
        report.update(status='failed', error=type(exc).__name__); raise
    finally:
        report['build_s'] = time.perf_counter()-start; q.save(stage/'build.json', report)
    print(json.dumps({k:report[k] for k in ('status','build_s','engine_bytes','engine_sha256')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=('export','build'))
    p.add_argument('--stage', type=Path, required=True); a=p.parse_args(); stage=a.stage.resolve()
    if platform.machine() != 'aarch64' or not stage.is_relative_to(q.BASE/'staging') or not stage.is_dir(): p.error('Existing isolated board stage required')
    globals()[a.mode](stage)
