"""M1b: isolated unchanged-ONNX batch2 audit/build/check; no HTTP or NER.

Reuse the release's actual _det_run via the existing AST-reference technique.
Images, input tensors, probability maps and document text stay in RAM only.
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import quantize_yolo_ptq as q

PROFILE = [(2, 3, 32, 32), (2, 3, 960, 704), (2, 3, 960, 960)]
SOURCE = q.RELEASE / 'pth/onnx_rec/ocr_det.onnx'
SOURCE_SHA = 'b4413116303d0b3fcf6266dfb1ac5ce6b8e0b6f8955fbf9b9a2f08e402f810b2'


def tensor_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def validate(x):
    if (not isinstance(x, np.ndarray) or x.dtype != np.float32 or x.ndim != 4
            or x.shape[0] not in (1, 2) or x.shape[1] != 3
            or any(d < 32 or d > 960 or d % 32 for d in x.shape[2:])
            or not np.isfinite(x).all()):
        raise ValueError('Expected finite FP32 [1/2,3,H,W], aligned H/W in 32..960')


def group_pairs(rows):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Nonempty, unique task IDs required')
    buckets = defaultdict(list)
    for index, row in enumerate(rows):
        shape = row['input_shape']
        if len(shape) != 4 or shape[0] != 1:
            raise ValueError('Plan must describe single-sample inputs')
        buckets[tuple(shape[1:])].append(index)
    return [indices[i:i+2] for indices in buckets.values() for i in range(0, len(indices), 2)]


def route(single, batched, x):
    validate(x)
    if single.errors or batched.errors or single.closed or batched.closed:
        raise RuntimeError('Unhealthy GPU route; no execution-error fallback')
    output = (single if len(x) == 1 else batched).run(None, {'x': x})[0]
    if output.shape != (len(x), 1, *x.shape[2:]) or output.dtype != np.float32 or not np.isfinite(output).all():
        raise ValueError('Invalid batch output contract')
    return output


def legacy_method():
    path = q.RELEASE / 'server.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_ONNXOCREngine')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_det_run')
    namespace = dict(cv2=cv2, np=np, math=math)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_det_run']


def legacy(method, image, output=None, expected=None):
    captured = []
    class Replay:
        def get_inputs(self):
            return [SimpleNamespace(name='x')]
        def run(self, names, feeds):
            x = feeds['x']
            validate(x)
            if expected is not None:
                np.testing.assert_array_equal(x, expected)
            captured.append(x.copy())
            return [np.zeros((1, 1, *x.shape[2:]), np.float32) if output is None else output]
    boxes = method(SimpleNamespace(det_session=Replay()), image)
    if len(captured) != 1:
        raise ValueError('Expected exactly one original det call per image')
    return captured[0], boxes


def fingerprints():
    cfgpath = q.RELEASE / 'edge_deploy/jetson/deployment.json'
    cfg = json.loads(cfgpath.read_text())
    paths = [cfgpath, SOURCE, q.RELEASE/'server.py', q.RELEASE/'jetson_source_manifest.json']
    for row in cfg['engines'].values():
        path = q.RELEASE/row['path']
        if q.digest(path) != row['sha256']:
            raise ValueError('Original engine changed')
        paths.append(path)
        for plugin in row.get('plugins', []):
            path = q.RELEASE/plugin['path']
            if q.digest(path) != plugin['sha256']:
                raise ValueError('Original plugin changed')
            paths.append(path)
    if q.digest(SOURCE) != SOURCE_SHA or cfg['engines']['det']['source_sha256'] != SOURCE_SHA:
        raise ValueError('Original detection ONNX changed')
    return {str(p.relative_to(q.RELEASE)): q.digest(p) for p in sorted(set(paths))}


def inputs(plan, method):
    result = []
    for row in plan['rows']:
        path = Path(plan['images'])/row['file']
        if path.name != row['file'] or q.digest(path) != row['id']:
            raise ValueError('Frozen input changed')
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Software image decode failed')
        tensor, _ = legacy(method, image)
        if list(tensor.shape) != row['input_shape'] or tensor_sha(tensor) != row['tensor_sha256']:
            raise ValueError('Original preprocessing changed')
        result.append((image, tensor))
    return result


def prepare(stage):
    import onnx
    prior = q.load_plan(q.BASE/'staging/quant_yolo_e92_r1')
    method = legacy_method()
    model = onnx.load(str(SOURCE)); onnx.checker.check_model(model)
    dims = lambda value: [d.dim_value or d.dim_param for d in value.type.tensor_type.shape.dim]
    graph = dict(inputs=[(v.name, dims(v)) for v in model.graph.input],
                 outputs=[(v.name, dims(v)) for v in model.graph.output],
                 operators=dict(Counter(n.op_type for n in model.graph.node)),
                 reshape_or_flatten=[n.name for n in model.graph.node if n.op_type in ('Reshape','Flatten')])
    if len(model.graph.input) != 1 or model.graph.input[0].name != 'x' or isinstance(dims(model.graph.input[0])[0], int):
        raise ValueError('Unmodified symbolic batch input required')
    rows = []
    for row in prior['calibration'] + prior['holdout']:
        path = Path(prior['images'])/row['file']
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Input decode failed')
        tensor, _ = legacy(method, image)
        rows.append(dict(id=row['sha256'], file=row['file'], image_hw=list(image.shape[:2]),
                         input_shape=list(tensor.shape), tensor_sha256=tensor_sha(tensor)))
    plan = dict(rows=rows, images=prior['images'], profile=PROFILE, graph=graph,
                fingerprints=fingerprints(), script_sha256=q.digest(__file__),
                helper_sha256=q.digest(q.__file__), workspace_bytes=256*1024**2,
                min_available_build_kib=8*1024**2, min_free_disk_bytes=2*1024**3,
                threads=2, cv2_threads=1, warmup=3, repeat=10,
                note='Original thirteen files; unchanged software decode and actual det method. '
                     'Standalone det inputs, not full business thumbnail/enhancement or GT/HTTP. '
                     'No model/metadata rewrite, no padding/resize to pair, no saved pixels/text. '
                     'Numeric allclose is diagnostic; exact original boxes form this narrow gate.')
    plan['groups'] = group_pairs(rows)
    if not any(len(g) == 2 for g in plan['groups']):
        raise ValueError('No distinct real-image compatible pair')
    q.save(stage/'plan.json', plan)
    print(json.dumps(dict(status='plan_frozen', shapes=dict(Counter(str(r['input_shape']) for r in rows)), groups=plan['groups'])), flush=True)


def load_plan(stage):
    plan = json.loads((stage/'plan.json').read_text())
    if (plan['fingerprints'] != fingerprints() or plan['script_sha256'] != q.digest(__file__)
            or plan['helper_sha256'] != q.digest(q.__file__) or plan['profile'] != [list(v) for v in PROFILE]
            or group_pairs(plan['rows']) != plan['groups']):
        raise ValueError('Frozen source/plan differs')
    return plan


def compare(a, b):
    if a.shape != b.shape or a.dtype != np.float32 or b.dtype != np.float32 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid finite output pair')
    difference = np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(allclose=bool(np.allclose(a,b,rtol=1e-3,atol=1e-4)),
                mae=float(difference.mean()), max_abs_error=float(difference.max()))


def require(stage, name, status):
    value = json.loads((stage/name).read_text())
    if value['status'] != status or value['plan_sha256'] != q.digest(stage/'plan.json'):
        raise ValueError('Prior gate did not pass')
    return value


def build(stage, plan):
    import tensorrt as trt
    require(stage, 'cpu_check.json', 'cpu_batch_independent')
    if q.memory()['MemAvailable_kib'] < plan['min_available_build_kib'] or shutil.disk_usage(stage).free < plan['min_free_disk_bytes']:
        raise RuntimeError('Resource reserve failed')
    logger = trt.Logger(trt.Logger.INFO); builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(SOURCE.read_bytes()):
        raise RuntimeError('; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if not trt.__version__.startswith('8.2.1') or network.num_inputs != 1 or network.num_outputs != 1:
        raise ValueError('Expected TRT8.2 and one input/output')
    cfg = builder.create_builder_config(); cfg.max_workspace_size = plan['workspace_bytes']
    for flag in (trt.BuilderFlag.FP16, trt.BuilderFlag.INT8, trt.BuilderFlag.TF32): cfg.clear_flag(flag)
    if int(cfg.flags) != 0: raise ValueError('Unexpected precision flags')
    cfg.avg_timing_iterations = 8; cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    profile = builder.create_optimization_profile(); profile.set_shape('x', *PROFILE)
    if [tuple(s) for s in profile.get_shape('x')] != PROFILE: raise ValueError('Profile differs')
    cfg.add_optimization_profile(profile)
    start = time.perf_counter(); blob = builder.build_serialized_network(network,cfg)
    if blob is None: raise RuntimeError('No engine returned')
    with (stage/'det_batch2.engine').open('xb') as handle: handle.write(bytes(blob))
    return dict(status='built_not_validated', engine_sha256=q.digest(stage/'det_batch2.engine'),
                engine_bytes=(stage/'det_batch2.engine').stat().st_size, build_s=time.perf_counter()-start,
                flags=int(cfg.flags), workspace_bytes=cfg.max_workspace_size, trt_version=trt.__version__,
                parsed_layers=network.num_layers, avg_timing_iterations=8)


def check(stage, plan, gpu):
    method = legacy_method(); data = inputs(plan, method)
    single = batched = None; rows = []; groups = []
    try:
        if gpu:
            from trt82_backend import Session
            built = require(stage,'build.json','built_not_validated')
            spec = json.loads((q.RELEASE/'edge_deploy/jetson/deployment.json').read_text())['engines']['det']
            single = Session(q.RELEASE/spec['path'], spec['sha256'], 'det_n1_control')
            batched = Session(stage/'det_batch2.engine', built['engine_sha256'], 'det_n2_candidate')
            run = lambda x: route(single, batched, x)
            reference = lambda x: single.run(None,{'x':x})[0]
        else:
            import onnxruntime as ort
            options=ort.SessionOptions(); options.intra_op_num_threads=2
            options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
            single=ort.InferenceSession(str(SOURCE),sess_options=options,providers=['CPUExecutionProvider'])
            run = reference = lambda x: single.run(None,{'x':x})[0]
        for indices in plan['groups']:
            x=np.concatenate([data[i][1] for i in indices]); validate(x)
            y=run(x); repeated=run(x)
            swapped=run(x[::-1].copy())[::-1].copy() if len(indices)==2 else y
            swap=compare(y,swapped); repeat_exact=bool(np.array_equal(y,repeated))
            groups.append(dict(indices=indices,batch=len(indices),swap=swap,repeat_exact=repeat_exact))
            if not swap['allclose'] or not repeat_exact: raise ValueError('Slot/repeat independence failed')
            for slot,index in enumerate(indices):
                image,tensor=data[index]; ref=reference(tensor)
                value=y[slot:slot+1].copy(); numeric=compare(ref,value)
                _,boxes_a=legacy(method,image,ref,tensor)
                _,boxes_b=legacy(method,image,value,tensor)
                boxes_equal=len(boxes_a)==len(boxes_b) and all(np.array_equal(a,b) for a,b in zip(boxes_a,boxes_b))
                rows.append(dict(index=index,id=plan['rows'][index]['id'],batch=len(indices),slot=slot,
                                 input_sha256=tensor_sha(tensor),numeric=numeric,
                                 box_counts=[len(boxes_a),len(boxes_b)],boxes_equal=boxes_equal))
            print('Compared group',indices,flush=True)
        passed=all(r['boxes_equal'] for r in rows)
        if not gpu: passed=passed and all(r['numeric']['allclose'] for r in rows)
        status=('gpu_det_boxes_consistent_not_full_business_validated' if gpu else 'cpu_batch_independent') if passed else 'completed_with_differences_not_accepted'
        result=dict(status=status,rows=rows,groups=groups,numeric_allclose_count=sum(r['numeric']['allclose'] for r in rows),
                    boxes_equal_count=sum(r['boxes_equal'] for r in rows),
                    note='Slot/repeat and original det boxes only; no rec/layout/fields/HTTP/GT or speedup measured.')
        if gpu:
            result.update(single=single.stats(),batched=batched.stats())
            inspector=batched.engine.create_engine_inspector();inspector.execution_context=batched.context
            info=json.loads(inspector.get_engine_information(batched.trt.LayerInformationFormat.JSON))
            q.save(stage/'layers.json',info)
            result['inspector_layers']=len(info.get('Layers',[]))
            if result['inspector_layers']!=batched.engine.num_layers:raise ValueError('Incomplete inspector')
        return result
    finally:
        if gpu:
            for session in (batched,single):
                if session is not None:session.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','cpu','build','check'))
    parser.add_argument('--stage',type=Path,required=True)
    args=parser.parse_args();stage=args.stage.resolve()
    if platform.machine()!='aarch64' or not stage.is_relative_to(q.BASE/'staging') or not stage.is_dir():
        parser.error('Existing isolated Jetson project stage required')
    state=subprocess.check_output(['systemctl','show','autoocr-jetson-dev.service','--property=ActiveState'],text=True)
    if state.strip()!='ActiveState=inactive':raise RuntimeError('Original service must remain stopped')
    cv2.setNumThreads(1)
    if args.mode=='prepare':prepare(stage);return
    plan=load_plan(stage)
    target=stage/({'cpu':'cpu_check.json','build':'build.json','check':'check.json'}[args.mode])
    if target.exists() or (args.mode=='build' and (stage/'det_batch2.engine').exists()):raise FileExistsError('Preserve earlier evidence')
    report=dict(status='running',plan_sha256=q.digest(stage/'plan.json'),memory_before=q.memory())
    try:
        report.update(build(stage,plan) if args.mode=='build' else check(stage,plan,args.mode=='check'))
    except Exception as exc:
        report.update(status='failed',error=type(exc).__name__);raise
    finally:
        report['memory_after']=q.memory()
        report['original_fingerprints_unchanged']=fingerprints()==plan['fingerprints']
        q.save(target,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','groups')},sort_keys=True),flush=True)


if __name__=='__main__':main()
