"""Isolated Jetson TRT8.2 INT8 PTQ trial; never changes a release or license.

prepare fixes a hash-disjoint calibration/holdout split before engine building.
build uses real calibration images and FP32 fallback (not FP16). A separate
inspect step executes once before exporting TRT8.2 dynamic-layer evidence.
evaluate changes only YOLO in the original offline business pipeline.
benchmark measures synchronized raw YOLO calls, NOT HTTP or end-to-end speed.
Reports deliberately omit recognized text and input tensors.
"""
import argparse
from collections import Counter
import contextlib
import ctypes as ct
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from types import MethodType

import numpy as np


BASE = Path('/home/xchq/work/autoocr')
RELEASE = BASE / 'releases/jetson-r2'
SOURCE_HASH = 'c72a55c86ae798d4cf7d85bbe11dd8bfe17cb1ec1d2d70be23ecf3c82dfb01f3'
PROFILE = [(1, 3, 64, 64), (1, 3, 640, 640), (1, 3, 640, 640)]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def business_rgb(path):
    from PIL import Image
    with Image.open(path) as original:
        im = original.convert('RGB')
        im.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        im.save(buffer, format='JPEG', quality=90)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.array(decoded.convert('RGB'))


def tensor_from_rgb(rgb, calibration=False):
    from ultralytics.data.augment import LetterBox
    # Calibration must use the profile OPT shape. Runtime retains original auto
    # stride64 padding; neither step stretches image aspect ratio or alters text.
    padded = LetterBox((640, 640), auto=not calibration, stride=64)(image=rgb)
    value = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    if calibration and value.shape != (1, 3, 640, 640):
        raise ValueError('Calibration shape must equal OPT')
    return value


def split_rows(rows, count):
    if not 1 <= count < len(rows):
        raise ValueError('Need nonempty disjoint calibration and holdout sets')
    for key in ('sha256', 'business_pixels_sha256'):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError('Duplicate source or preprocessed pixels; split manually by group')
    ordered = sorted(rows, key=lambda r: r['sha256'])
    return ordered[:count], ordered[count:]


def load_plan(stage):
    plan = json.loads((stage / 'plan.json').read_text(encoding='utf-8'))
    for group in ('calibration', 'holdout'):
        for row in plan[group]:
            path = Path(plan['images']) / row['file']
            if Path(row['file']).name != row['file'] or digest(path) != row['sha256']:
                raise ValueError('Dataset path/hash changed')
    all_rows = plan['calibration'] + plan['holdout']
    cal, test = split_rows(all_rows, len(plan['calibration']))
    if cal != plan['calibration'] or test != plan['holdout']:
        raise ValueError('Split changed')
    config_path = RELEASE / 'edge_deploy/jetson/deployment.json'
    if digest(config_path) != plan['deployment_sha256']:
        raise ValueError('Baseline deployment changed')
    return plan


def prepare(a):
    paths = sorted(p for p in a.images.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    rows = []
    for p in paths:
        pixels = business_rgb(p)
        rows.append({'file': p.name, 'sha256': digest(p), 'business_hw': list(pixels.shape[:2]),
                     'business_pixels_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()})
    cal, test = split_rows(rows, a.calibration_count)
    plan = {'images': str(a.images.resolve()), 'calibration': cal, 'holdout': test,
            'split_rule': 'ascending source sha256, first N calibration, rest holdout; exact pixel duplicates rejected',
            'deployment_sha256': digest(RELEASE / 'edge_deploy/jetson/deployment.json'),
            'onnx_sha256': SOURCE_HASH, 'profile': PROFILE,
            'calibration_preprocess': 'original business RGB thumbnail1500/LANCZOS/JPEG90, aspect-preserving square LetterBox640/pad114, RGB CHW /255',
            'runtime_preprocess': 'unchanged business + Ultralytics auto stride64 LetterBox',
            'limitations': 'Tiny convenience sample; no GT or document-group independence established. No augmented copies or holdout used for calibration.',
            'gate': 'Finite outputs, zero backend errors, all holdout ordinary/enhanced original text/Markdown/NER/field values strictly equal. No tolerance relaxation.'}
    save(a.stage / 'plan.json', plan)
    print(json.dumps({'status': 'split_frozen', 'calibration': len(cal), 'holdout': len(test)}), flush=True)


def build(a):
    import tensorrt as trt
    from trt82_backend import Cuda
    plan = load_plan(a.stage)
    if digest(a.onnx) != plan['onnx_sha256'] or shutil.disk_usage(a.stage).free < 400 * 1024**2:
        raise ValueError('Source hash differs or insufficient disk reserve')
    engine_path = a.stage / 'yolo_int8_fp32.engine'
    cache = a.stage / 'calibration.cache'
    for path in (engine_path, cache, a.stage / 'build.json', a.stage / 'layers.json'):
        if path.exists():
            raise ValueError('Refuse to replace experiment artifact')
    log = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(log)
    if not builder.platform_has_fast_int8:
        raise RuntimeError('No native INT8 support')
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, log)
    if not parser.parse(a.onnx.read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if network.num_inputs != 1 or network.get_input(0).name != 'images':
        raise ValueError('Unexpected model contract')
    config = builder.create_builder_config()
    config.max_workspace_size = 512 * 1024**2
    config.set_flag(trt.BuilderFlag.INT8)
    config.clear_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    profile = builder.create_optimization_profile()
    profile.set_shape('images', *PROFILE)
    if [tuple(x) for x in profile.get_shape('images')] != PROFILE:
        raise RuntimeError('Profile mismatch')
    config.add_optimization_profile(profile)
    calibration_profile = builder.create_optimization_profile()
    calibration_profile.set_shape('images', PROFILE[1], PROFILE[1], PROFILE[1])
    config.set_calibration_profile(calibration_profile)

    class Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.cuda, self.pointer = Cuda(), ct.c_void_p()
            self.cuda.call('cudaMalloc', ct.byref(self.pointer), int(np.prod(PROFILE[1])) * 4)
            self.index, self.error = 0, None
            self.tensor_hashes = []

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            try:
                if list(names) != ['images']:
                    raise ValueError('Unexpected calibration binding')
                if self.index == len(plan['calibration']):
                    return None
                row = plan['calibration'][self.index]
                rgb = business_rgb(Path(plan['images']) / row['file'])
                value = tensor_from_rgb(rgb, calibration=True)
                if not np.isfinite(value).all():
                    raise ValueError('Nonfinite calibration input')
                self.cuda.call('cudaMemcpy', self.pointer, ct.c_void_p(value.ctypes.data), value.nbytes, 1)
                self.tensor_hashes.append(hashlib.sha256(value.tobytes()).hexdigest())
                self.index += 1
                print('Calibrated %d/%d' % (self.index, len(plan['calibration'])), flush=True)
                return [self.pointer.value]
            except Exception as e:
                self.error = type(e).__name__ + ': ' + str(e)
                raise

        def read_calibration_cache(self):
            return None  # Always calibrate; no accidental cache reuse across data/models.

        def write_calibration_cache(self, data):
            with cache.open('xb') as f:
                f.write(bytes(data))

        def close(self):
            self.cuda.call('cudaFree', self.pointer)

    calibrator = Calibrator()
    config.int8_calibrator = calibrator
    report = {'status': 'building', 'precision_policy': 'INT8 enabled; FP32 fallback; FP16/TF32 disabled',
              'plan_sha256': digest(a.stage / 'plan.json'), 'onnx_sha256': digest(a.onnx),
              'trt_version': trt.__version__, 'workspace_bytes': config.max_workspace_size}
    start = time.perf_counter()
    try:
        serialized = builder.build_serialized_network(network, config)
        if serialized is None or calibrator.error or calibrator.index != len(plan['calibration']):
            raise RuntimeError('Build/calibration failed: ' + str(calibrator.error))
        with engine_path.open('xb') as f:
            f.write(bytes(serialized))
        # On TRT8.2, a dynamic Myelin context can report an internal slot error
        # if inspected before its first execution. Keep build and execution audit
        # separate: a serialized engine alone is not proof of working INT8.
        report.update(status='built_pending_execution_inspection', engine_sha256=digest(engine_path),
                      engine_bytes=engine_path.stat().st_size, calibration_batches=calibrator.index,
                      calibration_tensor_sha256=calibrator.tensor_hashes, cache_sha256=digest(cache))
    except Exception as e:
        report.update(status='failed', error=type(e).__name__ + ': ' + str(e))
        raise
    finally:
        calibrator.close()
        report['build_seconds'] = time.perf_counter() - start
        save(a.stage / 'build.json', report)
        print(json.dumps(report), flush=True)


def candidate(a):
    from trt82_backend import Session
    build_report = json.loads((a.stage / 'build.json').read_text())
    inspection_path = a.stage / 'inspection.json'
    if inspection_path.exists():
        checked = json.loads(inspection_path.read_text())
        if (checked['status'] != 'executed_int8_evidence_not_quality_accepted'
                or checked['build_report_sha256'] != digest(a.stage / 'build.json')):
            raise ValueError('Invalid separate inspection')
        build_report = checked
    elif build_report['status'] != 'built_int8_evidence_not_quality_accepted':
        raise ValueError('No successful audited INT8 build/inspection')
    if build_report['plan_sha256'] != digest(a.stage / 'plan.json'):
        raise ValueError('Plan was modified after build')

    class TrialSession(Session):
        def stats(self):
            row = super().stats()
            row['precision'] = 'int8_enabled_fp32_fallback'
            return row

    return TrialSession(a.stage / 'yolo_int8_fp32.engine', build_report['engine_sha256'], 'yolo_int8_candidate')


def inspect(a):
    """TRT8.2 dynamic Myelin inspector may need one execution before inspection.

    Preserve the original build failure report; audit the same engine separately.
    Only a calibration image is used here, never the holdout.
    """
    from trt82_backend import Session
    plan = load_plan(a.stage)
    target = a.stage / 'inspection.json'
    layer_path = a.stage / 'layers_after_execution.json'
    if target.exists() or layer_path.exists():
        raise ValueError('Inspection exists')
    built = json.loads((a.stage / 'build.json').read_text())
    if built['onnx_sha256'] != SOURCE_HASH or built['plan_sha256'] != digest(a.stage / 'plan.json'):
        raise ValueError('Build provenance mismatch')
    engine = a.stage / 'yolo_int8_fp32.engine'
    if built.get('engine_sha256') and built['engine_sha256'] != digest(engine):
        raise ValueError('Engine changed after build')
    session = Session(engine, digest(engine), 'candidate_inspection')
    report = {'status': 'running', 'build_report_sha256': digest(a.stage / 'build.json'),
              'plan_sha256': digest(a.stage / 'plan.json'), 'engine_sha256': digest(engine),
              'engine_bytes': engine.stat().st_size, 'cache_sha256': digest(a.stage / 'calibration.cache'),
              'note': 'Separate post-execution inspector audit; original build/inspector failure retained.'}
    try:
        row = plan['calibration'][0]
        x = tensor_from_rgb(business_rgb(Path(plan['images']) / row['file']), True)
        y = session.run(None, {'images': x})
        if len(y) != 1 or y[0].shape != (1, 15, 2100) or y[0].dtype != np.float32:
            raise ValueError('Candidate output contract differs')
        inspector = session.engine.create_engine_inspector()
        inspector.execution_context = session.context
        data = json.loads(inspector.get_engine_information(session.trt.LayerInformationFormat.JSON))
        layers = data.get('Layers', [])
        count = sum('int8' in json.dumps(row).lower() for row in layers)
        if not count or len(layers) != session.engine.num_layers:
            raise ValueError('Incomplete layer/INT8 evidence')
        save(layer_path, data)
        report.update(status='executed_int8_evidence_not_quality_accepted', layer_count=len(layers),
                      layers_with_int8_format=count, output_shape=list(y[0].shape), errors=session.errors,
                      context_device_memory_bytes=session.engine.device_memory_size,
                      calibration_tensor_sha256=hashlib.sha256(x.tobytes()).hexdigest())
    except Exception as e:
        report.update(status='failed', error=type(e).__name__ + ': ' + str(e))
        raise
    finally:
        session.close()
        save(target, report)
    print(json.dumps(report), flush=True)


def strict_compare(ref, out):
    keys = ('FullText', 'Text', 'MarkdownBodyLines', 'NER', 'special')
    eq = {k: ref[k] == out[k] for k in keys}
    eq['raw_line_texts'] = [r['text'] for r in ref['RawLines']] == [r['text'] for r in out['RawLines']]
    return {'passed': all(eq.values()), 'strict_equal': eq,
            'box_count_reference': len(ref['Box']), 'box_count_candidate': len(out['Box']),
            'boxes_exactly_equal': ref['Box'] == out['Box']}


def field_diagnostics(ref, out, special_text):
    """Explain structural drift, without changing the original acceptance gate.

    special_text must be the unchanged business _layout_special_text helper.
    Only booleans/counts leave this function; no document text is recorded.
    """
    def groups(value, nonempty):
        return [(row['Class'], row['Content']) for row in value['Text']
                if not nonempty or any(row['Content'])]

    return {
        'api_special_text_equal': {key: special_text(ref['special'], key) == special_text(out['special'], key)
                                   for key in ('title', 'category_no', 'page_no')},
        'special_item_counts': {key: [len(ref['special'].get(key, [])), len(out['special'].get(key, []))]
                                for key in ('title', 'category_no', 'page_no')},
        'text_class_content_equal_without_number': groups(ref, False) == groups(out, False),
        'nonempty_text_class_content_equal_without_number': groups(ref, True) == groups(out, True),
        'box_class_counts_reference': dict(Counter(row['Class'] for row in ref['Box'])),
        'box_class_counts_candidate': dict(Counter(row['Class'] for row in out['Box'])),
        'note': 'Diagnostic only; original strict gate unchanged. Existing business field formatting, no new normalization.'}


def evaluate(a):
    import torch
    from serve_jetson import deployment, load_models
    plan = load_plan(a.stage)
    target = a.stage / ('evaluation_details.json' if a.details else 'evaluation.json')
    if target.exists():
        raise ValueError('Report exists')
    report = {'status': 'running', 'rows': [], 'plan_sha256': digest(a.stage / 'plan.json'),
              'note': 'Holdout only; all other six models fixed TRT FP32. Strict text equality is not GT accuracy. Single passes are not benchmark.'}
    with open(os.devnull, 'w') as quiet:
        with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            server, root, cfg = deployment()
            pipeline, models = load_models(server, root, cfg)
        baseline = models.models['yolo']
        trial = candidate(a)
        sessions = {'reference': baseline, 'candidate': trial}
        active = 'reference'
        calls = []

        def forward(_model, x, augment=False, profile=False, visualize=False, embed=None):
            if augment or visualize or embed is not None:
                raise ValueError('Unsupported YOLO mode')
            value = x.detach().cpu().numpy()
            calls.append({'shape': list(value.shape), 'sha256': hashlib.sha256(value.tobytes()).hexdigest()})
            return torch.from_numpy(sessions[active].run(None, {'images': value})[0])

        pipeline._yolo_model.model.forward = MethodType(forward, pipeline._yolo_model.model)
        try:
            for index, image in enumerate(plan['holdout']):
                for image_type in (3, 1):
                    results, input_rows = [], []
                    for active in ('reference', 'candidate'):
                        models.models['yolo'] = sessions[active]
                        calls.clear()
                        with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                            result = pipeline.process(str(Path(plan['images']) / image['file']),
                                                      image_type=image_type, run_ner=True)
                        if not result['success'] or not models.healthy:
                            raise RuntimeError('Pipeline error; refuse partial success')
                        results.append(result['data'])
                        input_rows.append(list(calls))
                    row = strict_compare(*results)
                    if a.details:
                        row['field_diagnostics'] = field_diagnostics(*results, server._layout_special_text)
                    row.update(image_sha256=image['sha256'], image_type=image_type,
                               yolo_inputs_equal=input_rows[0] == input_rows[1], yolo_inputs=input_rows[0])
                    if not input_rows[0] or not row['yolo_inputs_equal']:
                        raise RuntimeError('YOLO not called with identical inputs')
                    report['rows'].append(row)
                    print(json.dumps({'holdout': index + 1, 'image_type': image_type, 'passed': row['passed']}), flush=True)
            report['status'] = ('small_holdout_strict_consistent_not_gt_validated' if all(r['passed'] for r in report['rows'])
                                else 'text_or_field_changes_not_accepted')
            report['yolo_execution'] = {k: v.stats() for k, v in sessions.items()}
            report['other_execution'] = {k: v for k, v in models.stats().items() if k != 'yolo'}
        except Exception as e:
            report.update(status='failed', error=type(e).__name__ + ': ' + str(e))
            raise
        finally:
            save(target, report)
            trial.close()
    print(json.dumps({'status': report['status'], 'cases': len(report['rows']),
                      'passed': sum(r['passed'] for r in report['rows'])}), flush=True)


def memory():
    row = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('NvMapMemUsed', 'MemAvailable', 'SwapFree'):
            row[key + '_kib'] = int(value.split()[0])
    return row


def benchmark(a):
    from trt82_backend import Session
    plan = load_plan(a.stage)
    target = a.stage / ('benchmark_' + a.variant + '_' + a.run_note + '.json')
    if target.exists():
        raise ValueError('Report exists')
    feeds = [{'images': tensor_from_rgb(business_rgb(Path(plan['images']) / r['file']))} for r in plan['holdout']]
    config = json.loads((RELEASE / 'edge_deploy/jetson/deployment.json').read_text())
    before = memory()
    native = (candidate(a) if a.variant == 'candidate' else
              Session(RELEASE / config['engines']['yolo']['path'], config['engines']['yolo']['sha256'], 'yolo_fp32_reference'))
    report = {'status': 'running', 'variant': a.variant, 'run_note': a.run_note, 'before': before,
              'warmup_cycles': a.warmup, 'measured_cycles': a.repeat, 'rows': [],
              'input_shapes': [list(x['images'].shape) for x in feeds],
              'input_sha256': [hashlib.sha256(x['images'].tobytes()).hexdigest() for x in feeds],
              'plan_sha256': digest(a.stage / 'plan.json'),
              'note': 'Raw Session.run wall includes packing/allocation/H2D/execute/sync/D2H/finite checks, excludes image decode/pre/post, single batch/serial. No GPU-kernel or full-service benchmark. Separate process per variant; no clocks changed.'}
    try:
        for _ in range(a.warmup):
            for feed in feeds:
                native.run(None, feed)
        report['after_warmup'] = memory()
        for cycle in range(a.repeat):
            for index, feed in enumerate(feeds):
                start = time.perf_counter()
                out = native.run(None, feed)
                report['rows'].append({'cycle': cycle, 'input_index': index, 'wall_ms': (time.perf_counter() - start) * 1000,
                                       'output_shape': list(out[0].shape)})
        values = [r['wall_ms'] for r in report['rows']]
        report.update(status='finite_calls_measured_not_quality_acceptance', mean_ms=float(np.mean(values)),
                      p50_ms=float(np.percentile(values, 50)), p95_ms=float(np.percentile(values, 95)),
                      after_measurement=memory(), execution=native.stats(),
                      context_bytes=native.engine.device_memory_size,
                      io_buffer_bytes=sum(v[1] for v in native.buffers.values()))
    except Exception as e:
        report.update(status='failed', error=type(e).__name__ + ': ' + str(e))
        raise
    finally:
        native.close()
        save(target, report)
    print(json.dumps({k: report[k] for k in ('status', 'variant', 'mean_ms', 'p50_ms', 'p95_ms', 'before', 'after_warmup', 'context_bytes')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'build', 'inspect', 'evaluate', 'benchmark'))
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--images', type=Path, default=BASE / 'testpic')
    p.add_argument('--onnx', type=Path, default=BASE / 'staging/yolo_onnx_r3/yolo.onnx')
    p.add_argument('--calibration-count', type=int, default=8)
    p.add_argument('--variant', choices=('reference', 'candidate'), default='reference')
    p.add_argument('--run-note', choices=('r1', 'r2'), default='r1')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeat', type=int, default=20)
    p.add_argument('--details', action='store_true', help='Additional field/structure diagnostics, saved separately')
    a = p.parse_args()
    a.stage = a.stage.resolve()
    if platform.machine() != 'aarch64' or not a.stage.is_relative_to(BASE / 'staging') or not a.stage.is_dir():
        p.error('Use an existing isolated experiment directory under board staging')
    if a.warmup < 1 or a.repeat < 1:
        p.error('Positive warmup/repeat required')
    os.environ['AUTOOCR_SOURCE_ROOT'] = str(RELEASE)
    os.environ['AUTOOCR_RUNTIME_DIR'] = str(BASE / 'runtime')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(RELEASE / 'edge_deploy/jetson'))
    globals()[a.mode](a)


if __name__ == '__main__':
    main()
