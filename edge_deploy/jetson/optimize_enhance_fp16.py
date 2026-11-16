"""E101 enhancement-only trial: same tiles and business postprocessing.

No image/text/tensor export, licensing, HTTP or release edits. Numeric tile
comparison is diagnostic; unchanged downstream text/fields is the quality gate.
"""
import argparse
import contextlib
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time
import unicodedata

import numpy as np
import build_enhance_fp16 as build
import optimize_rec_fp16 as common
import quantize_yolo_ptq as q
from diagnose_rec_difference import private_edit_summary


def metadata(stage):
    cfg = build.provenance()
    b = json.loads((stage / 'build.json').read_text())
    if (b['status'] != 'built_not_validated' or b['onnx_sha256'] != build.SOURCE_SHA
            or b['baseline_engine_sha256'] != cfg['engines']['enhance']['sha256']
            or b['script_sha256'] != q.digest(build.__file__)):
        raise ValueError('Build/source provenance mismatch')
    return {'script_sha256': q.digest(__file__), 'build_sha256': q.digest(stage / 'build.json'),
            'deployment_sha256': build.DEPLOYMENT_SHA, 'helper_sha256': q.digest(common.__file__),
            'comparison_helper_sha256': q.digest(q.__file__),
            'engine_sha256': b['engine_sha256'], 'onnx_sha256': build.SOURCE_SHA}


def candidate(stage):
    from trt82_backend import Session
    m = metadata(stage)
    class Trial(Session):
        def stats(self):
            return dict(super().stats(), precision='fp16_enabled_fp32_fallback_no_tf32')
    return Trial(stage / 'enhance_fp16.engine', m['engine_sha256'], 'enhance_candidate')


def compare_arrays(ref, out):
    if (ref.shape != build.OUTPUT_SHAPE or out.shape != ref.shape
            or ref.dtype != np.float32 or out.dtype != np.float32
            or not np.isfinite(ref).all() or not np.isfinite(out).all()):
        raise ValueError('Invalid enhancement outputs')
    diff = np.abs(ref.astype(np.float64) - out.astype(np.float64))
    return {'mae': float(diff.mean()), 'max_abs_error': float(diff.max()),
            'p99_abs_error': float(np.percentile(diff, 99)),
            'allclose_diagnostic': bool(np.allclose(ref, out, rtol=1e-3, atol=1e-4)),
            'raw_tile_global_mask_disagreement': int(np.count_nonzero(((1-ref) > .45) != ((1-out) > .45))),
            'reference_range': [float(ref.min()), float(ref.max())],
            'candidate_range': [float(out.min()), float(out.max())]}


def verify_usage(kind, ref_hashes, trial_hashes):
    if kind == 1:
        if not ref_hashes or ref_hashes != trial_hashes:
            raise ValueError('Enhancement bypassed or original tile inputs changed')
    elif kind == 3:
        if ref_hashes or trial_hashes:
            raise ValueError('Ordinary path unexpectedly called enhancement')
    else:
        raise ValueError('Unsupported test image type')


def text_diagnostics(ref, out):
    def protected(text):
        return ''.join(c for c in text if unicodedata.category(c)[0] in 'NPS')
    return {'number_symbol_sequence_equal': protected(ref) == protected(out),
            'edit_categories': private_edit_summary(ref, out),
            'note': 'Diagnostic only, not field assignment or a relaxed acceptance gate. No text exported.'}


def quality_ok(rows):
    enhanced = [r for r in rows if r['image_type'] == 1]
    ordinary = [r for r in rows if r['image_type'] == 3]
    return (len(rows) == 14 and len(enhanced) == 13 and len(ordinary) == 1
            and len({r['image_sha256'] for r in enhanced}) == 13
            and all(r['passed'] and r['same_original_inputs'] for r in rows))


def business_passed(strict_equal, field_diagnostics):
    # User accepts geometric differences, not text/field changes. Preserve the
    # full special-object comparison separately because it also includes bboxes.
    keys = ('FullText', 'Text', 'MarkdownBodyLines', 'NER', 'raw_line_texts')
    return (all(strict_equal[k] for k in keys)
            and all(field_diagnostics['api_special_text_equal'][k]
                    for k in ('title', 'category_no', 'page_no')))


def check(a):
    target = a.stage / 'check.json'
    common.fresh(target); common.fresh(a.stage / 'layers.json')
    plan = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
    report = dict(metadata(a.stage), status='running', rows=[],
                  note='First 3 hash-ordered E92 calibration images, enhanced path. Same real tiles, numeric/mask diagnostics only; not final merged mask, GT or benchmark.')
    models = trial = original = base = None
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = common.setup(a)
            base = models.models['enhance']
            original = base.run
            trial = candidate(a.stage)
            active = None
            def shadow(names, feeds):
                ref = original(names, feeds)[0]
                out = trial.run(names, feeds)[0]
                report['rows'].append(dict(compare_arrays(ref, out), image_sha256=active,
                    input_shape=list(feeds['images'].shape),
                    input_sha256=hashlib.sha256(feeds['images'].tobytes()).hexdigest()))
                return [ref]
            base.run = shadow
            for row in plan['calibration'][:3]:
                active = row['sha256']
                before = len(report['rows'])
                with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                    out = pipeline.process(str(Path(plan['images']) / row['file']), image_type=1, run_ner=True)
                if not out['success'] or not models.healthy or trial.errors or len(report['rows']) == before:
                    raise RuntimeError('Original enhanced pipeline or trial failed')
                print(json.dumps({'images_checked': len({r['image_sha256'] for r in report['rows']}),
                                  'tiles_checked': len(report['rows'])}), flush=True)
            inspector = trial.engine.create_engine_inspector()
            inspector.execution_context = trial.context
            layers = json.loads(inspector.get_engine_information(trial.trt.LayerInformationFormat.JSON))
            count = sum('fp16' in json.dumps(r).lower() for r in layers.get('Layers', []))
            if not count or len(layers['Layers']) != trial.engine.num_layers:
                raise RuntimeError('Incomplete FP16 execution inspection')
            q.save(a.stage / 'layers.json', layers)
            report.update(status='finite_tiles_compared_not_quality_accepted',
                          inspection={'layers': len(layers['Layers']), 'half_format_layers': count},
                          candidate_execution=trial.stats(), original_execution=models.stats())
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        try:
            if base is not None and original is not None:
                base.run = original
            if trial is not None:
                trial.close()
            common.close_models(models)
        finally:
            q.save(target, report)
    print(json.dumps({'status': report['status'], 'tiles': len(report['rows']), 'inspection': report['inspection']}), flush=True)


def checked(stage):
    c = json.loads((stage / 'check.json').read_text())
    if c['status'] != 'finite_tiles_compared_not_quality_accepted' or c['build_sha256'] != q.digest(stage / 'build.json'):
        raise ValueError('Real-tile numeric execution check required')


def evaluate(a):
    checked(a.stage)
    target = a.stage / 'evaluation.json'
    common.fresh(target)
    plan = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
    images = plan['calibration'] + plan['holdout']
    report = dict(metadata(a.stage), status='running', rows=[],
                  note='13 existing images enhanced + one ordinary negative control, paired order alternates. All other six models remain FP32. These images were seen in previous model trials; not independent GT accuracy. Timings diagnostic, not benchmark.')
    models = trial = original = base = None
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = common.setup(a)
            base, trial = models.models['enhance'], candidate(a.stage)
            original = base.run
            active, hashes = 'reference', None
            def routed(names, feeds):
                hashes[active].append(hashlib.sha256(feeds['images'].tobytes()).hexdigest())
                return original(names, feeds) if active == 'reference' else trial.run(names, feeds)
            base.run = routed
            for image, kind in [(images[0], 3)] + [(r, 1) for r in images]:
                hashes = {'reference': [], 'candidate': []}
                outputs, wall = {}, {}
                order = ('reference', 'candidate') if len(report['rows']) % 2 == 0 else ('candidate', 'reference')
                for active in order:
                    start = time.perf_counter()
                    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                        out = pipeline.process(str(Path(plan['images']) / image['file']), image_type=kind, run_ner=True)
                    wall[active] = (time.perf_counter()-start)*1000
                    if not out['success'] or not models.healthy or trial.errors:
                        raise RuntimeError('Full pipeline or candidate failed')
                    outputs[active] = out['data']
                verify_usage(kind, hashes['reference'], hashes['candidate'])
                row = q.strict_compare(outputs['reference'], outputs['candidate'])
                row.update(image_sha256=image['sha256'], image_type=kind, order=order, wall_ms=wall,
                    same_original_inputs=True, tile_input_sha256=hashes['reference'],
                    tile_calls=len(hashes['reference']),
                    field_diagnostics=q.field_diagnostics(outputs['reference'], outputs['candidate'], server._layout_special_text),
                    full_text_diagnostics=text_diagnostics(outputs['reference']['FullText'], outputs['candidate']['FullText']))
                row['strict_objects_passed'] = row['passed']
                row['passed'] = business_passed(row['strict_equal'], row['field_diagnostics'])
                report['rows'].append(row)
                print(json.dumps({'case': len(report['rows']), 'image_type': kind,
                                  'tile_calls_per_branch': row['tile_calls'], 'passed': row['passed']}), flush=True)
            report.update(status='all_cases_consistent_not_gt' if quality_ok(report['rows']) else 'differences_not_accepted',
                          candidate_execution=trial.stats(), original_execution=models.stats())
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        try:
            if base is not None and original is not None:
                base.run = original
            if trial is not None:
                trial.close()
            common.close_models(models)
        finally:
            q.save(target, report)
    print(json.dumps({'status': report['status'], 'cases': len(report['rows']),
                      'passed': sum(r['passed'] for r in report['rows'])}), flush=True)


def benchmark(a):
    checked(a.stage)
    quality = json.loads((a.stage / 'evaluation.json').read_text())
    if (quality['status'] != 'all_cases_consistent_not_gt' or not quality_ok(quality['rows'])
            or quality['build_sha256'] != q.digest(a.stage / 'build.json')):
        raise ValueError('Full business quality gate has not passed')
    target = a.stage / ('benchmark_%s_%s.json' % (a.variant, a.round))
    common.fresh(target)
    plan = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
    report = dict(metadata(a.stage), status='running', variant=a.variant, round=a.round, rows=[],
                  warmup_cycles=a.warmup, measured_cycles=a.repeat,
                  note='9 original tiles from first hash-ordered image; synchronized native Session.run only, fresh process per variant. No preprocessing/merge/JPEG/whole OCR/HTTP, no clock changes.')
    models = native = None
    feeds = []
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = common.setup(a)
            base = models.models['enhance']
            original = base.run
            def capture(names, values):
                feeds.append({k: v.copy() for k, v in values.items()})
                return original(names, values)
            base.run = capture
            image = plan['calibration'][0]
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                out = pipeline.process(str(Path(plan['images']) / image['file']), image_type=1, run_ner=True)
            base.run = original
            if not out['success'] or not models.healthy or len(feeds) != 9:
                raise RuntimeError('Unexpected original tile benchmark sequence')
            common.close_models(models)
            models = pipeline = base = original = None
            gc.collect()
        report['before'] = common.telemetry()
        from trt82_backend import Session
        base_cfg = cfg['engines']['enhance']
        native = candidate(a.stage) if a.variant == 'fp16' else Session(q.RELEASE / base_cfg['path'], base_cfg['sha256'], 'enhance_reference')
        report.update(measured_engine_sha256=report['engine_sha256'] if a.variant == 'fp16' else base_cfg['sha256'],
                      input_sha256=[hashlib.sha256(f['images'].tobytes()).hexdigest() for f in feeds])
        for _ in range(a.warmup):
            for feed in feeds:
                native.run(None, feed)
        report['after_warmup'] = common.telemetry()
        for cycle in range(a.repeat):
            for index, feed in enumerate(feeds):
                start = time.perf_counter()
                native.run(None, feed)
                report['rows'].append({'cycle': cycle, 'index': index, 'wall_ms': (time.perf_counter()-start)*1000})
        report.update(status='measured', timing=common.summary([r['wall_ms'] for r in report['rows']]),
                      execution=native.stats(), after=common.telemetry(), context_device_bytes=native.engine.device_memory_size,
                      io_buffer_bytes=sum(x[1] for x in native.buffers.values()))
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        try:
            if native is not None:
                native.close()
            common.close_models(models)
        finally:
            q.save(target, report)
    print(json.dumps({'variant': a.variant, 'round': a.round, 'timing': report['timing']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('check', 'evaluate', 'benchmark'))
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--variant', choices=('fp32', 'fp16'), default='fp16')
    p.add_argument('--round', choices=('r1', 'r2'), default='r1')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--repeat', type=int, default=20)
    a = p.parse_args()
    a.stage = a.stage.resolve()
    if (platform.machine() != 'aarch64' or not a.stage.is_relative_to(q.BASE / 'staging')
            or not a.stage.is_dir() or a.warmup < 1 or a.repeat < 1):
        p.error('Use existing board staging and positive counts')
    sys.path.insert(0, str(q.RELEASE / 'edge_deploy/jetson'))
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE), AUTOOCR_RUNTIME_DIR=str(a.stage / 'runtime'), AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL)
    globals()[a.mode](a)


if __name__ == '__main__':
    main()
