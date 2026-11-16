"""Repeat the first numeric/symbol regression, without text/image export."""
import argparse
import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sys

import optimize_enhance_fp16 as trial
import optimize_rec_fp16 as common
import quantize_yolo_ptq as q


def select_case(evaluation):
    if evaluation['status'] != 'differences_not_accepted':
        raise ValueError('Requires a completed rejected evaluation')
    return next(r for r in evaluation['rows'] if r['image_type'] == 1
                and not r['passed'] and not r['full_text_diagnostics']['number_symbol_sequence_equal'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, required=True)
    a = p.parse_args()
    a.stage = a.stage.resolve()
    if platform.machine() != 'aarch64' or not a.stage.is_relative_to(q.BASE / 'staging'):
        p.error('Board project staging only')
    target = a.stage / 'replay.json'
    common.fresh(target)
    evaluation = json.loads((a.stage / 'evaluation.json').read_text())
    metadata = trial.metadata(a.stage)
    if evaluation['build_sha256'] != metadata['build_sha256'] or evaluation['script_sha256'] != metadata['script_sha256']:
        raise ValueError('Trial changed after evaluation')
    case = select_case(evaluation)
    plan = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
    image = next(r for r in plan['calibration'] + plan['holdout'] if r['sha256'] == case['image_sha256'])
    sys.path.insert(0, str(q.RELEASE / 'edge_deploy/jetson'))
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE), AUTOOCR_RUNTIME_DIR=str(a.stage / 'runtime'), AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL)
    report = dict(metadata, status='running', diagnostic_script_sha256=q.digest(__file__),
                  evaluation_sha256=q.digest(a.stage / 'evaluation.json'), image_sha256=image['sha256'],
                  image_type=1, order=['reference_1', 'candidate_1', 'candidate_2', 'reference_2'],
                  traces={}, note='First numeric/symbol failure only; exploratory repeat, not new acceptance or benchmark. Original thresholds/inputs; no text exported.')
    models = native = base = original = None
    outputs = {}
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = common.setup(a)
            base = models.models['enhance']
            original = base.run
            native = trial.candidate(a.stage)
            active = None
            def routed(names, feeds):
                out = original(names, feeds) if active.startswith('reference') else native.run(names, feeds)
                report['traces'][active].append({'input_sha256': hashlib.sha256(feeds['images'].tobytes()).hexdigest(),
                    'output_sha256': hashlib.sha256(out[0].tobytes()).hexdigest()})
                return out
            base.run = routed
            for active in report['order']:
                report['traces'][active] = []
                with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                    result = pipeline.process(str(Path(plan['images']) / image['file']), image_type=1, run_ner=True)
                if not result['success'] or not models.healthy or native.errors:
                    raise RuntimeError('Pipeline or candidate execution failed')
                outputs[active] = result['data']
            expected = case['tile_input_sha256']
            if any([r['input_sha256'] for r in rows] != expected for rows in report['traces'].values()):
                raise RuntimeError('Original tile inputs changed')
            report['same_variant_repeats'] = {}
            for variant in ('reference', 'candidate'):
                r = q.strict_compare(outputs[variant + '_1'], outputs[variant + '_2'])
                r['tile_outputs_exact'] = report['traces'][variant + '_1'] == report['traces'][variant + '_2']
                report['same_variant_repeats'][variant] = r
            report['cross_variant'] = []
            for n in ('1', '2'):
                ref, out = outputs['reference_' + n], outputs['candidate_' + n]
                row = q.strict_compare(ref, out)
                row['fields'] = q.field_diagnostics(ref, out, server._layout_special_text)
                row['business_passed'] = trial.business_passed(row['strict_equal'], row['fields'])
                row['full_text_diagnostics'] = trial.text_diagnostics(ref['FullText'], out['FullText'])
                report['cross_variant'].append(row)
            report.update(status='replayed_not_accepted', candidate_execution=native.stats(), original_execution=models.stats())
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        try:
            if base is not None and original is not None:
                base.run = original
            if native is not None:
                native.close()
            common.close_models(models)
        finally:
            q.save(target, report)
    print(json.dumps({'status': report['status'],
        'repeat_exact': {k: v['tile_outputs_exact'] and v['passed'] and v['boxes_exactly_equal']
                         for k, v in report['same_variant_repeats'].items()},
        'cross_variant_business_equal': [r['business_passed'] for r in report['cross_variant']],
        'cross_variant_number_symbol_equal': [r['full_text_diagnostics']['number_symbol_sequence_equal'] for r in report['cross_variant']]}), flush=True)


if __name__ == '__main__':
    main()
