"""Classify mixed-rec regression edits without exporting recognized text.

Replays failed E100 cases with exact original GPU feeds. No acceptance relaxation,
width special-case, model change, activation, benchmark or deployment switch.
"""
import argparse
import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sys

import numpy as np
import rec_mixed_precision as mixed
import optimize_rec_fp16 as old
import quantize_yolo_ptq as q
from diagnose_rec_difference import private_edit_summary


def failure_cases(report):
    if report['status'] != 'differences_not_accepted' or len(report['rows']) != 26:
        raise ValueError('Expected completed rejected E100 evaluation')
    cases = [r for r in report['rows'] if not r['passed']]
    if not cases:
        raise ValueError('No failed cases')
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, required=True)
    a = p.parse_args()
    a.stage = a.stage.resolve()
    if platform.machine() != 'aarch64' or not a.stage.is_relative_to(q.BASE / 'staging'):
        p.error('Board project staging only')
    target = a.stage / 'difference_diagnostic.json'
    old.fresh(target)
    evaluation = json.loads((a.stage / 'evaluation.json').read_text())
    if evaluation['build_sha256'] != q.digest(a.stage / 'build.json'):
        raise ValueError('Build changed after evaluation')
    cases = failure_cases(evaluation)
    plan = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
    images = {r['sha256']: r for r in plan['calibration'] + plan['holdout']}
    sys.path.insert(0, str(q.RELEASE / 'edge_deploy/jetson'))
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE),
                      AUTOOCR_RUNTIME_DIR=str(a.stage / 'runtime'), AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL)
    report = dict(mixed.metadata(a.stage), status='running', cases=[],
                  diagnostic_script_sha256=q.digest(__file__),
                  edit_helper_sha256=q.digest(Path(__file__).with_name('diagnose_rec_difference.py')),
                  evaluation_sha256=q.digest(a.stage / 'evaluation.json'), note=__doc__)
    models = native = None
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = old.setup(a)
            rec = models.models['rec'].gpu
            original = rec.run
            native = mixed.candidate(a.stage)
            dictionary = pipeline._ocr_engine._char_dict
            active = None

            def shadow(names, feeds):
                ref = original(names, feeds)[0]
                out = native.run(names, feeds)[0]
                metrics = old.compare(ref, out)
                index = active['gpu_call_count']
                active['gpu_call_count'] += 1
                if not metrics['token_sequence_equal']:
                    ref2 = original(names, feeds)[0]
                    out2 = native.run(names, feeds)[0]
                    ref_text = ''.join(dictionary[t] for t in old.tokens(ref))
                    out_text = ''.join(dictionary[t] for t in old.tokens(out))
                    active['changes'].append(dict(metrics, gpu_call_index=index,
                        input_shape=list(feeds['x'].shape),
                        input_sha256=hashlib.sha256(feeds['x'].tobytes()).hexdigest(),
                        repeat_reference_exact=bool(np.array_equal(ref, ref2)),
                        repeat_candidate_exact=bool(np.array_equal(out, out2)),
                        edits=private_edit_summary(ref_text, out_text)))
                return [ref]

            rec.run = shadow
            for case in cases:
                active = {k: case[k] for k in ('group', 'image_sha256', 'image_type')}
                active.update(gpu_call_count=0, changes=[])
                report['cases'].append(active)
                image = images[case['image_sha256']]
                with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                    result = pipeline.process(str(Path(plan['images']) / image['file']),
                                              image_type=case['image_type'], run_ner=True)
                if not result['success'] or not models.healthy or native.errors:
                    raise RuntimeError('Original pipeline or candidate backend error')
                print(json.dumps({'case': len(report['cases']), 'gpu_calls': active['gpu_call_count'],
                                  'changed_crops': len(active['changes'])}), flush=True)
            rec.run = original
            report.update(status='diagnosed_not_accepted', candidate_execution=native.stats(),
                          original_pipeline_execution=models.stats())
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        try:
            if native is not None:
                native.close()
            old.close_models(models)
        finally:
            q.save(target, report)
    print(json.dumps({'status': report['status'], 'cases': len(report['cases']),
                      'changed_crops': sum(len(r['changes']) for r in report['cases'])}), flush=True)


if __name__ == '__main__':
    main()
