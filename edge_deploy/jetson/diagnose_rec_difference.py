"""Repeat same-input rec FP16 disagreements, without storing document text.

Exploratory diagnosis of a previously observed holdout failure; never a new
independent acceptance set, no tolerance relaxation or release modification.
"""
import argparse
import contextlib
import difflib
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import unicodedata

import numpy as np
import optimize_rec_fp16 as trial
import quantize_yolo_ptq as q


def category_counts(text):
    out = {'number': 0, 'punctuation_or_symbol': 0, 'whitespace': 0, 'other': 0}
    for c in text:
        cat = unicodedata.category(c)
        key = ('whitespace' if c.isspace() else 'number' if cat[0] == 'N'
               else 'punctuation_or_symbol' if cat[0] in 'PS' else 'other')
        out[key] += 1
    return out


def private_edit_summary(a, b):
    rows = []
    for tag, i, j, k, l in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag != 'equal':
            rows.append({'operation': tag, 'reference_count': j-i, 'candidate_count': l-k,
                         'reference_categories': category_counts(a[i:j]),
                         'candidate_categories': category_counts(b[k:l])})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, required=True)
    a = p.parse_args()
    a.stage = a.stage.resolve()
    if not a.stage.is_relative_to(q.BASE / 'staging'):
        p.error('Project staging only')
    target = a.stage / 'rec_difference_diagnostic.json'
    trial.fresh(target)
    evaluation = json.loads((a.stage / 'evaluate_rec_fp16.json').read_text())
    failures = [r for r in evaluation['rows'] if not r['passed']]
    if len(failures) != 1:
        raise ValueError('Expected the single observed failure; inspect report first')
    case = failures[0]
    plan = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
    image = next(r for r in plan['holdout'] if r['sha256'] == case['image_sha256'])
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE), AUTOOCR_RUNTIME_DIR=str(a.stage / 'runtime'),
                      AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL)
    sys.path.insert(0, str(q.RELEASE / 'edge_deploy/jetson'))
    report = {'status': 'running', 'rows': [], 'input_call_count': 0,
              'script_sha256': q.digest(__file__), 'image_sha256': image['sha256'],
              'evaluation_sha256': q.digest(a.stage / 'evaluate_rec_fp16.json'),
              'image_type': case['image_type'], 'note': __doc__}
    models = candidate = None
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = trial.setup(a)
            original = models.models['rec'].gpu.run
            candidate = trial.candidate(a.stage)
            dictionary = pipeline._ocr_engine._char_dict

            def run(names, feeds):
                ref = original(names, feeds)[0]
                out = candidate.run(names, feeds)[0]
                metrics = trial.compare(ref, out)
                index = report['input_call_count']
                report['input_call_count'] += 1
                if not metrics['token_sequence_equal']:
                    ref2, out2 = original(names, feeds)[0], candidate.run(names, feeds)[0]
                    a_text = ''.join(dictionary[t] for t in trial.tokens(ref))
                    b_text = ''.join(dictionary[t] for t in trial.tokens(out))
                    report['rows'].append(dict(metrics, gpu_call_index=index,
                        input_shape=list(feeds['x'].shape),
                        input_sha256=hashlib.sha256(feeds['x'].tobytes()).hexdigest(),
                        repeat_reference_exact=bool(np.array_equal(ref, ref2)),
                        repeat_candidate_exact=bool(np.array_equal(out, out2)),
                        edits=private_edit_summary(a_text, b_text)))
                return [ref]

            models.models['rec'].gpu.run = run
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                result = pipeline.process(str(Path(plan['images']) / image['file']),
                                          image_type=case['image_type'], run_ner=True)
            if not result['success'] or not models.healthy or candidate.errors:
                raise RuntimeError('Pipeline or backend error')
            models.models['rec'].gpu.run = original
            report.update(status='diagnosed_not_accepted', candidate_execution=candidate.stats())
    except Exception as e:
        report.update(status='failed', error=type(e).__name__)
        raise
    finally:
        if candidate is not None:
            candidate.close()
        trial.close_models(models)
        q.save(target, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
