"""E110 same-input format/numeric checks, then strict offline business gates.

No document text/tensors persisted. No HTTP, release switch or license changes.
"""
import contextlib
import gc
import json
import os
from pathlib import Path
import re
import time
import unicodedata

import numpy as np
import layout_hotspot_mixed as build

q = build.q
old = build.old


def metadata(stage):
    now = build.provenance(); b = json.loads((stage/'build.json').read_text())
    if (any(b[k] != v for k, v in now.items()) or b['status'] != 'built_not_validated'
            or b['targets'] != list(build.TARGETS) or b['profile'] != build.SHAPES
            or b['plugin_count'] != 18 or b['trt_version'] != build.TRT_VERSION
            or b['ranges_sha256'] != q.digest(stage/'ranges.json')
            or b['engine_sha256'] != q.digest(stage/'layout_hotspot.engine')):
        raise ValueError('Candidate/provenance differs')
    return dict(script_sha256=q.digest(__file__), build_script_sha256=now['script_sha256'],
                build_sha256=q.digest(stage/'build.json'), source_sha256=now['source_sha256'],
                plugin_sha256=now['plugin_sha256'], deployment_sha256=now['deployment_sha256'],
                plan_sha256=now['plan_sha256'], business_helper_sha256=q.digest(q.__file__),
                engine_sha256=b['engine_sha256'])


def candidate(stage):
    from trt82_backend import Session
    class Mixed(Session):
        def stats(self):
            return dict(super().stats(), precision='conv91_92_93_half_with_fp32_boundaries_other_compute_fp32')
    return Mixed(stage/'layout_hotspot.engine', metadata(stage)['engine_sha256'],
                   'layout_three_conv_mixed', [(build.PLUGIN, build.PLUGIN_SHA)])


def named_outputs(session, feeds):
    values = session.run(None, feeds)
    return dict(zip(build.output_names(session), values))


def reordered_outputs(session, feeds, wanted):
    # Native TRT8.2 wrapper accepts only all-output requests, not named subsets.
    values = named_outputs(session, feeds)
    if len(wanted) != len(values) or set(wanted) != set(values):
        raise ValueError('Original positional output contract differs')
    return [values[name] for name in wanted]


def compare(ref, out):
    if set(ref) != set(out): raise ValueError('Output names differ')
    result = {}
    for name in sorted(ref):
        x, y = ref[name], out[name]
        if x.shape != y.shape or x.dtype != y.dtype or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Output contract/finite check failed')
        if not x.size: raise ValueError('Unexpected empty model output')
        delta = np.abs(x.astype(np.float64)-y.astype(np.float64))
        result[name] = dict(shape=list(x.shape), dtype=str(x.dtype), elements=x.size,
                            all_finite=True, exactly_equal=bool(np.array_equal(x, y)),
                            mae=float(delta.mean()), max_abs_error=float(delta.max()),
                            allclose_diagnostic=bool(np.allclose(x,y,rtol=1e-3,atol=1e-4)))
    return result


def token(name, wanted):
    return re.search(r'(?<![\w.])'+re.escape(wanted)+r'(?![\w.])', name) is not None


def dtype_is(tensors, kind):
    pattern = r'\b(?:half|fp16)\b' if kind == 'half' else r'\b(?:float|fp32)\b'
    return bool(tensors) and all(re.search(pattern, v.get('Format/Datatype','').lower()) for v in tensors)


def inspect(info):
    rows = info.get('Layers', []); selected = []; half_rows = []; plugins = []
    if not rows or not all(isinstance(r, dict) for r in rows): raise ValueError('Detailed Inspector required')
    for name in build.TARGETS:
        found = [r for r in rows if token(r.get('Name',''), name) and r.get('ParameterType') == 'Convolution']
        if len(found) != 1: raise ValueError('Unique executed hotspot missing')
        row = found[0]
        if not dtype_is(row.get('Inputs', []), 'half') or not dtype_is(row.get('Outputs', []), 'half'):
            raise ValueError('Executed half convolution not evidenced')
        selected.append(row)
    for row in rows:
        tensors = row.get('Inputs', []) + row.get('Outputs', [])
        if any(re.search(r'\b(?:half|fp16)\b', v.get('Format/Datatype','').lower()) for v in tensors):
            half_rows.append(row['Name'])
            if row not in selected and row.get('LayerType') not in ('Reformat', 'NoOp'):
                raise ValueError('Unexpected additional half computation')
        if any(re.search(r'\bint8\b', v.get('Format/Datatype','').lower()) for v in tensors):
            raise ValueError('Unexpected INT8 format')
        if 'GridSample' in row.get('Name', ''):
            if not dtype_is(row.get('Inputs', []), 'float') or not dtype_is(row.get('Outputs', []), 'float'):
                raise ValueError('Sampling coordinate/feature format changed')
            plugins.append(row)
    if len(plugins) != 18: raise ValueError('18 float GridSample records required')
    return dict(layer_count=len(rows), hotspots=selected, half_format_layers=half_rows,
                fp32_grid_sample_count=len(plugins),
                note='Executed I/O formats, not proof of internal accumulator precision. FP32 downstream boundary enforced by constraints and absence of other half compute.')


def require(stage, filename, status):
    report = json.loads((stage/filename).read_text()); expected = metadata(stage)
    if report['status'] != status or any(report[k] != v for k,v in expected.items()):
        raise ValueError('Prior gate/provenance differs')
    return report


def check(a):
    target = a.stage/'check.json'; old.fresh(target); old.fresh(a.stage/'layers.json')
    report = dict(metadata(a.stage), status='running', rows=[]); trial = None
    try:
        rows, cfg, execution = build.capture(a); gc.collect()
        previous = json.loads((a.stage/'ranges.json').read_text())['rows']
        if [build.describe(r) for r in rows] != [{k:r[k] for k in build.describe(rows[0])} for r in previous]:
            raise ValueError('Input or original output drift')
        trial = candidate(a.stage)
        for row in rows:
            out = named_outputs(trial, row['feeds']); again = named_outputs(trial, row['feeds'])
            raw = compare(row['reference'], out)
            if set(out) != set(again) or not all(np.array_equal(out[k],again[k]) for k in out):
                raise ValueError('Candidate repeat drift')
            report['rows'].append(dict(build.describe(row), raw=raw, repeat_exact=True,
                candidate_sha256={k:build.sha_tensor(v) for k,v in sorted(out.items())}))
        inspector = trial.engine.create_engine_inspector(); inspector.execution_context = trial.context
        info = json.loads(inspector.get_engine_information(trial.trt.LayerInformationFormat.JSON))
        q.save(a.stage/'layers.json', info); report['inspection'] = inspect(info)
        report.update(status='finite_same_inputs_compared_not_business_accepted', execution=trial.stats(),
            original_execution=execution, note='6 identical original layout inputs; raw numeric differences diagnostic, not accuracy/business/benchmark. No coordinate/text normalization.')
    except Exception as e:
        report.update(status='failed', error=type(e).__name__); raise
    finally:
        if trial is not None: trial.close()
        q.save(target, report)
    print(json.dumps(dict(status=report['status'], cases=len(report['rows']))), flush=True)


def protected_equal(ref, out):
    def seq(text): return ''.join(c for c in text if unicodedata.category(c)[0] in 'NPS')
    return seq(ref) == seq(out)


def business_passed(row):
    return (all(row['strict_equal'][k] for k in ('FullText','Text','MarkdownBodyLines','NER','raw_line_texts'))
        and all(row['fields']['api_special_text_equal'][k] for k in ('title','category_no','page_no'))
        and row['number_symbol_sequence_equal'] and row['same_original_input']
        and row['cpu_fallback_calls']['reference'] == row['cpu_fallback_calls']['candidate'])


def evaluate(a):
    require(a.stage, 'check.json', 'finite_same_inputs_compared_not_business_accepted')
    target = a.stage/'evaluation.json'; old.fresh(target)
    plan = q.load_plan(q.BASE/'staging/quant_yolo_e92_r1')
    report = dict(metadata(a.stage), status='running', rows=[],
        note='13 existing images x ordinary/enhanced. Remaining 6 models and original rules unchanged. Strict full text, grouping, lines, Markdown, NER, special API text and same CPU fallback counts. Coordinate/internal geometry diagnostic. No GT/new normalization. Single-pass wall times not benchmark.')
    models = trial = base = original = None
    try:
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = old.setup(a)
            base = models.models['layout']; original = base.run; trial = candidate(a.stage)
            active = 'reference'; hashes = None
            def run(names, feeds):
                if trial.errors: raise RuntimeError('Candidate unhealthy; no fallback')
                hashes[active].append(build.input_metadata(feeds))
                # Preserve original API output ordering even if serialized binding order changes.
                if names not in (None, []): raise ValueError('Only all-output calls supported')
                return original(names,feeds) if active == 'reference' else reordered_outputs(trial,feeds,build.output_names(base))
            guarded = build.GuardedCall(run); base.run = guarded
            for image in plan['calibration'] + plan['holdout']:
                for kind in (3,1):
                    outputs = {}; wall = {}; counts = {}; hashes = {'reference':[], 'candidate':[]}
                    order = ('reference','candidate') if len(report['rows']) % 2 == 0 else ('candidate','reference')
                    for active in order:
                        before = models.stats(); start = time.perf_counter()
                        with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                            result = pipeline.process(str(Path(plan['images'])/image['file']),image_type=kind,run_ner=True)
                        guarded.require_healthy()
                        wall[active] = (time.perf_counter()-start)*1000; after = models.stats()
                        counts[active] = {r:after[r]['cpu_fallback']['calls']-before[r]['cpu_fallback']['calls'] for r in ('det','rec')}
                        if not result['success'] or not models.healthy or trial.errors: raise RuntimeError('Business/backend failure')
                        outputs[active] = result['data']
                    if len(hashes['reference']) != 1 or hashes['reference'] != hashes['candidate']:
                        raise ValueError('Layout input drift or candidate bypass')
                    ref, out = outputs['reference'], outputs['candidate']; row = q.strict_compare(ref,out)
                    row.update(image_sha256=image['sha256'],image_type=kind,order=order,wall_ms=wall,
                        same_original_input=True,inputs=hashes['reference'][0],cpu_fallback_calls=counts,
                        fields=q.field_diagnostics(ref,out,server._layout_special_text),
                        number_symbol_sequence_equal=protected_equal(ref['FullText'],out['FullText']),
                        reference_text_sha256=build.hashlib.sha256(ref['FullText'].encode()).hexdigest(),
                        candidate_text_sha256=build.hashlib.sha256(out['FullText'].encode()).hexdigest(),
                        strict_objects_passed=row['passed'])
                    row['passed'] = business_passed(row); report['rows'].append(row)
                    print(json.dumps(dict(case=len(report['rows']),passed=row['passed'],number_symbol_sequence_equal=row['number_symbol_sequence_equal'])),flush=True)
            report.update(status='all_26_business_consistent_not_gt' if len(report['rows']) == 26 and all(r['passed'] for r in report['rows']) else 'differences_not_accepted',
                          candidate_execution=trial.stats(),original_execution=models.stats())
    except Exception as e:
        report.update(status='failed',error=type(e).__name__); raise
    finally:
        if original is not None: base.run=original
        if trial is not None: trial.close()
        old.close_models(models); q.save(target,report)
    print(json.dumps(dict(status=report['status'],passed=sum(r['passed'] for r in report['rows']))),flush=True)


if __name__ == '__main__':
    a=build.arguments(('check','evaluate')); {'check':check,'evaluate':evaluate}[a.mode](a)
