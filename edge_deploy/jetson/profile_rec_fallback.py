"""E122: original-width CPU fallback attribution, no model/service changes.

All 26 existing development cases: one warm/cold-start pass, then forward and
reverse diagnostic passes. Hashes are calculated after the page timer. No input
tensors, output tensors, OCR text, or CTC class sequences are saved.
"""
import contextlib
import gc
import json
import os
from pathlib import Path
import time

import archive_ner_pipeline as previous

p = previous.p
n, q, old = p.n, p.q, p.old
STAGE = q.BASE / 'staging/rec_fallback_e122_r1'
PRIOR = p.STAGE
PRIOR_SHA = 'deac0406ee31282d8793d36d0a5b6952ade2795a6b852bddc065c45e4a157258'
AUDIT_SHA = '92efd5a7cb6b7db4809ed17d241013922024038bdc8a19b4e7dcdc7a74a46a07'
RULES = dict(cases=26, phases=['warmup', 'r1', 'r2'], reverse_r2=True,
             original_seven_models=True, run_ner=True, cpu_threads=2,
             mutate_inputs=False, accepted_for_release=False)


def provenance():
    if q.digest(PRIOR / 'technical_evidence.tar.gz') != PRIOR_SHA or q.digest(PRIOR / 'final_audit.json') != AUDIT_SHA:
        raise ValueError('E121 evidence drift')
    manifest = n.read(PRIOR / 'manifest.json')['files']
    # The manifest format is a list of technical file records.
    wanted = set(previous.SCRIPTS) | {'plan.json', 'pipeline_reference_r1.json'}
    rows = {r['name']: r for r in manifest}
    for name in wanted:
        if q.digest(PRIOR / name) != rows[name]['sha256']:
            raise ValueError('Prior source/report drift')
    if q.digest(old.SOURCE) != old.SOURCE_SHA:
        raise ValueError('Original recognition source drift')
    audit = n.read(PRIOR / 'final_audit.json')
    cfg_path = q.RELEASE / 'edge_deploy/jetson/deployment.json'
    if q.digest(cfg_path) != audit['deployment_sha256']:
        raise ValueError('Deployment drift')
    cfg = n.read(cfg_path)
    if {k: q.digest(q.RELEASE / v['path']) for k, v in cfg['engines'].items()} != audit['engines']:
        raise ValueError('Original engine drift')
    return dict(script_sha256=q.digest(__file__), prior_archive_sha256=PRIOR_SHA,
                prior_audit_sha256=AUDIT_SHA, source_sha256=old.SOURCE_SHA,
                prior_reference_sha256=q.digest(PRIOR / 'pipeline_reference_r1.json'),
                deployment_sha256=audit['deployment_sha256'], engines=audit['engines'],
                prior_sources={name: rows[name]['sha256'] for name in previous.SCRIPTS})


def prepare(a):
    old.fresh(a.stage / 'plan.json')
    prior = n.read(PRIOR / 'pipeline_reference_r1.json')
    cases = [dict(index=r['index'], image_sha256=r['image_sha256'], image_type=r['image_type'],
                  signature=r['signature'], counters=r['counters']) for r in prior['warmup']]
    if [r['index'] for r in cases] != list(range(26)):
        raise ValueError('Incomplete prior coverage')
    q.save(a.stage / 'plan.json', dict(provenance(), status='prepared', rules=RULES, cases=cases))


def outside_dimensions(shapes, profiles):
    if set(shapes) != set(profiles):
        raise ValueError('Input names differ')
    result = []
    for name, shape in shapes.items():
        lo, _, hi = profiles[name]
        if len(shape) != len(lo):
            raise ValueError('Input rank differs')
        result.extend(dict(input=name, axis=i, value=d, minimum=a, maximum=b)
                      for i, (d, a, b) in enumerate(zip(shape, lo, hi)) if not a <= d <= b)
    return result


class Probe:
    """Observe the existing routing layer; never implement an alternate route."""
    def __init__(self, session, factory):
        self.session, self.original, self.factory = session, session.run, factory
        self.rows, self.pending, self.creations = [], [], []

    def create_cpu(self, path):
        start = time.perf_counter()
        value = self.factory(path)
        elapsed = (time.perf_counter() - start) * 1000
        if Path(path) == self.session.onnx:
            options = value.get_session_options()
            self.creations.append(dict(wall_ms=elapsed, providers=value.get_providers(),
                                       threads=options.intra_op_num_threads,
                                       execution_mode=str(options.execution_mode)))
        return value

    def call(self, names, feeds):
        shapes = {k: list(v.shape) for k, v in feeds.items()}
        outside = outside_dimensions(shapes, self.session.gpu.profiles)
        route = 'cpu' if outside else 'gpu'
        cold = route == 'cpu' and self.session.cpu is None
        created = len(self.creations)
        start = time.perf_counter()
        out = self.original(names, feeds)
        elapsed = (time.perf_counter() - start) * 1000
        if len(out) != 1:
            raise ValueError('Recognition output contract differs')
        new_creations = self.creations[created:]
        if len(new_creations) != int(cold):
            raise ValueError('Unexpected CPU session creation')
        row = dict(call_index=len(self.rows), route=route, input_shapes=shapes,
                   input_dtypes={k: str(v.dtype) for k, v in feeds.items()},
                   output_shape=list(out[0].shape), output_dtype=str(out[0].dtype),
                   outside_dimensions=outside, cold_cpu=cold, route_wall_ms=elapsed,
                   cpu_creation_ms=sum(v['wall_ms'] for v in new_creations))
        self.rows.append(row)
        if route == 'cpu':
            self.pending.append((row, feeds, out[0]))
        return out

    def finish_page(self):
        for row, feeds, output in self.pending:
            tokens = old.tokens(output)
            row.update(input_sha256={k: p.b.h.sha_tensor(v) for k, v in feeds.items()},
                       output_sha256=p.b.h.sha_tensor(output),
                       ctc_sha256=p.digest_json(tokens), ctc_length=len(tokens))
        self.pending.clear()


def measured_indices(phase):
    return list(reversed(range(26))) if phase == 'r2' else list(range(26))


def validate_page(row, case, profiles):
    if any(row[k] != case[k] for k in ('index', 'image_sha256', 'image_type', 'signature', 'counters')):
        raise ValueError('Original business/counter drift')
    calls = row['calls']
    if [r['call_index'] for r in calls] != list(range(len(calls))):
        raise ValueError('Call coverage differs')
    for route, key in (('gpu', 'calls'), ('cpu', 'cpu_fallback_calls')):
        if sum(r['route'] == route for r in calls) != case['counters']['rec'][key]:
            raise ValueError('Routing count differs')
    for r in calls:
        outside = outside_dimensions(r['input_shapes'], profiles)
        if outside != r['outside_dimensions'] or r['route'] != ('cpu' if outside else 'gpu'):
            raise ValueError('Route does not match actual profile')
        old.summary([r['route_wall_ms'], r['cpu_creation_ms']])
        if r['cold_cpu'] and r['route'] != 'cpu':
            raise ValueError('GPU call cannot initialize CPU')
        if r['cpu_creation_ms'] > r['route_wall_ms'] or (not r['cold_cpu'] and r['cpu_creation_ms']):
            raise ValueError('Nested initialization timing invalid')
        if r['route'] == 'cpu' and any(k not in r for k in ('input_sha256', 'output_sha256', 'ctc_sha256', 'ctc_length')):
            raise ValueError('Missing fallback evidence')
    old.summary([row['wall_ms']])
    if sum(r['route_wall_ms'] for r in calls) > row['wall_ms']:
        raise ValueError('Nested recognition timing invalid')


def summarize(plan, report):
    if plan['rules'] != RULES or report['rules'] != RULES or report['status'] != 'diagnosed':
        raise ValueError('Incomplete diagnostic')
    if len(plan['cases']) != 26 or [v['index'] for v in plan['cases']] != list(range(26)):
        raise ValueError('Incomplete plan')
    profiles = report['profiles']
    if profiles != {'x': [list(v) for v in old.PROFILE]} or report['after_setup_execution']['rec']['profiles'] != profiles:
        raise ValueError('Original recognition profile differs')
    seen, round_stats = {}, []
    for phase in RULES['phases']:
        rows = [v for v in report['pages'] if v['phase'] == phase]
        if [r['index'] for r in rows] != measured_indices(phase):
            raise ValueError('Page order/coverage differs')
        for r in rows:
            validate_page(r, plan['cases'][r['index']], profiles)
            stable = [{k: v for k, v in c.items() if k not in ('route_wall_ms', 'cpu_creation_ms', 'cold_cpu')}
                      for c in r['calls']]
            if r['index'] in seen and seen[r['index']] != stable:
                raise ValueError('Original shapes/fallback outputs changed across passes')
            seen[r['index']] = stable
        if phase != 'warmup':
            cpu_ms = sum(c['route_wall_ms'] for r in rows for c in r['calls'] if c['route'] == 'cpu')
            page_ms = sum(r['wall_ms'] for r in rows)
            slow = [dict(index=r['index'], image_type=r['image_type'], wall_ms=r['wall_ms'],
                         rec_gpu_ms=sum(c['route_wall_ms'] for c in r['calls'] if c['route'] == 'gpu'),
                         rec_cpu_ms=sum(c['route_wall_ms'] for c in r['calls'] if c['route'] == 'cpu'))
                    for r in rows if any(c['route'] == 'cpu' for c in r['calls'])]
            round_stats.append(dict(round=phase, page_timing=old.summary([r['wall_ms'] for r in rows]),
                                    cpu_ms=cpu_ms, cpu_fraction_of_all_pages=cpu_ms / page_ms,
                                    fallback_pages=slow))
    if len(report['pages']) != 78 or any(c['cold_cpu'] for r in report['pages'] if r['phase'] != 'warmup' for c in r['calls']):
        raise ValueError('Extra rows or measured CPU initialization')
    cold = [c for r in report['pages'] for c in r['calls'] if c['cold_cpu']]
    creations = report['cpu_creations']
    if len(cold) != 1 or len(creations) != 1 or creations[0]['providers'] != ['CPUExecutionProvider'] or creations[0]['threads'] != 2 or creations[0]['execution_mode'] != 'ExecutionMode.ORT_SEQUENTIAL':
        raise ValueError('CPU runtime/init differs')
    if cold[0]['cpu_creation_ms'] != creations[0]['wall_ms']:
        raise ValueError('Initialization time differs')
    totals = p.counters(report['execution'], report['after_setup_execution'])
    for role, values in totals.items():
        for key, count in values.items():
            if count != sum(r['counters'][role][key] for r in report['pages']):
                raise ValueError('Total call counters differ')
    grouped = {}
    for r in report['pages']:
        if r['phase'] == 'warmup':
            continue
        for c in r['calls']:
            if c['route'] == 'cpu':
                key = tuple(c['input_shapes']['x'])
                grouped.setdefault(key, []).append(c['route_wall_ms'])
    return dict(status='fallback_attributed_not_optimized', cases=26, warm_pages=26, measured_pages=52,
                profiles=profiles, rounds=round_stats, cpu_initialization=creations[0],
                cpu_shapes=[dict(shape=list(k), timing=old.summary(v)) for k, v in sorted(grouped.items())],
                execution_delta=totals, accepted_for_release=False,
                note='Instrumented original offline run_ner=True pipeline; two diagnostic passes, not HTTP/GT or an optimization benchmark. CPU initialization nested in first warm call, excluded from measured passes. No model or input change; shape-specific GPU candidate not yet built.')


def measure(a):
    target = a.stage / 'diagnostic.json'
    old.fresh(target)
    meta = provenance()
    plan = n.read(a.stage / 'plan.json')
    if plan['status'] != 'prepared' or plan['rules'] != RULES or any(plan[k] != v for k, v in meta.items()):
        raise ValueError('Plan drift')
    report = dict(meta, rules=RULES, status='running', pid=os.getpid(), pages=[],
                  plan_sha256=q.digest(a.stage / 'plan.json'), before_load=p.b.snapshot())
    models = probe = factory = None
    try:
        import jetson_models as jm
        data = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
        images = data['calibration'] + data['holdout']
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, cfg = old.setup(a)
            rec = models.models['rec']
            if rec.cpu is not None:
                raise ValueError('CPU recognition unexpectedly already initialized')
            report.update(after_load=p.b.snapshot(), profiles=rec.gpu.profiles,
                          after_setup_execution=models.stats())
            factory = jm.cpu_session
            probe = Probe(rec, factory)
            guard = p.b.h.GuardedCall(probe.call)
            rec.run, jm.cpu_session = guard, probe.create_cpu
            for phase in RULES['phases']:
                for i in measured_indices(phase):
                    case = plan['cases'][i]
                    image = images[i // 2]
                    if image['sha256'] != case['image_sha256']:
                        raise ValueError('Image order drift')
                    probe.rows = []
                    before = models.stats()
                    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                        start = time.perf_counter()
                        value = pipeline.process(str(Path(data['images']) / image['file']), image_type=case['image_type'], run_ner=True)
                        wall = (time.perf_counter() - start) * 1000
                    guard.require_healthy()
                    if not value['success'] or not models.healthy:
                        raise ValueError('Incomplete business result')
                    probe.finish_page()
                    row = dict(index=i, phase=phase, image_sha256=image['sha256'], image_type=case['image_type'],
                               wall_ms=wall, calls=probe.rows, signature=p.signature(value['data'], server._layout_special_text),
                               counters=p.counters(models.stats(), before))
                    validate_page(row, case, report['profiles'])
                    report['pages'].append(row)
                    del value
                    if i in (0, 7, 14, 15, 25):
                        print(json.dumps(dict(phase=phase, index=i, wall_ms=wall,
                                              cpu_shapes=[c['input_shapes'] for c in row['calls'] if c['route'] == 'cpu'])), flush=True)
                report['after_' + phase] = p.b.snapshot()
            report.update(status='diagnosed', cpu_creations=probe.creations, execution=models.stats())
            report['summary'] = summarize(plan, report)
    except Exception as error:
        report.update(status='failed', error=type(error).__name__)
        raise
    finally:
        if probe is not None:
            probe.session.run = probe.original
        if factory is not None:
            jm.cpu_session = factory
        try:
            old.close_models(models)
        except Exception as error:
            report.update(status='failed', cleanup_error=type(error).__name__)
        gc.collect()
        report['after_close'] = p.b.snapshot()
        q.save(target, report)
    if report['status'] != 'diagnosed':
        raise RuntimeError('Cleanup failed')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    a = p.b.h.arguments(('prepare', 'measure'))
    if a.stage != STAGE:
        raise ValueError('Unexpected staging directory')
    {'prepare': prepare, 'measure': measure}[a.mode](a)
