"""E125: outcome-independent sample of existing testpic, original vs wide rec.

No images, text, tokens or tensors are persisted. Original file names stay in
RAM; the report uses SHA256. One packaged factory per spawned process. This is
quality/coverage validation, not a performance comparison or HTTP deployment.
"""
from collections import Counter
import contextlib
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np
import archive_rec_wide_adapter as previous

v, q, n, old, p = previous.v, previous.q, previous.n, previous.old, previous.p
STAGE = q.BASE / 'staging/rec_corpus_e125_r1'
DATA = q.BASE / 'testpic'
PRIOR_SHA = '22baed76fcda25a67d0105bb913300f5c107274578a9cf6e2db1a719a43aafaf'
PRIOR_AUDIT_SHA = '6eb8a564341340f2c0d7440b61bb1e4dbd14f57cbe68be854033255b36891269'
RULES = dict(sample_images=32, widest_aspect=4, tallest_aspect=4, hash_quantiles=24,
             paths=[3, 1], order=['reference', 'candidate'], originals_only=True,
             exclude_prior_13=True, cpu_threads=2, run_ner=True,
             worker_timeout_s=2400, no_auto_promotion=True)


def inventory(root):
    from PIL import Image
    rows, by_hash = [], {}
    # Case-insensitive suffix, flat known input directory only; do not scan outputs.
    for path in sorted(Path(root).iterdir()):
        if not path.is_file() or path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
            continue
        if path.is_symlink():
            raise ValueError('Input symlinks are not accepted')
        sha = q.digest(path)
        if sha in by_hash:
            raise ValueError('Duplicate file bytes; group before selecting')
        with Image.open(path) as image:
            width, height = image.size
            if getattr(image, 'n_frames', 1) != 1 or min(width, height) <= 0:
                raise ValueError('Expected single-page images')
            image.verify()
        rows.append(dict(sha256=sha, width=width, height=height, bytes=path.stat().st_size))
        by_hash[sha] = path
    return sorted(rows, key=lambda r: r['sha256']), by_hash


def select(rows, excluded):
    available = [r for r in rows if r['sha256'] not in excluded]
    if len(available) < RULES['sample_images']:
        raise ValueError('Insufficient new-to-regression images')
    wide = sorted(available, key=lambda r: (-r['width'] / r['height'], r['sha256']))[:RULES['widest_aspect']]
    used = {r['sha256'] for r in wide}
    tall = sorted([r for r in available if r['sha256'] not in used],
                  key=lambda r: (r['width'] / r['height'], r['sha256']))[:RULES['tallest_aspect']]
    used.update(r['sha256'] for r in tall)
    rest = sorted([r for r in available if r['sha256'] not in used], key=lambda r: r['sha256'])
    count = RULES['hash_quantiles']
    uniform = [rest[i * (len(rest) - 1) // (count - 1)] for i in range(count)]
    result = [dict(r, selection='wide_aspect') for r in wide]
    result += [dict(r, selection='tall_aspect') for r in tall]
    result += [dict(r, selection='hash_quantile') for r in uniform]
    if len(result) != 32 or len({r['sha256'] for r in result}) != 32:
        raise ValueError('Selection size/uniqueness differs')
    return result


def provenance():
    if q.digest(v.STAGE / 'technical_evidence.tar.gz') != PRIOR_SHA or q.digest(v.STAGE / 'final_audit.json') != PRIOR_AUDIT_SHA:
        raise ValueError('E124 evidence changed')
    plan = v.metadata()
    run = n.read(v.STAGE / 'run.json')
    reports = {name: n.read(v.STAGE / (name + '.json')) for name in v.RULES['order']}
    if run['reports_sha256'] != {name + '.json': q.digest(v.STAGE / (name + '.json')) for name in v.RULES['order']}:
        raise ValueError('Prior reports changed')
    if previous.summarize(plan, run, reports)['status'] != 'offline_adapter_validated_not_http_enabled':
        raise ValueError('E124 gate missing')
    return dict(prior_archive_sha256=PRIOR_SHA, prior_audit_sha256=PRIOR_AUDIT_SHA,
                package=plan['package'], script_sha256=q.digest(__file__))


def prepare():
    old.fresh(STAGE / 'plan.json')
    rows, _ = inventory(DATA)
    prior = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
    excluded = sorted(r['sha256'] for group in ('calibration', 'holdout') for r in prior[group])
    if not set(excluded).issubset(r['sha256'] for r in rows):
        raise ValueError('Original 13 images missing')
    chosen = select(rows, excluded)
    q.save(STAGE / 'plan.json', dict(provenance(), status='prepared', rules=RULES,
        inventory=rows, excluded_sha256=excluded, selected=chosen,
        cases=[dict(index=i, image_sha256=r['sha256'], image_type=kind)
               for i, (r, kind) in enumerate((r, kind) for r in chosen for kind in RULES['paths'])],
        note='32 existing images new to this regression, not newly acquired or a GT/independent dataset. Eight aspect extremes plus 24 hash quantiles chosen before outcomes; original13 excluded. Uppercase suffixes included. All file names remain on board/in RAM. No input transformations for coverage.'))
    print(json.dumps(dict(inventory=len(rows), excluded=len(excluded), selected=len(chosen), cases=64)), flush=True)


def metadata():
    plan = n.read(STAGE / 'plan.json')
    meta = provenance()
    if plan['status'] != 'prepared' or plan['rules'] != RULES or any(plan[k] != value for k, value in meta.items()):
        raise ValueError('Frozen plan/source changed')
    if plan['selected'] != select(plan['inventory'], set(plan['excluded_sha256'])):
        raise ValueError('Selected sample changed')
    return plan


class Probe:
    def __init__(self, session, enabled):
        self.session, self.original, self.enabled = session, session.run, enabled
        self.rows, self.errors = [], []

    def __call__(self, names, feeds):
        if self.errors:
            raise RuntimeError('Recognition observer unhealthy')
        try:
            rec = self.session
            gpu = rec.gpu
            route = ('original_gpu' if (gpu.base if self.enabled else gpu).in_profile(feeds)
                     else 'wide_gpu' if self.enabled and gpu.wide.in_profile(feeds) else 'cpu')
            shape = list(feeds['x'].shape)
            sha = p.b.h.sha_tensor(feeds['x'])
            cold = route == 'cpu' and rec.cpu is None
            output = self.original(names, feeds)
            if len(output) != 1:
                raise ValueError('Recognition output contract changed')
            tokens = old.tokens(output[0])
            if sha != p.b.h.sha_tensor(feeds['x']):
                raise ValueError('Recognition mutated original input')
            self.rows.append(dict(index=len(self.rows), input_shape=shape, input_sha256=sha,
                input_dtype=str(feeds['x'].dtype), output_shape=list(output[0].shape),
                output_sha256=p.b.h.sha_tensor(output[0]), ctc_sha256=p.digest_json(tokens),
                ctc_length=len(tokens), route=route, cold_cpu=cold))
            return output
        except Exception as err:
            self.errors.append(type(err).__name__)
            raise


def coverage(rows):
    calls = [c for r in rows for c in r['rec_calls']]
    widths = [c['input_shape'][3] for c in calls]
    return dict(pages=len(rows), rec_calls=len(calls), width_min=min(widths) if widths else None,
                width_max=max(widths) if widths else None,
                routes=dict(Counter(c['route'] for c in calls)),
                wide_widths=sorted({width for width in widths if 1537 <= width <= 2304}),
                over_2304_widths=sorted({width for width in widths if width > 2304}),
                near_upper_2200_2304_widths=sorted({width for width in widths if 2200 <= width <= 2304}),
                cpu_creations=sum(c['cold_cpu'] for c in calls))


def worker(variant, plan):
    path = STAGE / (variant + '.json')
    old.fresh(path)
    report = dict(status='running', variant=variant, rules=RULES, pid=os.getpid(), parent_pid=os.getppid(),
        spawn_method=mp.get_start_method(), startup_environment=p.b.runtime_environment(),
        before_load=p.b.snapshot(), rows=[])
    models = probe = None
    started = time.monotonic()
    try:
        rows, files = inventory(DATA)
        if rows != plan['inventory']:
            raise ValueError('Input inventory changed after selection')
        sys.path.insert(0, str(v.OVERLAY))
        import serve_rec_wide as factory
        import rec_wide_adapter as adapter
        if Path(factory.__file__).parent != v.OVERLAY or Path(adapter.__file__).parent != v.OVERLAY:
            raise ValueError('Not using immutable packaged adapter')
        os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE), AUTOOCR_RUNTIME_DIR=str(STAGE / ('runtime_' + variant)),
                          AUTOOCR_LOG_FILE=os.devnull)
        if variant == 'candidate':
            os.environ[factory.CONFIG_ENV] = str(v.OVERLAY / 'config.json')
        else:
            os.environ.pop(factory.CONFIG_ENV, None)
        import logging
        logging.disable(logging.CRITICAL)
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, _ = factory.load_business()
            rec = models.models['rec']
            enabled = isinstance(rec.gpu, adapter.WideRoute)
            if enabled != (variant == 'candidate'):
                raise ValueError('Backend selection differs')
            probe = Probe(rec, enabled)
            rec.run = probe
            report.update(after_load=p.b.snapshot(), initial_execution=models.stats(),
                factory_sha256=q.digest(factory.__file__), adapter_sha256=q.digest(adapter.__file__))
            for case in plan['cases']:
                before = models.stats()
                probe.rows = []
                with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                    t = time.monotonic()
                    value = pipeline.process(str(files[case['image_sha256']]),
                                             image_type=case['image_type'], run_ner=True)
                    wall = time.monotonic() - t
                if not value['success'] or not adapter.healthy(models) or probe.errors:
                    raise ValueError('Incomplete business result')
                counts = p.counters(models.stats(), before)
                if (sum(c['route'] != 'cpu' for c in probe.rows) != counts['rec']['calls']
                        or sum(c['route'] == 'cpu' for c in probe.rows) != counts['rec']['cpu_fallback_calls']):
                    raise ValueError('Observer missed recognition calls')
                result = dict(case, signature=p.signature(value['data'], server._layout_special_text),
                              counters=counts, rec_calls=probe.rows, diagnostic_wall_s=wall)
                del value
                result['after_page'] = p.b.snapshot()
                report['rows'].append(result)
                print(json.dumps(dict(variant=variant, page=case['index'] + 1, total=64,
                    rec_calls=len(probe.rows), max_rec_width=max((c['input_shape'][3] for c in probe.rows), default=0),
                    cpu_rec=counts['rec']['cpu_fallback_calls'])), flush=True)
            report.update(status='completed_observed_not_gt', execution=models.stats(), coverage=coverage(report['rows']),
                          process_wall_s=time.monotonic() - started, after_pages=p.b.snapshot())
    except Exception as err:
        report.update(status='failed', error=type(err).__name__)
        raise
    finally:
        if probe is not None:
            probe.session.run = probe.original
        if models is not None:
            try:
                adapter.close_models(models)
                report['all_gpu_sessions_closed'] = all(row.get('closed', False) for row in models.stats().values())
            except Exception as err:
                report.update(status='failed', cleanup_error=type(err).__name__)
        gc.collect()
        report['after_close'] = p.b.snapshot()
        q.save(path, report)
    if report['status'] != 'completed_observed_not_gt':
        raise RuntimeError('Cleanup failed')


def run():
    plan = metadata()
    for name in ('run', 'reference', 'candidate'):
        old.fresh(STAGE / (name + '.json'))
    report = dict(status='running', rules=RULES, parent_pid=os.getpid(), workers=[],
                  startup_environment=p.b.runtime_environment(), plan_sha256=q.digest(STAGE / 'plan.json'))
    proc = None
    try:
        context = mp.get_context('spawn')
        for variant in RULES['order']:
            proc = context.Process(target=worker, args=(variant, plan))
            proc.start()
            pid, started = proc.pid, time.monotonic()
            while proc.is_alive():
                proc.join(timeout=1)
                if time.monotonic() - started > RULES['worker_timeout_s']:
                    raise RuntimeError('Worker timeout')
            report['workers'].append(dict(variant=variant, pid=pid, exitcode=proc.exitcode))
            if proc.exitcode:
                raise RuntimeError('Worker failed')
            proc.close()
            proc = None
        report.update(status='completed', reports_sha256={name + '.json': q.digest(STAGE / (name + '.json')) for name in RULES['order']})
    except Exception as err:
        report.update(status='failed', error=type(err).__name__)
        raise
    finally:
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=10)
        q.save(STAGE / 'run.json', report)


if __name__ == '__main__':
    args = p.b.h.arguments(('prepare', 'run'))
    if args.stage != STAGE:
        raise ValueError('Unexpected stage')
    {'prepare': prepare, 'run': run}[args.mode]()
