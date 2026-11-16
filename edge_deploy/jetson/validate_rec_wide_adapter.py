"""E124 immutable overlay packaging, offline restart/rollback and bounded soak."""
import argparse
import contextlib
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import numpy as np
import archive_rec_wide as previous

w, q, n, old, p = previous.w, previous.q, previous.n, previous.old, previous.b.p
STAGE = q.BASE / 'staging/rec_wide_adapter_e124_r1'
OVERLAY = q.BASE / 'overlays/rec-wide-e124-r1'
PRIOR_SHA = '9fa12e50df66697049938ee0481baaef1a549d1ee8522e40b18bf9b3ef14963f'
PRIOR_AUDIT_SHA = 'd275705d26875472e651c31def234e9c1ca970d0c134bf4a54f9e0fa2152ba61'
RULES = dict(order=['off_before', 'on_soak', 'off_after'], warmup_cycles=1,
             soak_cycles=4, off_cycles=1, worker_timeout_s=1800,
             maximum_postwarm_nvmap_growth_mib=64, maximum_postwarm_rss_growth_mib=128,
             minimum_mem_available_mib=1024, maximum_swap_free_drop_mib=16,
             boundaries=[1536, 1537, 2304, 2305], http_started=False,
             license_or_counter_called=False, original_release_changed=False)


def provenance():
    if q.digest(w.STAGE / 'technical_evidence.tar.gz') != PRIOR_SHA or q.digest(w.STAGE / 'final_audit.json') != PRIOR_AUDIT_SHA:
        raise ValueError('E123 evidence changed')
    result = previous.validate_reports(w.STAGE)
    if result['status'] != 'wide_pipeline_gate_passed_not_promoted':
        raise ValueError('Prior performance/quality gate missing')
    return dict(prior_archive_sha256=PRIOR_SHA, prior_audit_sha256=PRIOR_AUDIT_SHA)


def copy_new(source, target):
    with Path(source).open('rb') as src, Path(target).open('xb') as dst:
        shutil.copyfileobj(src, dst)


def prepare():
    import rec_wide_adapter as adapter
    old.fresh(STAGE / 'plan.json')
    old.fresh(OVERLAY)
    meta = provenance()
    audit = previous.audit(w.STAGE)
    OVERLAY.mkdir(parents=True, exist_ok=False)
    for name in adapter.ADAPTERS:
        copy_new(STAGE / name, OVERLAY / name)
    copy_new(w.STAGE / 'rec_wide_fp32.engine', OVERLAY / 'rec_wide_fp32.engine')
    config = adapter.configuration(OVERLAY)
    q.save(OVERLAY / 'config.json', config)
    adapter.validate_config(OVERLAY / 'config.json')
    q.save(STAGE / 'plan.json', dict(meta, status='prepared', rules=RULES,
        package={name: q.digest(OVERLAY / name) for name in (*adapter.ADAPTERS, 'config.json', 'rec_wide_fp32.engine')},
        source_sha256={name: q.digest(STAGE / name) for name in (*adapter.ADAPTERS, 'validate_rec_wide_adapter.py')},
        cases=n.read(w.STAGE / 'plan.json')['cases'], initial_audit=audit,
        note='Opt-in overlay; original release and systemd untouched. Missing env delegates original. Empty/bad env fails. Restart-only rollback, no live switching; offline workers, no license/counter/HTTP. Bounded repeated development data, not GT/long-duration production proof.'))
    print('Prepared immutable opt-in overlay; service remains off', flush=True)


def metadata():
    plan = n.read(STAGE / 'plan.json')
    if plan['status'] != 'prepared' or plan['rules'] != RULES:
        raise ValueError('Plan changed')
    for name, expected in plan['package'].items():
        if q.digest(OVERLAY / name) != expected:
            raise ValueError('Package changed')
    for name, expected in plan['source_sha256'].items():
        if q.digest(STAGE / name) != expected:
            raise ValueError('Harness changed')
    return plan


def synthetic_boundaries(models, adapter):
    """Original/extended/CPU routes; unchanged synthetic tensors, not real OCR GT."""
    from jetson_models import cpu_session
    rec = models.models['rec']
    reference = cpu_session(rec.onnx)
    rows = []
    for width in RULES['boundaries']:
        # Do not pad/resize real OCR inputs to manufacture coverage evidence.
        feed = {'x': np.zeros((1, 3, 48, width), np.float32)}
        original_hash = p.b.h.sha_tensor(feed['x'])
        before = rec.stats()
        output = rec.run(None, feed)[0]
        again = rec.run(None, feed)[0]
        expected = reference.run(None, feed)[0]
        metrics = old.compare(expected, output)
        after = rec.stats()
        route = 'original_gpu' if width <= 1536 else 'wide_gpu' if width <= 2304 else 'cpu'
        delta = dict(original=after['original_execution']['calls'] - before['original_execution']['calls'],
                     wide=after['wide_execution']['calls'] - before['wide_execution']['calls'],
                     cpu=after['cpu_fallback']['calls'] - before['cpu_fallback']['calls'])
        wanted = dict(original=2 if route == 'original_gpu' else 0,
                      wide=2 if route == 'wide_gpu' else 0, cpu=2 if route == 'cpu' else 0)
        good = (adapter.healthy(models) and delta == wanted and metrics['token_sequence_equal']
                and metrics['allclose_diagnostic'] and original_hash == p.b.h.sha_tensor(feed['x'])
                and p.b.h.sha_tensor(output) == p.b.h.sha_tensor(again))
        rows.append(dict(width=width, route=route, call_delta=delta, input_sha256=original_hash,
                         candidate_sha256=p.b.h.sha_tensor(output), repeated_exact=np.array_equal(output, again),
                         passed=bool(good), **metrics))
        if not good:
            raise ValueError('Boundary routing/output differs')
    return rows


def resource_summary(warm, checkpoints):
    samples = [warm] + checkpoints
    nv = [(s['memory']['NvMapMemUsed_kib'] - warm['memory']['NvMapMemUsed_kib']) / 1024 for s in samples]
    rss = [(s['process_rss_kib'] - warm['process_rss_kib']) / 1024 for s in samples]
    free = min(s['memory']['MemAvailable_kib'] / 1024 for s in samples)
    swap = max((warm['memory']['SwapFree_kib'] - s['memory']['SwapFree_kib']) / 1024 for s in samples)
    return dict(max_postwarm_nvmap_growth_mib=max(nv), max_postwarm_rss_growth_mib=max(rss),
                minimum_mem_available_mib=free, maximum_swap_free_drop_mib=swap,
                passed=(max(nv) <= RULES['maximum_postwarm_nvmap_growth_mib']
                        and max(rss) <= RULES['maximum_postwarm_rss_growth_mib']
                        and free >= RULES['minimum_mem_available_mib']
                        and swap <= RULES['maximum_swap_free_drop_mib']),
                note='Post-page/cycle snapshots, not instantaneous peak or multi-hour leak proof. RSS and NvMap overlap; do not add them.')


def worker(variant, plan):
    path = STAGE / (variant + '.json')
    old.fresh(path)
    start = time.monotonic()
    report = dict(status='running', variant=variant, rules=RULES, pid=os.getpid(), parent_pid=os.getppid(),
                  spawn_method=mp.get_start_method(), startup_environment=p.b.runtime_environment(),
                  before_load=p.b.snapshot(), rows=[], checkpoints=[])
    models = None
    try:
        sys.path.insert(0, str(OVERLAY))
        import serve_rec_wide as factory
        import rec_wide_adapter as adapter
        if Path(factory.__file__).parent != OVERLAY or Path(adapter.__file__).parent != OVERLAY:
            raise ValueError('Did not load packaged adapter')
        os.environ['AUTOOCR_SOURCE_ROOT'] = str(q.RELEASE)
        os.environ['AUTOOCR_RUNTIME_DIR'] = str(STAGE / ('runtime_' + variant))
        os.environ['AUTOOCR_LOG_FILE'] = os.devnull
        if variant == 'on_soak':
            os.environ[factory.CONFIG_ENV] = str(OVERLAY / 'config.json')
        else:
            os.environ.pop(factory.CONFIG_ENV, None)
        import logging
        logging.disable(logging.CRITICAL)
        data = q.load_plan(q.BASE / 'staging/quant_yolo_e92_r1')
        images = data['calibration'] + data['holdout']
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, _ = factory.load_business()
            rec = models.models['rec']
            report.update(after_load=p.b.snapshot(), after_load_execution=models.stats(),
                          overlay_loaded=isinstance(rec.gpu, adapter.WideRoute),
                          loaded_factory_sha256=q.digest(factory.__file__),
                          loaded_adapter_sha256=q.digest(adapter.__file__))
            if report['overlay_loaded'] != (variant == 'on_soak'):
                raise ValueError('Enable/rollback selection differs')
            cycles = RULES['warmup_cycles'] + RULES['soak_cycles'] if variant == 'on_soak' else RULES['off_cycles']
            for cycle in range(cycles):
                indices = list(range(26)) if cycle % 2 == 0 else list(reversed(range(26)))
                for i in indices:
                    case = plan['cases'][i]
                    image = images[i // 2]
                    if image['sha256'] != case['image_sha256']:
                        raise ValueError('Image changed')
                    before = models.stats()
                    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                        t = time.perf_counter()
                        value = pipeline.process(str(Path(data['images']) / image['file']),
                                                 image_type=case['image_type'], run_ner=True)
                        wall = (time.perf_counter() - t) * 1000
                    signature = p.signature(value['data'], server._layout_special_text)
                    counts = p.counters(models.stats(), before)
                    wide = (rec.gpu.wide.calls - before['rec']['wide_execution']['calls']) if variant == 'on_soak' else 0
                    if (not value['success'] or not adapter.healthy(models) or signature != case['signature']
                            or counts != w.expected_counters(case, 'candidate' if variant == 'on_soak' else 'reference')
                            or wide != (case['counters']['rec']['cpu_fallback_calls'] if variant == 'on_soak' else 0)):
                        raise ValueError('Business/routing differs')
                    del value
                    report['rows'].append(dict(index=i, cycle=cycle, signature=signature, counters=counts,
                                               wide_calls=wide, wall_ms=wall))
                    if variant == 'on_soak' and cycle:
                        report['checkpoints'].append(p.b.snapshot())
                gc.collect()
                if cycle == 0:
                    report['after_warmup'] = p.b.snapshot()
                print(variant, 'cycle', cycle + 1, '/', cycles, 'pages', len(report['rows']), flush=True)
            report.update(after_pages=p.b.snapshot(), page_execution=models.stats())
            if variant == 'on_soak':
                report['resources'] = resource_summary(report['after_warmup'], report['checkpoints'])
                report['boundaries'] = synthetic_boundaries(models, adapter)
                report['after_boundaries'] = p.b.snapshot()
            if not adapter.healthy(models):
                raise ValueError('Backend not healthy')
            report.update(status='completed_offline_not_http', execution=models.stats(),
                          process_wall_s=time.monotonic() - start)
    except Exception as err:
        report.update(status='failed', error=type(err).__name__)
        raise
    finally:
        if models is not None:
            try:
                adapter.close_models(models)
                report['all_gpu_sessions_closed'] = all(r.get('closed', False) for r in models.stats().values())
            except Exception as err:
                report.update(status='failed', cleanup_error=type(err).__name__)
        gc.collect()
        report['after_close'] = p.b.snapshot()
        q.save(path, report)
    if report['status'] != 'completed_offline_not_http':
        raise RuntimeError('Worker cleanup failed')


def run():
    plan = metadata()
    provenance()
    for name in (*RULES['order'], 'run'):
        old.fresh(STAGE / (name + '.json'))
    report = dict(status='running', rules=RULES, parent_pid=os.getpid(),
                  startup_environment=p.b.runtime_environment(), workers=[])
    proc = None
    try:
        context = mp.get_context('spawn')
        for variant in RULES['order']:
            proc = context.Process(target=worker, args=(variant, plan))
            proc.start()
            pid, start = proc.pid, time.monotonic()
            while proc.is_alive():
                proc.join(timeout=1)
                if time.monotonic() - start > RULES['worker_timeout_s']:
                    raise RuntimeError('Worker timeout')
            report['workers'].append(dict(variant=variant, pid=pid, exitcode=proc.exitcode))
            if proc.exitcode:
                raise RuntimeError('Worker failed')
            proc.close()
            proc = None
        report.update(status='completed_offline', reports_sha256={name + '.json': q.digest(STAGE / (name + '.json')) for name in RULES['order']})
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
