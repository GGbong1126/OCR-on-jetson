"""E126 bounded, single-worker offline endurance over the frozen 45-image union.

No HTTP/activation, raw documents, tensors or token IDs. This does not benchmark
speed or change the packaged adapter. Post-page gates and 5s samples are distinct.
"""
import contextlib
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import archive_rec_corpus as previous

c, q, n, old, p = previous.c, previous.q, previous.n, previous.old, previous.p
STAGE = q.BASE / 'staging/rec_soak_e126_r1'
PRIOR_SHA = '0ed599e97cf8f23f9b8b605de22af717720398f2aa28ce8472ac23d02d88636e'
PRIOR_AUDIT_SHA = 'a1e0d893f71213eac177822ce8165c31e368c30d2df6a29642185a7a90e69af5'
RULES = dict(images=45, cases=90, warmup_cycles=1, minimum_soak_s=900,
             maximum_soak_cycles=6, worker_timeout_s=3600, sampling_interval_s=5,
             cpu_threads=2, run_ner=True, maximum_postwarm_nvmap_growth_mib=64,
             maximum_postwarm_rss_growth_mib=128, minimum_mem_available_mib=1024,
             maximum_swap_free_drop_mib=16, maximum_idle_nvmap_growth_mib=64,
             original_release_changed=False, http_started=False, license_or_counter_called=False)


def union_cases(prior13, expanded):
    rows = []
    for r in prior13:
        rows.append(dict(image_sha256=r['image_sha256'], image_type=r['image_type'],
            signature=r['signature'], counters=c.v.w.expected_counters(r, 'candidate'),
            wide_calls=r['counters']['rec']['cpu_fallback_calls'], source='E124'))
    for r in expanded:
        rows.append(dict(image_sha256=r['image_sha256'], image_type=r['image_type'],
            signature=r['signature'], counters=r['counters'],
            wide_calls=sum(v['route'] == 'wide_gpu' for v in r['rec_calls']), source='E125'))
    keys = [(r['image_sha256'], r['image_type']) for r in rows]
    if (len(rows) != 90 or len(set(keys)) != 90 or len({r['image_sha256'] for r in rows}) != 45
            or any((sha, kind) not in keys for sha, _ in keys for kind in (3, 1))):
        raise ValueError('Expected 45 unique images with both paths')
    return [dict(r, index=i) for i, r in enumerate(rows)]


def provenance():
    if q.digest(c.STAGE / 'technical_evidence.tar.gz') != PRIOR_SHA or q.digest(c.STAGE / 'final_audit.json') != PRIOR_AUDIT_SHA:
        raise ValueError('E125 evidence changed')
    if previous.validate_reports()['status'] != 'expanded_corpus_consistent_not_gt':
        raise ValueError('E125 quality prerequisite missing')
    source = c.metadata()
    return dict(prior_archive_sha256=PRIOR_SHA, prior_audit_sha256=PRIOR_AUDIT_SHA,
                package=source['package'], script_sha256=q.digest(__file__))


def prepare():
    old.fresh(STAGE / 'plan.json')
    meta = provenance()
    source = n.read(c.STAGE / 'plan.json')
    rows, _ = c.inventory(c.DATA)
    if rows != source['inventory']:
        raise ValueError('Corpus changed')
    cases = union_cases(n.read(c.v.STAGE / 'plan.json')['cases'], n.read(c.STAGE / 'candidate.json')['rows'])
    q.save(STAGE / 'plan.json', dict(meta, status='prepared', rules=RULES, inventory=rows, cases=cases,
        initial_audit=c.v.previous.audit(c.v.w.STAGE),
        note='One fresh enabled worker, 90 warmup pages, complete alternating-order cycles until >=900s after warmup. Six-cycle/3600s caps fail if minimum duration absent. Business signatures and all seven model routes checked every page. No per-rec tensor hashing, new accuracy/speed claim or HTTP.'))
    print('Prepared 45 images / 90 cases, >=900s postwarm observation', flush=True)


def metadata():
    plan = n.read(STAGE / 'plan.json')
    if (plan['status'] != 'prepared' or plan['rules'] != RULES
            or any(plan[k] != value for k, value in provenance().items())
            or plan['cases'] != union_cases(n.read(c.v.STAGE / 'plan.json')['cases'],
                                           n.read(c.STAGE / 'candidate.json')['rows'])):
        raise ValueError('Plan or source changed')
    return plan


def indices(cycle):
    return list(range(RULES['cases'])) if cycle % 2 == 0 else list(reversed(range(RULES['cases'])))


def resources(warm, samples):
    result = c.v.resource_summary(warm, samples)
    # E124 uses the same frozen numerical limits; assert, do not silently inherit.
    for key in ('maximum_postwarm_nvmap_growth_mib', 'maximum_postwarm_rss_growth_mib',
                'minimum_mem_available_mib', 'maximum_swap_free_drop_mib'):
        if RULES[key] != c.v.RULES[key]:
            raise ValueError('Resource gate implementation differs')
    return result


def check_page(case, signature, counts, wide, healthy):
    if not healthy or signature != case['signature'] or counts != case['counters'] or wide != case['wide_calls']:
        raise ValueError('Business or route drift')


def worker(plan):
    path = STAGE / 'worker.json'
    old.fresh(path)
    report = dict(status='running', rules=RULES, pid=os.getpid(), parent_pid=os.getppid(),
                  spawn_method=mp.get_start_method(), startup_environment=p.b.runtime_environment(),
                  before_load=p.b.snapshot(), rows=[], checkpoints=[], cycles=[])
    models = None
    started = time.monotonic()
    try:
        rows, files = c.inventory(c.DATA)
        if rows != plan['inventory']:
            raise ValueError('Corpus drift')
        sys.path.insert(0, str(c.v.OVERLAY))
        import serve_rec_wide as factory
        import rec_wide_adapter as adapter
        if Path(factory.__file__).parent != c.v.OVERLAY or Path(adapter.__file__).parent != c.v.OVERLAY:
            raise ValueError('Packaged modules not loaded')
        os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE), AUTOOCR_RUNTIME_DIR=str(STAGE / 'runtime'),
            AUTOOCR_LOG_FILE=os.devnull, AUTOOCR_REC_WIDE_CONFIG=str(c.v.OVERLAY / 'config.json'))
        import logging
        logging.disable(logging.CRITICAL)
        with open(os.devnull, 'w') as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                server, pipeline, models, _ = factory.load_business()
            rec = models.models['rec']
            if not isinstance(rec.gpu, adapter.WideRoute):
                raise ValueError('Wide adapter missing')
            report.update(after_load=p.b.snapshot(), initial_execution=models.stats(),
                          factory_sha256=q.digest(factory.__file__), adapter_sha256=q.digest(adapter.__file__))
            for cycle in range(1 + RULES['maximum_soak_cycles']):
                cycle_started = time.monotonic()
                for i in indices(cycle):
                    case = plan['cases'][i]
                    before = models.stats()
                    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                        t = time.monotonic()
                        value = pipeline.process(str(files[case['image_sha256']]),
                                                 image_type=case['image_type'], run_ner=True)
                        wall = time.monotonic() - t
                    signature = p.signature(value['data'], server._layout_special_text)
                    counts = p.counters(models.stats(), before)
                    wide = rec.gpu.wide.calls - before['rec']['wide_execution']['calls']
                    check_page(case, signature, counts, wide, value['success'] and adapter.healthy(models))
                    del value
                    report['rows'].append(dict(index=i, cycle=cycle, signature=signature,
                        counters=counts, wide_calls=wide, diagnostic_wall_s=wall))
                    if cycle:
                        snap = p.b.snapshot()
                        report['checkpoints'].append(snap)
                        if not resources(report['after_warmup'], [snap])['passed']:
                            raise ValueError('Post-page resource gate exceeded')
                    if len(report['rows']) % 15 == 0:
                        print(json.dumps(dict(cycle=cycle, total_pages=len(report['rows']),
                            phase='warmup' if cycle == 0 else 'soak',
                            postwarm_s=0 if cycle == 0 else time.monotonic() - report['after_warmup']['monotonic_s'])), flush=True)
                gc.collect()
                end = p.b.snapshot()
                report['cycles'].append(dict(cycle=cycle, wall_s=time.monotonic() - cycle_started, after_cycle=end))
                if cycle == 0:
                    report['after_warmup'] = end
                else:
                    elapsed = end['monotonic_s'] - report['after_warmup']['monotonic_s']
                    if elapsed >= RULES['minimum_soak_s']:
                        break
            report.update(postwarm_s=report['cycles'][-1]['after_cycle']['monotonic_s'] - report['after_warmup']['monotonic_s'],
                          resources=resources(report['after_warmup'], report['checkpoints']),
                          execution=models.stats(), after_pages=p.b.snapshot())
            if report['postwarm_s'] < RULES['minimum_soak_s'] or not report['resources']['passed']:
                raise ValueError('Duration or resources not accepted')
            report.update(status='completed_bounded_soak_not_http', process_wall_s=time.monotonic() - started)
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
    if report['status'] != 'completed_bounded_soak_not_http':
        raise RuntimeError('Cleanup failed')


def child_snapshot(pid):
    result = p.b.snapshot()
    result.pop('process_rss_kib', None)
    for line in Path('/proc/%d/status' % pid).read_text().splitlines():
        if line.startswith('VmRSS:'):
            result['process_rss_kib'] = int(line.split()[1])
    if 'process_rss_kib' not in result:
        raise ProcessLookupError('Worker exited')
    return result


def run():
    plan = metadata()
    for name in ('run.json', 'worker.json'):
        old.fresh(STAGE / name)
    report = dict(status='running', rules=RULES, parent_pid=os.getpid(),
        startup_environment=p.b.runtime_environment(), before_worker=p.b.snapshot(), samples=[],
        plan_sha256=q.digest(STAGE / 'plan.json'))
    proc = None
    try:
        proc = mp.get_context('spawn').Process(target=worker, args=(plan,))
        proc.start()
        report['worker_pid'] = proc.pid
        started = last_sample = time.monotonic()
        while proc.is_alive():
            proc.join(timeout=1)
            now = time.monotonic()
            if now - started > RULES['worker_timeout_s']:
                raise RuntimeError('Worker timed out')
            if proc.is_alive() and now - last_sample >= RULES['sampling_interval_s']:
                try:
                    snap = child_snapshot(proc.pid)
                except (FileNotFoundError, ProcessLookupError):
                    if proc.is_alive():
                        proc.join(timeout=1)
                    if proc.is_alive():
                        raise
                    continue
                report['samples'].append(snap)
                last_sample = now
                if snap['memory']['MemAvailable_kib'] < RULES['minimum_mem_available_mib'] * 1024:
                    raise RuntimeError('Low available memory, aborting worker')
        report['worker_exitcode'] = proc.exitcode
        if proc.exitcode:
            raise RuntimeError('Worker failed')
        proc.close()
        proc = None
        report.update(status='completed', worker_sha256=q.digest(STAGE / 'worker.json'), after_exit=p.b.snapshot())
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
