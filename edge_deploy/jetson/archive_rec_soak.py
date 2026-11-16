"""E126 independent duration/quality/resource gates, technical-only archive."""
import argparse
import json
from pathlib import Path
import tarfile
import soak_rec_corpus as s

q, n, old, p = s.q, s.n, s.old, s.p
SCRIPTS = s.previous.SCRIPTS + ('soak_rec_corpus.py', 'archive_rec_soak.py', 'test_rec_soak.py')


def print_result(meta, result):
    print(json.dumps(meta))
    print(json.dumps(result))


def summarize(plan, run, worker):
    if (plan['status'] != 'prepared' or plan['rules'] != s.RULES or run['rules'] != s.RULES
            or worker['rules'] != s.RULES or run['status'] != 'completed'
            or worker['status'] != 'completed_bounded_soak_not_http'
            or run['worker_exitcode'] or worker['pid'] != run['worker_pid']
            or worker['parent_pid'] != run['parent_pid'] or worker['pid'] == run['parent_pid']
            or worker['spawn_method'] != 'spawn' or worker['startup_environment'] != run['startup_environment']
            or not worker['all_gpu_sessions_closed']
            or worker['factory_sha256'] != plan['package']['serve_rec_wide.py']
            or worker['adapter_sha256'] != plan['package']['rec_wide_adapter.py']):
        raise ValueError('Protocol/package/worker failure')
    cases = plan['cases']
    if (len(cases) != 90 or [r['index'] for r in cases] != list(range(90))
            or len({(r['image_sha256'], r['image_type']) for r in cases}) != 90
            or len({r['image_sha256'] for r in cases}) != 45
            or any((sha, kind) not in {(r['image_sha256'], r['image_type']) for r in cases}
                   for sha in {r['image_sha256'] for r in cases} for kind in (3, 1))):
        raise ValueError('Case coverage differs')
    cycles = worker['cycles']
    count = len(cycles)
    if not 2 <= count <= 1 + s.RULES['maximum_soak_cycles'] or [r['cycle'] for r in cycles] != list(range(count)):
        raise ValueError('Cycle protocol differs')
    expected_order = [(cy, i) for cy in range(count) for i in s.indices(cy)]
    if [(r['cycle'], r['index']) for r in worker['rows']] != expected_order:
        raise ValueError('Missing or reordered cases')
    for row in worker['rows']:
        s.check_page(cases[row['index']], row['signature'], row['counters'], row['wide_calls'], True)
        old.summary([row['diagnostic_wall_s']])
    delta = p.counters(worker['execution'], worker['initial_execution'])
    if set(delta) != {'rec', 'det', 'layout', 'cell', 'yolo', 'enhance', 'ner'}:
        raise ValueError('Missing model')
    for role, counts in delta.items():
        for key, val in counts.items():
            if val != sum(r['counters'][role][key] for r in worker['rows']):
                raise ValueError('Total model counters differ')
    wide = sum(r['wide_calls'] for r in worker['rows'])
    if (worker['initial_execution']['rec']['wide_execution']['calls'] != 3
            or worker['execution']['rec']['wide_execution']['calls'] != 3 + wide):
        raise ValueError('Wide lifetime counters differ')
    warm = worker['after_warmup']
    if warm != cycles[0]['after_cycle']:
        raise ValueError('Warm baseline differs')
    end = cycles[-1]['after_cycle']['monotonic_s']
    elapsed = end - warm['monotonic_s']
    if (elapsed != worker['postwarm_s'] or elapsed < s.RULES['minimum_soak_s']
            or any(row['after_cycle']['monotonic_s'] - warm['monotonic_s'] >= s.RULES['minimum_soak_s'] for row in cycles[1:-1])):
        raise ValueError('Duration or stop rule differs')
    checks = worker['checkpoints']
    if (len(checks) != (count - 1) * 90 or any(not warm['monotonic_s'] <= r['monotonic_s'] <= end for r in checks)
            or any(a['monotonic_s'] >= b['monotonic_s'] for a, b in zip(checks, checks[1:]))):
        raise ValueError('Post-page resource sample coverage differs')
    resources = s.resources(warm, checks)
    if worker['resources'] != resources:
        raise ValueError('Resource statistics differ')
    samples = [r for r in run['samples'] if warm['monotonic_s'] <= r['monotonic_s'] <= end]
    stamps = [warm['monotonic_s']] + [r['monotonic_s'] for r in samples] + [end]
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    if not samples or min(gaps) < 0 or max(gaps) > 3 * s.RULES['sampling_interval_s']:
        raise ValueError('Telemetry sampling has gaps')
    sampled = dict(count=len(samples), max_gap_s=max(gaps),
        maximum_nvmap_kib=max(r['memory']['NvMapMemUsed_kib'] for r in samples),
        maximum_process_rss_kib=max(r['process_rss_kib'] for r in samples),
        minimum_mem_available_kib=min(r['memory']['MemAvailable_kib'] for r in samples),
        minimum_swap_free_kib=min(r['memory']['SwapFree_kib'] for r in samples),
        note='Five-second worker RSS/global GPU allocations, not exact peaks; not added together.')
    idle_growth = (run['after_exit']['memory']['NvMapMemUsed_kib'] - run['before_worker']['memory']['NvMapMemUsed_kib']) / 1024
    passed = (resources['passed'] and idle_growth <= s.RULES['maximum_idle_nvmap_growth_mib']
              and sampled['minimum_mem_available_kib'] >= s.RULES['minimum_mem_available_mib'] * 1024)
    return dict(status='bounded_soak_passed_not_http' if passed else 'bounded_soak_resource_not_accepted',
        images=45, cases=90, total_pages=len(worker['rows']), warmup_pages=90,
        postwarm_pages=len(checks), postwarm_cycles=count - 1, postwarm_s=elapsed,
        business_consistent=True, page_call_delta=delta, wide_page_calls=wide,
        resources=resources, periodic_samples=sampled, idle_nvmap_growth_mib=idle_growth,
        original_release_replaced=False, http_validated=False,
        note='Repeated existing data, sequential offline OCR+NER, no GT/new speedup/concurrency or multi-hour proof. Real near2304 and >2304 inputs remain absent. No in-process CPU-fallback residency stress.')


def archive(stage):
    if stage != s.STAGE:
        raise ValueError('Unexpected stage')
    for name in ('summary.json', 'final_audit.json', 'manifest.json', 'technical_evidence.tar.gz', 'archive_meta.json'):
        old.fresh(stage / name)
    plan = s.metadata()
    if s.c.inventory(s.c.DATA)[0] != plan['inventory']:
        raise ValueError('Input corpus changed')
    run, worker = n.read(stage / 'run.json'), n.read(stage / 'worker.json')
    if run['plan_sha256'] != q.digest(stage / 'plan.json') or run['worker_sha256'] != q.digest(stage / 'worker.json'):
        raise ValueError('Plan/worker report drift')
    result = summarize(plan, run, worker)
    audit = s.c.v.previous.audit(s.c.v.w.STAGE)
    pids = {run['parent_pid'], run['worker_pid']}
    live = [int(line.split()[0]) for line in p.prior.previous.command('ps', '-eo', 'pid,args').splitlines()[1:]
            if int(line.split()[0]) in pids or '/soak_rec_corpus.py ' in line]
    if live:
        raise ValueError('Experiment still running')
    audit.update(previous_audit_sha256=s.PRIOR_AUDIT_SHA, package=plan['package'], experiment_process_ids=live)
    q.save(stage / 'summary.json', result)
    q.save(stage / 'final_audit.json', audit)
    names = SCRIPTS + ('plan.json', 'run.json', 'worker.json', 'tests.log', 'prepare.log', 'run.log', 'summary.json', 'final_audit.json')
    paths = {name: stage / name for name in names}
    paths['previous_final_audit.json'] = s.c.STAGE / 'final_audit.json'
    q.save(stage / 'manifest.json', dict(files=p.prior.safe_files(paths, q.BASE / 'staging'),
        note='Technical allowlist, no input filenames/images/models/text/tensors/token IDs/license data.'))
    with (stage / 'technical_evidence.tar.gz').open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as tar:
            for name, path in paths.items():
                tar.add(path, arcname=name, recursive=False)
            tar.add(stage / 'manifest.json', arcname='manifest.json', recursive=False)
    meta = dict(bytes=(stage / 'technical_evidence.tar.gz').stat().st_size,
        sha256=q.digest(stage / 'technical_evidence.tar.gz'), members=len(paths) + 1,
        audit_sha256=q.digest(stage / 'final_audit.json'))
    q.save(stage / 'archive_meta.json', meta)
    print_result(meta, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    archive(parser.parse_args().stage.resolve())
