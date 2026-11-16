"""E124 independently checked offline overlay acceptance and technical archive."""
import argparse
from pathlib import Path
import tarfile
import validate_rec_wide_adapter as v

q, n, old, p = v.q, v.n, v.old, v.p
SCRIPTS = v.previous.SCRIPTS + ('rec_wide_adapter.py', 'serve_rec_wide.py',
    'validate_rec_wide_adapter.py', 'archive_rec_wide_adapter.py',
    'test_rec_wide_adapter.py', 'test_archive_rec_wide_adapter.py')
REPORTS = ('plan.json', 'run.json', 'off_before.json', 'on_soak.json', 'off_after.json')
LOGS = ('tests.log', 'tests_final.log', 'prepare.log', 'run.log')


def summarize(plan, run, reports):
    if (plan['status'] != 'prepared' or plan['rules'] != v.RULES or run['rules'] != v.RULES
            or run['status'] != 'completed_offline' or len(plan['cases']) != 26
            or set(reports) != set(v.RULES['order'])
            or [r['variant'] for r in run['workers']] != v.RULES['order']
            or len({r['pid'] for r in run['workers']}) != 3):
        raise ValueError('Incomplete protocol/worker isolation')
    totals = {}
    for worker in run['workers']:
        variant = worker['variant']
        x = reports[variant]
        enabled = variant == 'on_soak'
        if (worker['exitcode'] or x['pid'] != worker['pid'] or x['parent_pid'] != run['parent_pid']
                or x['pid'] == run['parent_pid'] or x['variant'] != variant
                or x['status'] != 'completed_offline_not_http' or x['rules'] != v.RULES
                or x['spawn_method'] != 'spawn' or x['startup_environment'] != run['startup_environment']
                or x['overlay_loaded'] != enabled or not x['all_gpu_sessions_closed']
                or x['loaded_factory_sha256'] != plan['package']['serve_rec_wide.py']
                or x['loaded_adapter_sha256'] != plan['package']['rec_wide_adapter.py']):
            raise ValueError('Worker/package/cleanup differs')
        cycles = v.RULES['warmup_cycles'] + v.RULES['soak_cycles'] if enabled else v.RULES['off_cycles']
        expected_order = [(c, i) for c in range(cycles) for i in (range(26) if c % 2 == 0 else reversed(range(26)))]
        if [(r['cycle'], r['index']) for r in x['rows']] != expected_order:
            raise ValueError('Page coverage differs')
        for row in x['rows']:
            case = plan['cases'][row['index']]
            if (row['signature'] != case['signature']
                    or row['counters'] != v.w.expected_counters(case, 'candidate' if enabled else 'reference')
                    or row['wide_calls'] != (case['counters']['rec']['cpu_fallback_calls'] if enabled else 0)):
                raise ValueError('Business or per-page route differs')
            old.summary([row['wall_ms']])
        delta = p.counters(x['page_execution'], x['after_load_execution'])
        for role, counts in delta.items():
            for name, count in counts.items():
                if count != sum(r['counters'][role][name] for r in x['rows']):
                    raise ValueError('Total model invocation counts differ')
        if set(delta) != {'det', 'rec', 'layout', 'cell', 'yolo', 'enhance', 'ner'}:
            raise ValueError('Incomplete business modules')
        p.counters(x['execution'], x['page_execution'])  # Reject final error latches too.
        totals[variant] = dict(pages=len(x['rows']), page_call_delta=delta,
                               wide_page_calls=sum(r['wide_calls'] for r in x['rows']),
                               process_wall_s=x['process_wall_s'])
        if enabled:
            if len(x['checkpoints']) != v.RULES['soak_cycles'] * 26:
                raise ValueError('Resource sample coverage differs')
            resources = v.resource_summary(x['after_warmup'], x['checkpoints'])
            if resources != x['resources']:
                raise ValueError('Resource calculation differs')
            if [r['width'] for r in x['boundaries']] != v.RULES['boundaries']:
                raise ValueError('Boundary coverage differs')
            for row in x['boundaries']:
                width = row['width']
                route = 'original_gpu' if width <= 1536 else 'wide_gpu' if width <= 2304 else 'cpu'
                wanted = dict(original=2 if route == 'original_gpu' else 0,
                              wide=2 if route == 'wide_gpu' else 0, cpu=2 if route == 'cpu' else 0)
                if (not row['passed'] or not row['token_sequence_equal'] or not row['allclose_diagnostic']
                        or not row['repeated_exact'] or row['route'] != route or row['call_delta'] != wanted):
                    raise ValueError('Boundary route/quality differs')
            rec, loaded, page_rec = (x[k]['rec'] for k in ('execution', 'after_load_execution', 'page_execution'))
            if (loaded['wide_execution']['calls'] != 3
                    or page_rec['wide_execution']['calls'] != 3 + totals[variant]['wide_page_calls']
                    or rec['wide_execution']['calls'] != page_rec['wide_execution']['calls'] + 4
                    or rec['original_execution']['calls'] != page_rec['original_execution']['calls'] + 2
                    or rec['cpu_fallback']['calls'] != page_rec['cpu_fallback']['calls'] + 2
                    or rec['wide_profiles'] != {'x': v.w.PROFILE}):
                raise ValueError('Warmup/boundary lifetime usage differs')
        elif 'wide_execution' in x['page_execution']['rec'] or x.get('boundaries'):
            raise ValueError('Rollback still uses candidate')
    resources = reports['on_soak']['resources']
    return dict(status='offline_adapter_validated_not_http_enabled' if resources['passed'] else 'offline_resource_gate_not_met',
                total_pages=sum(r['pages'] for r in totals.values()), workers=totals,
                resources=resources, boundary_cases=4, boundary_cpu_calls=2,
                rollback_verified=True, original_release_replaced=False, http_validated=False,
                note='Existing 13 images, repeated offline OCR+NER, no GT or independent data. Four soak cycles after one warm cycle; not multi-hour endurance or HTTP. Width2304/2305 are synthetic, not representative text acceptance. No new speedup claim.')


def archive(stage):
    if stage != v.STAGE:
        raise ValueError('Unexpected stage')
    for name in ('summary.json', 'final_audit.json', 'manifest.json', 'technical_evidence.tar.gz', 'archive_meta.json'):
        old.fresh(stage / name)
    plan = v.metadata()
    meta = v.provenance()
    if any(plan[k] != value for k, value in meta.items()):
        raise ValueError('Evidence source changed')
    run = n.read(stage / 'run.json')
    if run['reports_sha256'] != {name + '.json': q.digest(stage / (name + '.json')) for name in v.RULES['order']}:
        raise ValueError('Worker reports changed')
    reports = {name: n.read(stage / (name + '.json')) for name in v.RULES['order']}
    result = summarize(plan, run, reports)
    audit = v.previous.audit(v.w.STAGE)
    command = p.prior.previous.command
    pids = {run['parent_pid']} | {r['pid'] for r in run['workers']}
    live = [int(line.split()[0]) for line in command('ps', '-eo', 'pid,args').splitlines()[1:]
            if int(line.split()[0]) in pids or '/validate_rec_wide_adapter.py ' in line]
    if live:
        raise ValueError('E124 experiment still active')
    audit.update(previous_audit_sha256=v.PRIOR_AUDIT_SHA, experiment_process_ids=live,
                 package=plan['package'], overlay=str(v.OVERLAY), original_release_replaced=False)
    q.save(stage / 'summary.json', result)
    q.save(stage / 'final_audit.json', audit)
    paths = {name: stage / name for name in SCRIPTS + REPORTS + LOGS + ('summary.json', 'final_audit.json')}
    paths['overlay_config.json'] = v.OVERLAY / 'config.json'
    paths['previous_final_audit.json'] = v.w.STAGE / 'final_audit.json'
    rows = p.prior.safe_files(paths, q.BASE)
    q.save(stage / 'manifest.json', dict(files=rows, note='Technical allowlist only. No engine/weights/images/tensors/text/tokens/license.'))
    with (stage / 'technical_evidence.tar.gz').open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as tar:
            for name, path in paths.items():
                tar.add(path, arcname=name, recursive=False)
            tar.add(stage / 'manifest.json', arcname='manifest.json', recursive=False)
    meta = dict(bytes=(stage / 'technical_evidence.tar.gz').stat().st_size,
                sha256=q.digest(stage / 'technical_evidence.tar.gz'), members=len(paths) + 1,
                audit_sha256=q.digest(stage / 'final_audit.json'))
    q.save(stage / 'archive_meta.json', meta)
    print(meta)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    archive(parser.parse_args().stage.resolve())
