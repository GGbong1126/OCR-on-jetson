"""E125 coverage/strict quality gates and technical-only evidence archive."""
import argparse
from pathlib import Path
import tarfile
import validate_rec_corpus as c

q, n, old, p = c.q, c.n, c.old, c.p
SCRIPTS = c.previous.SCRIPTS + ('validate_rec_corpus.py', 'archive_rec_corpus.py', 'test_rec_corpus.py')
REPORTS = ('plan.json', 'run.json', 'reference.json', 'candidate.json')


def expected_route(shape, variant):
    if len(shape) != 4 or shape[:3] != [1, 3, 48]:
        return 'cpu'
    if 32 <= shape[3] <= 1536:
        return 'original_gpu'
    if variant == 'candidate' and 1537 <= shape[3] <= 2304:
        return 'wide_gpu'
    return 'cpu'


def summarize(plan, run, reports):
    if (plan['status'] != 'prepared' or plan['rules'] != c.RULES or run['status'] != 'completed'
            or run['rules'] != c.RULES or set(reports) != {'reference', 'candidate'}
            or [r['variant'] for r in run['workers']] != c.RULES['order']
            or len({r['pid'] for r in run['workers']}) != 2
            or plan['selected'] != c.select(plan['inventory'], set(plan['excluded_sha256']))):
        raise ValueError('Protocol/source selection differs')
    expected_cases = [dict(index=i, image_sha256=r['sha256'], image_type=kind)
        for i, (r, kind) in enumerate((r, kind) for r in plan['selected'] for kind in c.RULES['paths'])]
    if plan['cases'] != expected_cases or len(expected_cases) != 64:
        raise ValueError('Case plan differs')
    for worker in run['workers']:
        name = worker['variant']
        report = reports[name]
        if (worker['exitcode'] or report['pid'] != worker['pid'] or report['parent_pid'] != run['parent_pid']
                or report['pid'] == run['parent_pid'] or report['variant'] != name
                or report['spawn_method'] != 'spawn' or report['startup_environment'] != run['startup_environment']
                or report['status'] != 'completed_observed_not_gt' or report['rules'] != c.RULES
                or report['factory_sha256'] != plan['package']['serve_rec_wide.py']
                or report['adapter_sha256'] != plan['package']['rec_wide_adapter.py']
                or not report['all_gpu_sessions_closed'] or len(report['rows']) != 64):
            raise ValueError('Worker incomplete or package drift')
        if report['coverage'] != c.coverage(report['rows']):
            raise ValueError('Coverage calculation differs')
        cpu_seen = False
        for row, case in zip(report['rows'], expected_cases):
            if any(row[k] != val for k, val in case.items()):
                raise ValueError('Case order changed')
            old.summary([row['diagnostic_wall_s']])
            if [v['index'] for v in row['rec_calls']] != list(range(len(row['rec_calls']))):
                raise ValueError('Missing recognition observation')
            for call in row['rec_calls']:
                route = expected_route(call['input_shape'], name)
                if call['route'] != route or call['cold_cpu'] != (route == 'cpu' and not cpu_seen):
                    raise ValueError('Profile route or CPU lifetime differs')
                if route == 'cpu':
                    cpu_seen = True
            if (row['counters']['rec']['calls'] != sum(v['route'] != 'cpu' for v in row['rec_calls'])
                    or row['counters']['rec']['cpu_fallback_calls'] != sum(v['route'] == 'cpu' for v in row['rec_calls'])):
                raise ValueError('Recognition observer missed calls')
        delta = p.counters(report['execution'], report['initial_execution'])
        if set(delta) != {'det', 'rec', 'layout', 'cell', 'yolo', 'enhance', 'ner'}:
            raise ValueError('Missing model')
        for role, values in delta.items():
            for key, count in values.items():
                if count != sum(r['counters'][role][key] for r in report['rows']):
                    raise ValueError('Total calls differ')
    paired = []
    for a, b in zip(reports['reference']['rows'], reports['candidate']['rows']):
        same_nonrec_counts = {k: val for k, val in a['counters'].items() if k != 'rec'} == {
            k: val for k, val in b['counters'].items() if k != 'rec'}
        x, y = a['rec_calls'], b['rec_calls']
        same_inputs = len(x) == len(y) and all(
            (r['input_shape'], r['input_dtype'], r['input_sha256']) == (s['input_shape'], s['input_dtype'], s['input_sha256'])
            for r, s in zip(x, y))
        if not same_inputs:
            changed_ctc = None
            unexpected_output = None
        else:
            changed_ctc = [r['index'] for r, s in zip(x, y) if (r['ctc_sha256'], r['ctc_length']) != (s['ctc_sha256'], s['ctc_length'])]
            unexpected_output = [r['index'] for r, s in zip(x, y) if s['route'] != 'wide_gpu' and r['output_sha256'] != s['output_sha256']]
        sig_equal = a['signature'] == b['signature']
        good = sig_equal and same_nonrec_counts and same_inputs and not changed_ctc and not unexpected_output
        paired.append(dict(index=a['index'], image_sha256=a['image_sha256'], image_type=a['image_type'],
            signature_equal=sig_equal, same_nonrec_counts=same_nonrec_counts, same_rec_inputs=same_inputs,
            changed_ctc_indices=changed_ctc, unchanged_backend_output_drift_indices=unexpected_output,
            passed=bool(good)))
    passed = all(row['passed'] for row in paired)
    return dict(status='expanded_corpus_consistent_not_gt' if passed else 'expanded_corpus_not_accepted',
        inventory_images=len(plan['inventory']), selected_images=32, paired_cases=64,
        business_executions=128, passed_cases=sum(row['passed'] for row in paired),
        failed_case_indices=[r['index'] for r in paired if not r['passed']], pairs=paired,
        coverage={name: report['coverage'] for name, report in reports.items()},
        original_release_replaced=False, http_validated=False,
        note='32 existing images not used in prior13, selected before inference. Same corpus may contain related documents; no GT/independence/accuracy or speedup claimed. Per-rec hashing perturbs wall time. Snapshot resources are not peak or endurance acceptance.')


def validate_reports():
    plan = c.metadata()
    run = n.read(c.STAGE / 'run.json')
    if run['plan_sha256'] != q.digest(c.STAGE / 'plan.json'):
        raise ValueError('Plan hash changed')
    if run['reports_sha256'] != {name + '.json': q.digest(c.STAGE / (name + '.json')) for name in c.RULES['order']}:
        raise ValueError('Worker reports changed')
    reports = {name: n.read(c.STAGE / (name + '.json')) for name in c.RULES['order']}
    return summarize(plan, run, reports)


def archive(stage):
    if stage != c.STAGE:
        raise ValueError('Unexpected stage')
    for name in ('summary.json', 'final_audit.json', 'manifest.json', 'technical_evidence.tar.gz', 'archive_meta.json'):
        old.fresh(stage / name)
    current_inventory, _ = c.inventory(c.DATA)
    if current_inventory != n.read(stage / 'plan.json')['inventory']:
        raise ValueError('Image inventory changed before final audit')
    result = validate_reports()
    audit = c.v.previous.audit(c.v.w.STAGE)
    run = n.read(stage / 'run.json')
    pids = {run['parent_pid']} | {r['pid'] for r in run['workers']}
    live = [int(line.split()[0]) for line in p.prior.previous.command('ps', '-eo', 'pid,args').splitlines()[1:]
        if int(line.split()[0]) in pids or '/validate_rec_corpus.py ' in line]
    if live:
        raise ValueError('Experiment still running')
    audit.update(previous_audit_sha256=c.PRIOR_AUDIT_SHA, package=c.metadata()['package'], experiment_process_ids=live)
    q.save(stage / 'summary.json', result)
    q.save(stage / 'final_audit.json', audit)
    files = SCRIPTS + REPORTS + ('tests.log', 'prepare.log', 'run.log', 'summary.json', 'final_audit.json')
    paths = {name: stage / name for name in files}
    paths['previous_final_audit.json'] = c.v.STAGE / 'final_audit.json'
    q.save(stage / 'manifest.json', dict(files=p.prior.safe_files(paths, q.BASE / 'staging'),
        note='Technical allowlist; no image names, images, weights, tensors, recognized text, token IDs or license data.'))
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
    print({k: value for k, value in result.items() if k != 'pairs'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, type=Path)
    archive(parser.parse_args().stage.resolve())
