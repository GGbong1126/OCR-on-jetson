"""E122 technical-only evidence and immutable original deployment audit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import profile_rec_fallback as d

q, n, old = d.q, d.n, d.old
SCRIPTS = d.previous.SCRIPTS + ('profile_rec_fallback.py', 'archive_rec_fallback.py',
                              'test_profile_rec_fallback.py', 'test_archive_rec_fallback.py')
REPORTS = ('plan.json', 'diagnostic.json', 'tests.log', 'tests_final.log',
           'prepare.log', 'measure.log', 'summary.json', 'final_audit.json')


def validate_reports(stage):
    meta = d.provenance()
    plan, report = n.read(stage / 'plan.json'), n.read(stage / 'diagnostic.json')
    if plan['status'] != 'prepared' or any(plan[k] != v or report[k] != v for k, v in meta.items()):
        raise ValueError('Source/provenance drift')
    if report['plan_sha256'] != q.digest(stage / 'plan.json'):
        raise ValueError('Plan changed')
    result = d.summarize(plan, report)
    if result != report['summary']:
        raise ValueError('Stored summary differs')
    return result


def audit(stage):
    prior = n.read(d.PRIOR / 'final_audit.json')
    cfg_path = q.RELEASE / 'edge_deploy/jetson/deployment.json'
    cfg = n.read(cfg_path)
    engines = {k: q.digest(q.RELEASE / v['path']) for k, v in cfg['engines'].items()}
    plugins = {v['path']: q.digest(q.RELEASE / v['path']) for r in cfg['engines'].values() for v in r.get('plugins', [])}
    if engines != prior['engines'] or plugins != prior['plugins'] or q.digest(cfg_path) != prior['deployment_sha256']:
        raise ValueError('Original release changed')
    command = d.p.prior.previous.command
    packages = sorted(tuple(line.split()) for line in command('dpkg-query', '-W', '-f=${Package}\t${Version}\n', 'nvidia-l4t-*').splitlines() if line.strip())
    pkgsha = hashlib.sha256(json.dumps(packages, separators=(',', ':')).encode()).hexdigest()
    mount = command('findmnt', '-n', '-o', 'TARGET,SOURCE,FSTYPE,OPTIONS', '-T', str(q.BASE))
    systemd = command('systemctl', 'show', 'autoocr-jetson.service', '-p', 'MainPID', '-p', 'ActiveState', '-p', 'UnitFileState')
    listeners = command('ss', '-ltn', 'sport = :5051')
    if pkgsha != prior['l4t_package_rows_sha256'] or mount.split() != prior['mount'].split() or set(systemd.splitlines()) != {'MainPID=0', 'ActiveState=inactive', 'UnitFileState=disabled'} or len(listeners.splitlines()) != 1:
        raise ValueError('BSP/mount/service changed')
    pid = n.read(stage / 'diagnostic.json')['pid']
    processes = [int(line.split()[0]) for line in command('ps', '-eo', 'pid,args').splitlines()[1:]
                 if int(line.split()[0]) == pid or '/profile_rec_fallback.py ' in line]
    if processes:
        raise ValueError('Diagnostic still running')
    return dict(deployment_sha256=q.digest(cfg_path), engines=engines, plugins=plugins, mount=mount,
                systemd=systemd, listeners=listeners, l4t_package_rows_sha256=pkgsha,
                previous_audit_sha256=d.AUDIT_SHA, experiment_process_ids=processes,
                memory=q.memory(), storage=shutil.disk_usage(q.BASE)._asdict(),
                note='Original seven models/config/plugin/BSP/mount unchanged; no service, activation, power or clock changes. Idle memory snapshot, not peak.')


def archive(stage):
    if stage != d.STAGE:
        raise ValueError('Unexpected stage')
    for name in ('summary.json', 'final_audit.json', 'manifest.json', 'technical_evidence.tar.gz', 'archive_meta.json'):
        old.fresh(stage / name)
    result = validate_reports(stage)
    q.save(stage / 'final_audit.json', audit(stage))
    q.save(stage / 'summary.json', result)
    paths = {name: stage / name for name in SCRIPTS + REPORTS}
    paths['previous_final_audit.json'] = d.PRIOR / 'final_audit.json'
    paths['previous_reference.json'] = d.PRIOR / 'pipeline_reference_r1.json'
    rows = d.p.prior.safe_files(paths, q.BASE / 'staging')
    q.save(stage / 'manifest.json', dict(files=rows, note='Technical allowlist only; no images, tensors, text, CTC class sequences, models or license data.'))
    with (stage / 'technical_evidence.tar.gz').open('xb') as f:
        with tarfile.open(fileobj=f, mode='w:gz') as archive_file:
            for name, path in paths.items():
                archive_file.add(path, arcname=name, recursive=False)
            archive_file.add(stage / 'manifest.json', arcname='manifest.json', recursive=False)
    meta = dict(bytes=(stage / 'technical_evidence.tar.gz').stat().st_size,
                sha256=q.digest(stage / 'technical_evidence.tar.gz'), members=len(paths) + 1,
                audit_sha256=q.digest(stage / 'final_audit.json'))
    q.save(stage / 'archive_meta.json', meta)
    print(json.dumps(meta))
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    archive(parser.parse_args().stage.resolve())
