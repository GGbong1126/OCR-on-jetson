"""E109 technical-only archive and unchanged original release audit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import summarize_enhance_batch2 as summary
q = summary.e.q

REQUIRED = ('build_enhance_batch2.py', 'evaluate_enhance_batch2.py',
    'summarize_enhance_batch2.py', 'archive_enhance_batch2.py',
    'build_enhance_fp16.py', 'build_enhance_fp32.py', 'evaluate_enhance_fp32.py',
    'optimize_enhance_fp16.py', 'optimize_rec_fp16.py', 'quantize_yolo_ptq.py',
    'diagnose_rec_difference.py', 'test_build_enhance_batch2.py',
    'test_evaluate_enhance_batch2.py', 'test_summarize_enhance_batch2.py',
    'test_enhance_fp16_trial.py', 'build_tests.log', 'evaluate_tests.log',
    'summary_tests.log', 'legacy_tests.log', 'resume.json', 'export.json', 'export.log',
    'build.json', 'build.log', 'check.json', 'check.log', 'layers.json',
    'evaluation.json', 'evaluation.log', 'summary.json', 'final_audit.json')
OPTIONAL = ('benchmark_runner.log',) + tuple('benchmark_%s_%s.%s' % (v, r, ext)
                 for v in ('reference', 'candidate') for r in ('r1', 'r2') for ext in ('json', 'log'))


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def run(stage):
    if not stage.is_relative_to(q.BASE/'staging') or not stage.is_dir():
        raise ValueError('Isolated stage required')
    for n in ('final_audit.json', 'manifest.json', 'technical_evidence.tar.gz', 'archive_meta.json'):
        if (stage/n).exists():
            raise ValueError('Refuse overwrite')
    if summary.from_stage(stage) != json.loads((stage/'summary.json').read_text()):
        raise ValueError('Summary not reproduced')
    previous = json.loads((q.BASE/'staging/enhance_fp32_e108_r1/final_audit.json').read_text())
    path = q.RELEASE/'edge_deploy/jetson/deployment.json'
    cfg = json.loads(path.read_text())
    engines = {k: q.digest(q.RELEASE/v['path']) for k, v in cfg['engines'].items()}
    plugins = {p['path']: q.digest(q.RELEASE/p['path']) for v in cfg['engines'].values() for p in v.get('plugins', [])}
    if q.digest(path) != previous['deployment_sha256'] or engines != previous['engines'] or plugins != previous['plugins']:
        raise ValueError('Original release changed')
    install = q.BASE/'staging/jp461-r1/install_audit.json'
    norm = lambda v: sorted(tuple(line.split()) for line in v.splitlines() if line.strip())
    packages = norm(command('dpkg-query', '-W', '-f=${Package}\t${Version}\n', 'nvidia-l4t-*'))
    if packages != norm(json.loads(install.read_text())['l4t_packages_before']):
        raise ValueError('BSP changed')
    unit = command('systemctl', 'show', 'autoocr-jetson.service', '-p', 'MainPID', '-p', 'ActiveState', '-p', 'UnitFileState')
    listeners = command('ss', '-ltn', 'sport = :5051')
    mount = command('findmnt', '-n', '-o', 'TARGET,SOURCE,FSTYPE,OPTIONS', '-T', str(q.BASE))
    if (set(unit.splitlines()) != {'MainPID=0', 'ActiveState=inactive', 'UnitFileState=disabled'}
            or len(listeners.splitlines()) != 1 or mount.split() != previous['mount'].split()):
        raise ValueError('Service/mount state differs')
    tokens = ('build_enhance_batch2.py build', 'build_enhance_batch2.py export',
              'evaluate_enhance_batch2.py check', 'evaluate_enhance_batch2.py evaluate',
              'evaluate_enhance_batch2.py benchmark')
    processes = [int(line.split()[0]) for line in command('ps', '-eo', 'pid,args').splitlines()
                 if any(token in line for token in tokens)]
    if processes:
        raise ValueError('Experiment running')
    resume = json.loads((stage/'resume.json').read_text())
    origin = q.BASE/'staging/enhance_batch2_e109_r1'
    if any(q.digest(origin/n) != v for n, v in resume['preserved_origin_sha256'].items()):
        raise ValueError('Original partial experiment changed')
    names = REQUIRED + tuple(n for n in OPTIONAL if (stage/n).exists())
    audit = dict(deployment_sha256=q.digest(path), engines=engines, plugins=plugins,
        systemd=unit, listeners=listeners, mount=mount, l4t_packages_unchanged=True,
        installation_audit_sha256=q.digest(install),
        l4t_package_rows_sha256=hashlib.sha256(json.dumps(packages, separators=(',', ':')).encode()).hexdigest(),
        experiment_process_ids=processes, memory=q.memory(), storage=shutil.disk_usage(q.BASE)._asdict(),
        scripts_sha256={n: q.digest(stage/n) for n in names if n.endswith('.py')},
        preserved_partial_experiment=True,
        note='Idle snapshots, not peak. Original release/BSP/mount unchanged; no HTTP/activation/clock/power changes or automatic promotion.')
    q.save(stage/'final_audit.json', audit)
    entries = []
    for n in names:
        p = stage/n
        if p.is_symlink() or not p.is_file() or p.resolve().parent != stage:
            raise ValueError('Unsafe archive member')
        entries.append(dict(name=n, bytes=p.stat().st_size, sha256=q.digest(p)))
    q.save(stage/'manifest.json', dict(files=entries,
        note='Explicit technical-only allowlist; no model/image/tensor/recognized text/license/runtime.'))
    archive = stage/'technical_evidence.tar.gz'
    with archive.open('xb') as f:
        with tarfile.open(fileobj=f, mode='w:gz') as tar:
            for n in (*names, 'manifest.json'):
                tar.add(stage/n, arcname=n, recursive=False)
    meta = dict(members=len(names)+1, bytes=archive.stat().st_size, sha256=q.digest(archive))
    q.save(stage/'archive_meta.json', meta)
    print(json.dumps(meta))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, required=True)
    run(p.parse_args().stage.resolve())
