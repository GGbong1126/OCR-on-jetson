"""E106 read-only release audit and explicit technical-only evidence archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import decompose_det_error as d
import summarize_det_error as summary

q=d.q
NAMES=('probe_det_precision_boundary.py','diagnose_det_boundary.py','det_hotspot_mixed.py',
       'optimize_rec_fp16.py','quantize_yolo_ptq.py','decompose_det_error.py',
       'probe_det_alternative.py','evaluate_det_alternative.py','summarize_det_error.py',
       'archive_det_error.py','test_decompose_det_error.py','test_probe_det_alternative.py',
       'tests.log','alternative_tests.log','decomposition.json','decomposition.log',
       'alternative_build.json','alternative_build.log','alternative_check.json',
       'alternative_check.log','alternative_layers.json','summary.json','final_audit.json')


def command(*args):return subprocess.check_output(args,text=True).strip()


def run(stage):
    if not stage.is_relative_to(q.BASE/'staging') or not stage.is_dir():raise ValueError('Isolated stage required')
    if summary.summarize(stage)!=json.loads((stage/'summary.json').read_text()):raise ValueError('Summary does not reproduce')
    prior=json.loads((d.PRIOR/'final_audit.json').read_text())
    path=q.RELEASE/'edge_deploy/jetson/deployment.json';config=json.loads(path.read_text())
    engines={k:q.digest(q.RELEASE/v['path']) for k,v in config['engines'].items()}
    plugins={p['path']:q.digest(q.RELEASE/p['path']) for v in config['engines'].values() for p in v.get('plugins',[])}
    if q.digest(path)!=prior['deployment_sha256'] or engines!=prior['engines'] or plugins!=prior['plugins']:raise ValueError('Release changed')
    install=q.BASE/'staging/jp461-r1/install_audit.json'
    packages=command('dpkg-query','-W','-f=${Package}\t${Version}\n','nvidia-l4t-*')
    old=json.loads(install.read_text())['l4t_packages_before']
    norm=lambda value:sorted(tuple(line.split()) for line in value.splitlines() if line.strip())
    if norm(packages)!=norm(old):raise ValueError('BSP changed')
    unit=command('systemctl','show','autoocr-jetson.service','-p','MainPID','-p','ActiveState','-p','UnitFileState')
    listeners=command('ss','-ltn','sport = :5051')
    mount=command('findmnt','-n','-o','TARGET,SOURCE,FSTYPE,OPTIONS','-T',str(q.BASE))
    if set(unit.splitlines())!={'MainPID=0','ActiveState=inactive','UnitFileState=disabled'} or len(listeners.splitlines())!=1:raise ValueError('Unexpected service state')
    if mount.split()!=prior['mount'].split():raise ValueError('Mount changed')
    procs=[]
    for line in command('ps','-eo','pid,args').splitlines():
        if any(name in line for name in ('decompose_det_error.py --stage','probe_det_alternative.py --stage','evaluate_det_alternative.py --stage')):
            procs.append(int(line.split()[0]))
    if procs:raise ValueError('Experiment still running')
    d.prior_reports()
    audit=dict(deployment_sha256=q.digest(path),engines=engines,plugins=plugins,systemd=unit,listeners=listeners,
        mount=mount,l4t_packages_unchanged=True,installation_audit_sha256=q.digest(install),
        l4t_packages_sha256=hashlib.sha256(packages.encode()).hexdigest(),experiment_process_ids=procs,
        memory=q.memory(),storage=shutil.disk_usage(q.BASE)._asdict(),
        scripts_sha256={name:q.digest(stage/name) for name in NAMES if name.endswith('.py')},
        note='Idle snapshot, not service/peak memory. Original release, prior micro-engines, BSP and mount unchanged; no HTTP/activation/power/clock changes. Candidate remains isolated.')
    q.save(stage/'final_audit.json',audit)
    members=[]
    for name in NAMES:
        p=stage/name
        if p.is_symlink() or not p.is_file() or p.resolve().parent!=stage:raise ValueError('Unsafe archive member')
        members.append(dict(name=name,bytes=p.stat().st_size,sha256=q.digest(p)))
    q.save(stage/'manifest.json',dict(files=members,note='Explicit technical allowlist; no engine/weights/image/tensor/OCR text/license/runtime.'))
    archive=stage/'technical_evidence.tar.gz'
    with archive.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as tar:
            for name in (*NAMES,'manifest.json'):tar.add(stage/name,arcname=name,recursive=False)
    meta=dict(members=len(NAMES)+1,bytes=archive.stat().st_size,sha256=q.digest(archive))
    q.save(stage/'archive_meta.json',meta);print(json.dumps(meta))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,type=Path)
    a=p.parse_args();run(a.stage.resolve())
