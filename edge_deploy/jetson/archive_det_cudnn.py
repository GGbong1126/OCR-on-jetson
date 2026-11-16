"""E107 unchanged-release audit and strictly technical allowlist archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import build_det_cudnn as build
import summarize_det_cudnn as summary

q=build.q
REQUIRED=('build_det_cudnn.py','evaluate_det_cudnn.py','summarize_det_cudnn.py','archive_det_cudnn.py',
          'test_build_det_cudnn.py','test_evaluate_det_cudnn.py','test_evaluate_det_hotspot.py','test_summarize_det_hotspot.py',
          'probe_det_precision_boundary.py','diagnose_det_boundary.py','det_hotspot_mixed.py','optimize_rec_fp16.py',
          'quantize_yolo_ptq.py','decompose_det_error.py','probe_det_alternative.py','evaluate_det_hotspot.py','summarize_det_hotspot.py',
          'build_tests.log','evaluate_tests.log','legacy_tests.log','ranges.json','build.json','build.log','summary.json','final_audit.json')
OPTIONAL=('check.json','check.log','layers.json','evaluation.json','evaluation.log')+tuple(
    'benchmark_%s_%s.%s'%(v,r,e) for v in ('reference','mixed') for r in ('r1','r2') for e in ('json','log'))


def command(*args):return subprocess.check_output(args,text=True).strip()


def run(stage):
    if not stage.is_relative_to(q.BASE/'staging') or not stage.is_dir():raise ValueError('Isolated stage required')
    if summary.summarize(stage)!=json.loads((stage/'summary.json').read_text()):raise ValueError('Summary does not reproduce')
    prior=json.loads((build.PRIOR/'final_audit.json').read_text())
    path=q.RELEASE/'edge_deploy/jetson/deployment.json';config=json.loads(path.read_text())
    engines={k:q.digest(q.RELEASE/v['path']) for k,v in config['engines'].items()}
    plugins={p['path']:q.digest(q.RELEASE/p['path']) for v in config['engines'].values() for p in v.get('plugins',[])}
    if q.digest(path)!=prior['deployment_sha256'] or engines!=prior['engines'] or plugins!=prior['plugins']:raise ValueError('Release changed')
    install=q.BASE/'staging/jp461-r1/install_audit.json'
    packages=command('dpkg-query','-W','-f=${Package}\t${Version}\n','nvidia-l4t-*')
    norm=lambda value:sorted(tuple(line.split()) for line in value.splitlines() if line.strip())
    if norm(packages)!=norm(json.loads(install.read_text())['l4t_packages_before']):raise ValueError('BSP changed')
    unit=command('systemctl','show','autoocr-jetson.service','-p','MainPID','-p','ActiveState','-p','UnitFileState')
    listeners=command('ss','-ltn','sport = :5051')
    mount=command('findmnt','-n','-o','TARGET,SOURCE,FSTYPE,OPTIONS','-T',str(q.BASE))
    if set(unit.splitlines())!={'MainPID=0','ActiveState=inactive','UnitFileState=disabled'} or len(listeners.splitlines())!=1:raise ValueError('Service state changed')
    if mount.split()!=prior['mount'].split():raise ValueError('Mount changed')
    processes=[int(line.split()[0]) for line in command('ps','-eo','pid,args').splitlines()
               if any(name in line for name in ('build_det_cudnn.py --stage','evaluate_det_cudnn.py check','evaluate_det_cudnn.py evaluate','evaluate_det_cudnn.py benchmark'))]
    if processes:raise ValueError('Experiment still running')
    names=REQUIRED+tuple(n for n in OPTIONAL if (stage/n).exists())
    audit=dict(deployment_sha256=q.digest(path),engines=engines,plugins=plugins,systemd=unit,listeners=listeners,
        mount=mount,l4t_packages_unchanged=True,installation_audit_sha256=q.digest(install),
        l4t_package_rows_sha256=hashlib.sha256(json.dumps(norm(packages),separators=(',',':')).encode()).hexdigest(),
        experiment_process_ids=processes,memory=q.memory(),storage=shutil.disk_usage(q.BASE)._asdict(),
        scripts_sha256={n:q.digest(stage/n) for n in names if n.endswith('.py')},
        note='Idle memory only, not service peak. Seven engines/plugin/deployment/BSP/existing mount unchanged; no HTTP/activation, clocks/power, Git or release promotion.')
    q.save(stage/'final_audit.json',audit)
    entries=[]
    for n in names:
        p=stage/n
        if p.is_symlink() or not p.is_file() or p.resolve().parent!=stage:raise ValueError('Unsafe technical member')
        entries.append(dict(name=n,bytes=p.stat().st_size,sha256=q.digest(p)))
    q.save(stage/'manifest.json',dict(files=entries,note='Technical allowlist only; excludes engine, model, image, tensor, OCR text, license and runtime.'))
    archive=stage/'technical_evidence.tar.gz'
    with archive.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as tar:
            for n in (*names,'manifest.json'):tar.add(stage/n,arcname=n,recursive=False)
    meta=dict(members=len(names)+1,bytes=archive.stat().st_size,sha256=q.digest(archive))
    q.save(stage/'archive_meta.json',meta);print(json.dumps(meta))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,type=Path);a=p.parse_args();run(a.stage.resolve())
