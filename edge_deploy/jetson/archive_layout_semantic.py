"""E112 explicit technical evidence archive, unchanged-release/BSP audit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import summarize_layout_semantic as s
m=s.m;q=m.q
REQUIRED=('layout_semantic_trial.py','summarize_layout_semantic.py','archive_layout_semantic.py',
    'test_layout_semantic_trial.py','test_summarize_layout_semantic.py',
    'layout_outputs_fp32.py','evaluate_layout_outputs.py','diagnose_layout_outputs.py',
    'layout_hotspot_mixed.py','evaluate_layout_hotspot.py','optimize_rec_fp16.py','quantize_yolo_ptq.py',
    'plan.json','prepare.log','quality.json','quality.log','layers.json','tests.log','tests_complete.log',
    'summary.json','final_audit.json')
OPTIONAL=tuple('benchmark_%s_%s.%s'%(v,r,ext) for v in ('reference','candidate') for r in ('r1','r2') for ext in ('json','log'))
PRIOR=('prepare.json','build.json','check.json','layers.json','output_diagnostic.json','business_diagnostic.json',
       'final_audit.json','manifest.json','archive_meta.json')


def command(*args):return subprocess.check_output(args,text=True).strip()


def run(stage):
    if stage!=q.BASE/'staging/layout_semantic_e112_r1' or not stage.is_dir():raise ValueError('Unexpected stage')
    for n in ('final_audit.json','manifest.json','technical_evidence.tar.gz','archive_meta.json'):m.old.fresh(stage/n)
    if s.from_stage(stage)!=m.read(stage/'summary.json'):raise ValueError('Summary differs')
    previous=m.read(m.PRIOR/'final_audit.json');path=q.RELEASE/'edge_deploy/jetson/deployment.json';cfg=m.read(path)
    engines={k:q.digest(q.RELEASE/v['path']) for k,v in cfg['engines'].items()}
    plugins={p['path']:q.digest(q.RELEASE/p['path']) for v in cfg['engines'].values() for p in v.get('plugins',[])}
    if q.digest(path)!=previous['deployment_sha256'] or engines!=previous['engines'] or plugins!=previous['plugins']:
        raise ValueError('Original release changed')
    install=q.BASE/'staging/jp461-r1/install_audit.json'
    norm=lambda v:sorted(tuple(line.split()) for line in v.splitlines() if line.strip())
    packages=norm(command('dpkg-query','-W','-f=${Package}\t${Version}\n','nvidia-l4t-*'))
    if packages!=norm(m.read(install)['l4t_packages_before']):raise ValueError('BSP changed')
    unit=command('systemctl','show','autoocr-jetson.service','-p','MainPID','-p','ActiveState','-p','UnitFileState')
    listeners=command('ss','-ltn','sport = :5051');mount=command('findmnt','-n','-o','TARGET,SOURCE,FSTYPE,OPTIONS','-T',str(q.BASE))
    if (set(unit.splitlines())!={'MainPID=0','ActiveState=inactive','UnitFileState=disabled'}
            or len(listeners.splitlines())!=1 or mount.split()!=previous['mount'].split()):raise ValueError('Service/mount changed')
    processes=[int(line.split()[0]) for line in command('ps','-eo','pid,args').splitlines() if 'layout_semantic_trial.py ' in line]
    if processes:raise ValueError('Experiment still running')
    members={n:stage/n for n in REQUIRED+tuple(n for n in OPTIONAL if (stage/n).exists())}
    members.update({'e111_'+n:m.PRIOR/n for n in PRIOR})
    audit=dict(deployment_sha256=q.digest(path),engines=engines,plugins=plugins,systemd=unit,listeners=listeners,mount=mount,
        l4t_packages_unchanged=True,installation_audit_sha256=q.digest(install),previous_audit_sha256=q.digest(m.PRIOR/'final_audit.json'),
        l4t_package_rows_sha256=hashlib.sha256(json.dumps(packages,separators=(',',':')).encode()).hexdigest(),
        prior_archive_unchanged=q.digest(m.PRIOR/'technical_evidence.tar.gz')==m.PRIOR_ARCHIVE_SHA,
        candidate_engine_sha256=q.digest(m.PRIOR/m.e.b.ENGINE),experiment_process_ids=processes,
        memory=q.memory(),storage=shutil.disk_usage(q.BASE)._asdict(),
        scripts_sha256={n:q.digest(p) for n,p in members.items() if n.endswith('.py')},
        note='Original release/BSP/mount and E111 history unchanged; no HTTP/license/clock/power changes. Idle snapshot is not service peak.')
    q.save(stage/'final_audit.json',audit);entries=[]
    for n,p in members.items():
        if Path(n).name!=n or p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(q.BASE/'staging'):
            raise ValueError('Unsafe member')
        entries.append(dict(name=n,bytes=p.stat().st_size,sha256=q.digest(p)))
    q.save(stage/'manifest.json',dict(files=entries,note='Technical explicit allowlist only; no model, images, tensors, document text, runtime or license.'))
    archive=stage/'technical_evidence.tar.gz'
    with archive.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as tar:
            for n,p in members.items():tar.add(p,arcname=n,recursive=False)
            tar.add(stage/'manifest.json',arcname='manifest.json',recursive=False)
    meta=dict(members=len(entries)+1,bytes=archive.stat().st_size,sha256=q.digest(archive))
    q.save(stage/'archive_meta.json',meta);print(json.dumps(meta))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,type=Path)
    run(p.parse_args().stage.resolve())
