"""E110 technical allowlist archive, original release/BSP/mount audit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import summarize_layout_hotspot as s
q=s.e.q

REQUIRED=('layout_hotspot_mixed.py','evaluate_layout_hotspot.py','optimize_rec_fp16.py',
          'quantize_yolo_ptq.py','summarize_layout_hotspot.py','archive_layout_hotspot.py',
          'test_layout_hotspot_mixed.py','test_evaluate_layout_hotspot.py','test_summarize_layout_hotspot.py',
          'tests.log','summary_tests.log','ranges.json','ranges.log','build.json','build.log','summary.json','final_audit.json')
OPTIONAL=('check.json','check.log','layers.json','evaluation.json','evaluation.log','benchmark_layout_hotspot.py',
          'diagnose_layout_hotspot.py','test_diagnose_layout_hotspot.py','diagnostic_tests.log','precision_diagnostic.json')+tuple(
    'benchmark_%s_%s.%s'%(v,r,ext) for v in ('reference','candidate') for r in ('r1','r2') for ext in ('json','log'))
PRIOR={'r1':('layout_hotspot_mixed.py','ranges.json','ranges.log'),
       'r2':('layout_hotspot_mixed.py','ranges.json','ranges.log','build.json','build.log')}


def command(*args): return subprocess.check_output(args,text=True).strip()


def run(stage):
    if stage!=q.BASE/'staging/layout_hotspot_e110_r3' or not stage.is_dir(): raise ValueError('Expected isolated stage')
    for n in ('final_audit.json','manifest.json','technical_evidence.tar.gz','archive_meta.json'):
        if (stage/n).exists(): raise ValueError('Refuse overwrite')
    if s.from_stage(stage)!=json.loads((stage/'summary.json').read_text()): raise ValueError('Summary differs')
    previous=json.loads((q.BASE/'staging/enhance_batch2_e109_r2/final_audit.json').read_text())
    path=q.RELEASE/'edge_deploy/jetson/deployment.json'; cfg=json.loads(path.read_text())
    engines={k:q.digest(q.RELEASE/v['path']) for k,v in cfg['engines'].items()}
    plugins={p['path']:q.digest(q.RELEASE/p['path']) for v in cfg['engines'].values() for p in v.get('plugins',[])}
    if (q.digest(path)!=previous['deployment_sha256'] or engines!=previous['engines'] or plugins!=previous['plugins']):
        raise ValueError('Original release changed')
    install=q.BASE/'staging/jp461-r1/install_audit.json'
    norm=lambda v: sorted(tuple(line.split()) for line in v.splitlines() if line.strip())
    packages=norm(command('dpkg-query','-W','-f=${Package}\t${Version}\n','nvidia-l4t-*'))
    if packages!=norm(json.loads(install.read_text())['l4t_packages_before']): raise ValueError('BSP changed')
    unit=command('systemctl','show','autoocr-jetson.service','-p','MainPID','-p','ActiveState','-p','UnitFileState')
    listeners=command('ss','-ltn','sport = :5051')
    mount=command('findmnt','-n','-o','TARGET,SOURCE,FSTYPE,OPTIONS','-T',str(q.BASE))
    if (set(unit.splitlines())!={'MainPID=0','ActiveState=inactive','UnitFileState=disabled'}
            or len(listeners.splitlines())!=1 or mount.split()!=previous['mount'].split()):
        raise ValueError('Service or mount changed')
    tokens=('layout_hotspot_mixed.py ranges','layout_hotspot_mixed.py build',
            'evaluate_layout_hotspot.py check','evaluate_layout_hotspot.py evaluate','benchmark_layout_hotspot.py --stage')
    processes=[int(line.split()[0]) for line in command('ps','-eo','pid,args').splitlines() if any(token in line for token in tokens)]
    if processes: raise ValueError('Experiment still running')
    names=REQUIRED+tuple(n for n in OPTIONAL if (stage/n).exists())
    members={n:stage/n for n in names}
    for prior,items in PRIOR.items():
        members.update({prior+'_'+n:q.BASE/('staging/layout_hotspot_e110_'+prior)/n for n in items})
    audit=dict(deployment_sha256=q.digest(path),engines=engines,plugins=plugins,systemd=unit,listeners=listeners,mount=mount,
        l4t_packages_unchanged=True,installation_audit_sha256=q.digest(install),
        l4t_package_rows_sha256=hashlib.sha256(json.dumps(packages,separators=(',',':')).encode()).hexdigest(),
        experiment_process_ids=processes,memory=q.memory(),storage=shutil.disk_usage(q.BASE)._asdict(),
        scripts_sha256={n:q.digest(p) for n,p in members.items() if n.endswith('.py')},
        note='Original release, BSP, mount unchanged; no activation/HTTP/clock/power/release changes. Idle snapshot, not service peak. Preserve r1 capture-budget and r2 old-binding API failures.')
    q.save(stage/'final_audit.json',audit)
    entries=[]
    for n,p in members.items():
        if Path(n).name!=n or p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(q.BASE/'staging'):
            raise ValueError('Unsafe archive member')
        entries.append(dict(name=n,bytes=p.stat().st_size,sha256=q.digest(p)))
    q.save(stage/'manifest.json',dict(files=entries,note='Explicit technical-only allowlist, no image/model/tensor/OCR text/license/runtime.'))
    archive=stage/'technical_evidence.tar.gz'
    with archive.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as tar:
            for n,p in members.items(): tar.add(p,arcname=n,recursive=False)
            tar.add(stage/'manifest.json',arcname='manifest.json',recursive=False)
    meta=dict(members=len(entries)+1,bytes=archive.stat().st_size,sha256=q.digest(archive))
    q.save(stage/'archive_meta.json',meta); print(json.dumps(meta))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,type=Path)
    run(p.parse_args().stage.resolve())
