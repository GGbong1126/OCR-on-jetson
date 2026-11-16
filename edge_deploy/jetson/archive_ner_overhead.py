"""E120 technical-only benchmark evidence and immutable-release audit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import benchmark_ner_protected as b

n,q,old,previous=b.n,b.q,b.old,b.previous
SCRIPTS=('ner_protected_fp32.py','evaluate_ner_mixed.py','ner_norm_probe.py','ner_softmax_probe.py','ner_mixed_trial.py',
    'layout_hotspot_mixed.py','optimize_rec_fp16.py','quantize_yolo_ptq.py','archive_ner_protected.py',
    'benchmark_ner_protected.py','archive_ner_overhead.py','test_benchmark_ner_protected.py','test_archive_ner_overhead.py')
REPORTS=('plan.json','run.json')+tuple('benchmark_%s_%s.json'%(v,r) for r,v in b.ORDER)
FILES=SCRIPTS+REPORTS+('tests.log','tests_final.log','prepare.log','measure.log','summary.json','final_audit.json')


def validate_reports(stage):
    meta=b.metadata(stage);plan=n.read(stage/'plan.json');run=n.read(stage/'run.json')
    if any(run[k]!=v for k,v in meta.items()):raise ValueError('Run provenance changed')
    hashes={name:q.digest(stage/name) for name in REPORTS[2:]}
    if run['benchmark_reports_sha256']!=hashes:raise ValueError('Worker report changed')
    reports={(r,v):n.read(stage/('benchmark_%s_%s.json'%(v,r))) for r,v in b.ORDER}
    result=b.summarize(plan,run,reports)
    result['reports_sha256']={name:q.digest(stage/name) for name in REPORTS}
    return result


def safe_files(paths,root):
    rows=[]
    for name,path in paths.items():
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()) or Path(name).name!=name:
            raise ValueError('Unsafe evidence path')
        rows.append(dict(name=name,bytes=path.stat().st_size,sha256=q.digest(path)))
    return rows


def archive(stage):
    if stage!=b.STAGE:raise ValueError('Unexpected stage')
    for name in ('summary.json','final_audit.json','manifest.json','technical_evidence.tar.gz','archive_meta.json'):old.fresh(stage/name)
    result=validate_reports(stage);prior=n.read(b.PRIOR/'final_audit.json')
    cfg_path=q.RELEASE/'edge_deploy/jetson/deployment.json';cfg=n.read(cfg_path)
    engines={k:q.digest(q.RELEASE/v['path']) for k,v in cfg['engines'].items()}
    plugins={v['path']:q.digest(q.RELEASE/v['path']) for r in cfg['engines'].values() for v in r.get('plugins',[])}
    if engines!=prior['engines'] or plugins!=prior['plugins'] or q.digest(cfg_path)!=prior['deployment_sha256']:raise ValueError('Original release changed')
    command=previous.command
    packages=sorted(tuple(line.split()) for line in command('dpkg-query','-W','-f=${Package}\t${Version}\n','nvidia-l4t-*').splitlines() if line.strip())
    pkgsha=hashlib.sha256(json.dumps(packages,separators=(',',':')).encode()).hexdigest()
    mount=command('findmnt','-n','-o','TARGET,SOURCE,FSTYPE,OPTIONS','-T',str(q.BASE))
    systemd=command('systemctl','show','autoocr-jetson.service','-p','MainPID','-p','ActiveState','-p','UnitFileState');listeners=command('ss','-ltn','sport = :5051')
    if pkgsha!=prior['l4t_package_rows_sha256'] or mount.split()!=prior['mount'].split() or set(systemd.splitlines())!={'MainPID=0','ActiveState=inactive','UnitFileState=disabled'} or len(listeners.splitlines())!=1:raise ValueError('BSP/mount/service changed')
    run=n.read(stage/'run.json');expected_pids={run['parent_pid']}|{w['pid'] for w in run['workers']}
    processes=[int(line.split()[0]) for line in command('ps','-eo','pid,args').splitlines()[1:]
        if int(line.split()[0]) in expected_pids or '/benchmark_ner_protected.py ' in line]
    if processes:raise ValueError('Experiment still running')
    q.save(stage/'final_audit.json',dict(deployment_sha256=q.digest(cfg_path),engines=engines,plugins=plugins,mount=mount,systemd=systemd,listeners=listeners,
        l4t_package_rows_sha256=pkgsha,previous_audit_sha256=b.AUDIT_SHA,experiment_process_ids=processes,memory=q.memory(),storage=shutil.disk_usage(q.BASE)._asdict(),
        note='Original seven engines/plugin/BSP/mount unchanged. HTTP inactive; no activation/count, power/clock changes. Idle snapshot not peak. Original E119 FP32 candidate and plugins retained separately.'))
    q.save(stage/'summary.json',result)
    paths={name:stage/name for name in FILES};paths['previous_final_audit.json']=b.PRIOR/'final_audit.json'
    first=q.BASE/'staging/ner_overhead_e120_r1'
    for name in ('benchmark_ner_protected.py','test_benchmark_ner_protected.py','tests.log','plan.json','prepare.log','run.json','measure.log',
                 'benchmark_reference_r1.json','diagnose_ner_spawn.py','spawn_diagnostic.log','spawn_diagnostic.json','spawn_before.json','spawn_after.json'):
        paths['first_attempt_'+name]=first/name
    rows=safe_files(paths,q.BASE/'staging')
    q.save(stage/'manifest.json',dict(files=rows,note='Technical allowlist only; no engines/plugins/weights/images/raw token arrays/logits/tensors/text/license.'))
    with (stage/'technical_evidence.tar.gz').open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as t:
            for name,path in paths.items():t.add(path,arcname=name,recursive=False)
            t.add(stage/'manifest.json',arcname='manifest.json',recursive=False)
    meta=dict(bytes=(stage/'technical_evidence.tar.gz').stat().st_size,sha256=q.digest(stage/'technical_evidence.tar.gz'),members=len(paths)+1,
        audit_sha256=q.digest(stage/'final_audit.json'))
    q.save(stage/'archive_meta.json',meta);print(json.dumps(meta));print(json.dumps(result))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',type=Path,required=True);archive(ap.parse_args().stage.resolve())
