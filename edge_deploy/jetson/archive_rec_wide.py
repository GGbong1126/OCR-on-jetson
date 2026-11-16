"""E123 validation, technical evidence allowlist and unchanged-release audit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import benchmark_rec_wide as b

w,q,n,old=b.w,b.q,b.n,b.old
SCRIPTS=w.previous.SCRIPTS+('rec_wide_fp32.py','benchmark_rec_wide.py','archive_rec_wide.py',
                          'test_rec_wide_fp32.py','test_benchmark_rec_wide.py','test_archive_rec_wide.py')
REPORTS=('plan.json','build.json','check.json','layers.json','quality.json','run.json')+tuple('pipeline_%s_%s.json'%(v,r) for r,v in b.p.ORDER)
LOGS=('tests.log','tests_final.log','prepare.log','build.log','check.log','quality.log','measure.log')


def validate_reports(stage):
    meta=w.metadata(stage,True);plan=n.read(stage/'plan.json');build=n.read(stage/'build.json')
    if build['profile']!=w.PROFILE or build['flags']!=0:raise ValueError('Build precision/profile differs')
    checked=n.read(stage/'check.json')
    if any(checked[k]!=v for k,v in meta.items()) or checked['rules']!=w.RULES:raise ValueError('Check source drift')
    layers=n.read(stage/'layers.json')
    inspected=w.inspection(layers,checked['inspection']['layers'])
    if inspected!=checked['inspection'] or q.digest(stage/'layers.json')!=checked['layers_sha256']:raise ValueError('Inspection drift')
    if len(checked['rows'])!=4 or [r['width'] for r in checked['boundaries']]!=w.BOUNDARIES:raise ValueError('Input coverage differs')
    for row,expected in zip(checked['rows'],plan['crops']):
        if row['case']!=expected['case'] or row['shape']!=expected['input_shapes']['x'] or row['input_sha256']!=expected['input_sha256']['x'] or row['reference_output_sha256']!=expected['output_sha256']:
            raise ValueError('Original tensor evidence drift')
    if checked['execution']['errors'] or checked['execution']['calls']!=18:raise ValueError('Check execution errors')
    check_good=inspected['passed'] and all(r['token_sequence_equal'] and r['repeated_exact'] for r in checked['rows']+checked['boundaries'])
    if check_good != (checked['status']=='wide_ctc_consistent_not_gt'):raise ValueError('Check verdict differs')
    if not check_good:
        if (stage/'run.json').exists():raise ValueError('Benchmark despite failed check')
        return dict(status='wide_tensor_quality_not_accepted',accepted_for_release=False)
    quality=n.read(stage/'quality.json')
    if quality['status']!='all_26_business_consistent_not_gt':
        if (stage/'run.json').exists():raise ValueError('Benchmark despite failed business')
        return dict(status='wide_business_quality_not_accepted',passed=sum(r['passed'] for r in quality['rows']),accepted_for_release=False)
    meta=b.validate_quality(stage)
    if not (stage/'run.json').exists():
        return dict(status='wide_quality_consistent_not_benchmarked',passed=26,accepted_for_release=False)
    run=n.read(stage/'run.json')
    if any(run[k]!=v for k,v in meta.items()) or run['benchmark_script_sha256']!=q.digest(b.__file__):raise ValueError('Benchmark provenance drift')
    hashes={f'pipeline_{v}_{r}.json':q.digest(stage/f'pipeline_{v}_{r}.json') for r,v in b.p.ORDER}
    if hashes!=run['reports_sha256']:raise ValueError('Worker report drift')
    reports={(r,v):n.read(stage/f'pipeline_{v}_{r}.json') for r,v in b.p.ORDER}
    result=b.summarize(plan,run,reports)
    result.update(reports_sha256={name:q.digest(stage/name) for name in REPORTS},
                  engine_bytes=build['engine_bytes'],build_s=build['build_s'],business_passed=26)
    return result


def audit(stage):
    prior=n.read(w.PRIOR/'final_audit.json');cfg_path=q.RELEASE/'edge_deploy/jetson/deployment.json';cfg=n.read(cfg_path)
    engines={k:q.digest(q.RELEASE/v['path']) for k,v in cfg['engines'].items()}
    plugins={v['path']:q.digest(q.RELEASE/v['path']) for r in cfg['engines'].values() for v in r.get('plugins',[])}
    if engines!=prior['engines'] or plugins!=prior['plugins'] or q.digest(cfg_path)!=prior['deployment_sha256']:raise ValueError('Original release changed')
    command=b.p.prior.previous.command
    packages=sorted(tuple(line.split()) for line in command('dpkg-query','-W','-f=${Package}\t${Version}\n','nvidia-l4t-*').splitlines() if line.strip())
    pkgsha=hashlib.sha256(json.dumps(packages,separators=(',',':')).encode()).hexdigest()
    mount=command('findmnt','-n','-o','TARGET,SOURCE,FSTYPE,OPTIONS','-T',str(q.BASE))
    systemd=command('systemctl','show','autoocr-jetson.service','-p','MainPID','-p','ActiveState','-p','UnitFileState');listeners=command('ss','-ltn','sport = :5051')
    if pkgsha!=prior['l4t_package_rows_sha256'] or mount.split()!=prior['mount'].split() or set(systemd.splitlines())!={'MainPID=0','ActiveState=inactive','UnitFileState=disabled'} or len(listeners.splitlines())!=1:raise ValueError('BSP/mount/service changed')
    expected=set()
    for name in ('build.json','check.json','quality.json','run.json'):
        if not (stage/name).exists():continue
        row=n.read(stage/name)
        if 'pid' in row:expected.add(row['pid'])
        if 'parent_pid' in row:expected.add(row['parent_pid'])
        expected.update(v['pid'] for v in row.get('workers',[]))
    processes=[int(line.split()[0]) for line in command('ps','-eo','pid,args').splitlines()[1:]
               if int(line.split()[0]) in expected or any('/'+s+' ' in line for s in ('rec_wide_fp32.py','benchmark_rec_wide.py'))]
    if processes:raise ValueError('Experiment still running')
    return dict(deployment_sha256=q.digest(cfg_path),engines=engines,plugins=plugins,mount=mount,systemd=systemd,listeners=listeners,
        l4t_package_rows_sha256=pkgsha,previous_audit_sha256=w.AUDIT_SHA,experiment_process_ids=processes,memory=q.memory(),
        storage=shutil.disk_usage(q.BASE)._asdict(),note='Original seven engines/plugin/config/BSP/mount unchanged. No HTTP/activation/power/clock changes. Candidate separate; idle snapshot not peak.')


def archive(stage):
    if stage!=w.STAGE:raise ValueError('Unexpected stage')
    for name in ('summary.json','final_audit.json','manifest.json','technical_evidence.tar.gz','archive_meta.json'):old.fresh(stage/name)
    result=validate_reports(stage);q.save(stage/'final_audit.json',audit(stage));q.save(stage/'summary.json',result)
    paths={name:stage/name for name in SCRIPTS}
    paths.update({name:stage/name for name in REPORTS+LOGS+('summary.json','final_audit.json') if (stage/name).is_file()})
    paths.update({'previous_final_audit.json':w.PRIOR/'final_audit.json','previous_diagnostic.json':w.PRIOR/'diagnostic.json'})
    rows=b.p.prior.safe_files(paths,q.BASE/'staging')
    q.save(stage/'manifest.json',dict(files=rows,note='Technical allowlist, no engines/cache/weights/images/tensors/text/tokens/license.'))
    with (stage/'technical_evidence.tar.gz').open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as t:
            for name,path in paths.items():t.add(path,arcname=name,recursive=False)
            t.add(stage/'manifest.json',arcname='manifest.json',recursive=False)
    meta=dict(bytes=(stage/'technical_evidence.tar.gz').stat().st_size,sha256=q.digest(stage/'technical_evidence.tar.gz'),
              members=len(paths)+1,audit_sha256=q.digest(stage/'final_audit.json'))
    q.save(stage/'archive_meta.json',meta);print(json.dumps(meta));print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',type=Path,required=True)
    archive(parser.parse_args().stage.resolve())
