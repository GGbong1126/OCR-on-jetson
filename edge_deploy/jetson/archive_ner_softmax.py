"""E118 technical-only archive, coverage/precision/quality gates and release audit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import ner_softmax_probe as p

n,q,old=p.n,p.q,p.old
SCRIPTS=('ner_softmax_probe.py','ner_mask_softmax.cu','archive_ner_softmax.py','test_ner_softmax_probe.py','test_archive_ner_softmax.py',
    'ner_mixed_trial.py','layout_hotspot_mixed.py','optimize_rec_fp16.py','quantize_yolo_ptq.py','build_grid_plugin.py')
FILES=SCRIPTS+('plan.json','prepare.log','tests.log','libner_softmax.build.json','plugin_build.log','build.json','build.log',
    'check.json','check.log','layers.json','summary.json','final_audit.json')


def summarize(plan,build,check,layers):
    if plan['status']!='prepared' or plan['rules']!=p.RULES or check['rules']!=p.RULES or build['status']!='built_not_validated' or build['flags']!=1024 or not build['serialization_roundtrip']:raise ValueError('Plan/build drift')
    if p.inspect(layers)!=check['inspection']:raise ValueError('Precision inspection differs')
    real,syn=check['real_rows'],check['synthetic_rows']
    expected=[(r['image_sha256'],r['image_type'],j) for r in plan['prior_real_rows'] for j in range(12)]
    if len(expected)!=72 or [(r['image_sha256'],r['image_type'],r['layer']) for r in real]!=expected:raise ValueError('Real coverage differs')
    keys=('length','kind','input_shape','input_sha256','mask_sha256','score_abs_max','active_keys')
    if [{k:r[k] for k in keys} for r in syn]!=plan['synthetic'] or len(syn)!=32:raise ValueError('Synthetic coverage differs')
    rows=real+syn
    if any(not r['repeat_exact'] or any(not r[k]['all_finite'] for k in ('cpu','fp64_softmax')) for r in rows):raise ValueError('Nonfinite/repeat failure')
    if check['execution']['calls']!=208 or check['execution']['errors']!=0:raise ValueError('Execution counter differs')
    status=p.result_status(real,syn)
    if status!=check['status']:raise ValueError('Decision differs')
    def stats(rs):
        return dict(cases=len(rs),cpu_passed=sum(r['cpu']['allclose'] for r in rs),fp64_softmax_passed=sum(r['fp64_softmax']['allclose'] for r in rs),
            cpu_max_abs=max(r['cpu']['max_abs_error'] for r in rs),fp64_softmax_max_abs=max(r['fp64_softmax']['max_abs_error'] for r in rs),
            cpu_mae_max=max(r['cpu']['mae'] for r in rs),row_sum_error_max=max(r['row_sum_error'] for r in rs),
            probability_contract_passed=sum(r['probability_contract_passed'] for r in rs),score_abs_max=max(r['score_abs_max'] for r in rs))
    return dict(status=status,real=stats(real),synthetic=stats(syn),plugin_calls=208,inspection=check['inspection'],
        all_masked_cases=sum(r['active_keys']==0 for r in rows),all_masked_uniform_cases=sum(r['all_masked_uniform'] is True for r in rows),
        full_ner_candidate=False,business_cases=0,benchmark_calls=0,accepted_for_release=False,
        note='Only isolated FP32 attention boundary. Original CPU intermediate scores, not original TRT hidden features. FP64 reference preserves FP32 logit rounding. No whole-NER, GT, speedup or release claim.')


def command(*args):return subprocess.check_output(args,text=True).strip()


def archive(stage):
    if stage!=p.STAGE:raise ValueError('Unexpected stage')
    for name in ('summary.json','final_audit.json','manifest.json','technical_evidence.tar.gz','archive_meta.json'):old.fresh(stage/name)
    meta=p.metadata(stage,True);plan,build,check=(n.read(stage/name) for name in ('plan.json','build.json','check.json'))
    if any(r[k]!=v for r in (build,check) for k,v in meta.items()) or check['build_sha256']!=q.digest(stage/'build.json'):raise ValueError('Provenance drift')
    if q.digest(stage/'softmax.engine')!=build['engine_sha256']:raise ValueError('Candidate changed')
    result=summarize(plan,build,check,n.read(stage/'layers.json'))
    prior=n.read(p.PRIOR/'final_audit.json');cfg_path=q.RELEASE/'edge_deploy/jetson/deployment.json';cfg=n.read(cfg_path)
    engines={k:q.digest(q.RELEASE/v['path']) for k,v in cfg['engines'].items()}
    plugins={v['path']:q.digest(q.RELEASE/v['path']) for r in cfg['engines'].values() for v in r.get('plugins',[])}
    if engines!=prior['engines'] or plugins!=prior['plugins'] or q.digest(cfg_path)!=prior['deployment_sha256']:raise ValueError('Original release changed')
    packages=sorted(tuple(line.split()) for line in command('dpkg-query','-W','-f=${Package}\t${Version}\n','nvidia-l4t-*').splitlines() if line.strip())
    pkgsha=hashlib.sha256(json.dumps(packages,separators=(',',':')).encode()).hexdigest()
    mount=command('findmnt','-n','-o','TARGET,SOURCE,FSTYPE,OPTIONS','-T',str(q.BASE))
    systemd=command('systemctl','show','autoocr-jetson.service','-p','MainPID','-p','ActiveState','-p','UnitFileState');listeners=command('ss','-ltn','sport = :5051')
    if pkgsha!=prior['l4t_package_rows_sha256'] or mount.split()!=prior['mount'].split() or set(systemd.splitlines())!={'MainPID=0','ActiveState=inactive','UnitFileState=disabled'} or len(listeners.splitlines())!=1:raise ValueError('BSP/mount/service changed')
    processes=[int(line.split()[0]) for line in command('ps','-eo','pid,args').splitlines() if any(s in line for s in ('/ner_softmax_probe.py ','/build_grid_plugin.py '))]
    if processes:raise ValueError('Experiment still running')
    q.save(stage/'final_audit.json',dict(deployment_sha256=q.digest(cfg_path),engines=engines,plugins=plugins,mount=mount,systemd=systemd,listeners=listeners,
        l4t_package_rows_sha256=pkgsha,previous_audit_sha256=p.AUDIT_SHA,experiment_process_ids=processes,memory=q.memory(),storage=shutil.disk_usage(q.BASE)._asdict(),
        note='Original seven models/BSP/mount unchanged; HTTP inactive. No activation/count, power or clocks changes. Idle snapshot not peak.'))
    result['reports_sha256']={name:q.digest(stage/name) for name in ('plan.json','build.json','check.json','libner_softmax.build.json')};q.save(stage/'summary.json',result)
    paths={name:stage/name for name in FILES};paths['previous_final_audit.json']=p.PRIOR/'final_audit.json';rows=[]
    for name,path in paths.items():
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(q.BASE/'staging') or Path(name).name!=name:raise ValueError('Unsafe evidence')
        rows.append(dict(name=name,bytes=path.stat().st_size,sha256=q.digest(path)))
    q.save(stage/'manifest.json',dict(files=rows,note='Technical allowlist only; no model/plugin binaries, images, tokens, tensors, document text or license.'))
    with (stage/'technical_evidence.tar.gz').open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as t:
            for name,path in paths.items():t.add(path,arcname=name,recursive=False)
            t.add(stage/'manifest.json',arcname='manifest.json',recursive=False)
    out=dict(bytes=(stage/'technical_evidence.tar.gz').stat().st_size,sha256=q.digest(stage/'technical_evidence.tar.gz'),members=len(paths)+1)
    q.save(stage/'archive_meta.json',out);print(json.dumps(out));print(json.dumps(result))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',type=Path,required=True)
    archive(ap.parse_args().stage.resolve())
