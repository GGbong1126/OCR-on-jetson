"""E117 independent FP64 stability summary; historical CPU gates retained."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import ner_norm_stable as p

n,q,old=p.n,p.q,p.old
SCRIPTS=('ner_norm_stable.py','ner_layernorm_stable.cu','archive_ner_norm_stable.py','test_ner_norm_stable.py','test_archive_ner_norm_stable.py',
    'ner_norm_probe.py','diagnose_ner_norm.py','ner_mixed_trial.py','layout_hotspot_mixed.py','optimize_rec_fp16.py','quantize_yolo_ptq.py','build_grid_plugin.py')
FILES=SCRIPTS+('plan.json','prepare.log','tests.log','tests_final.log','libner_stable.build.json','plugin_build.log','build.json','build.log',
    'check.json','check.log','layers.json','summary.json','final_audit.json')


def ranges(rows):
    return dict(cases=len(rows),allclose=sum(r['allclose'] for r in rows),mae_min=min(r['mae'] for r in rows),
        mae_max=max(r['mae'] for r in rows),max_abs_error=max(r['max_abs_error'] for r in rows))


def summarize(plan,build,check,layers):
    if plan['status']!='prepared' or plan['rules']!=p.RULES or check['rules']!=p.RULES or build['status']!='built_not_validated' or build['flags']!=1024 or not build['serialization_roundtrip']:raise ValueError('Plan/build/rules differ')
    if p.p.inspect(layers)!=check['inspection']:raise ValueError('Inspector drift')
    real=check['real_rows'];syn=check['synthetic_rows']
    if len(real)!=150 or len({(r['image_sha256'],r['image_type'],r['norm_index']) for r in real})!=150:raise ValueError('Real coverage differs')
    if [dict(length=r['length'],kind=r['kind'],input_sha256=r['input_sha256']) for r in syn]!=plan['synthetic'] or len(syn)!=30:raise ValueError('Synthetic coverage differs')
    for r in real+syn:
        if not r['repeat_exact'] or not r['emulator_vs_candidate']['allclose']:raise ValueError('Candidate repeat/emulation differs')
        if any(not r[k]['all_finite'] for k in ('original_vs_fp64','old_vs_fp64','candidate_vs_fp64','candidate_vs_original','emulator_vs_candidate')):raise ValueError('Nonfinite outputs')
    for key,calls in (('old',180),('candidate',360)):
        if check['execution'][key]['calls']!=calls or check['execution'][key]['errors']:raise ValueError('Execution counters differ')
    status=p.result_status(real,syn)
    if status!=check['status']:raise ValueError('Stored decision differs')
    low=[r for r in syn if r['kind']=='near_constant']
    return dict(status=status,real_cases=150,synthetic_cases=30,plugin_calls=540,
        real={key:ranges([r[key] for r in real]) for key in ('candidate_vs_fp64','candidate_vs_original','old_vs_fp64')},
        synthetic={key:ranges([r[key] for r in syn]) for key in ('candidate_vs_fp64','candidate_vs_original','old_vs_fp64','original_vs_fp64')},
        legacy_cpu_synthetic_failed=[dict(length=r['length'],kind=r['kind']) for r in syn if not r['candidate_vs_original']['allclose']],
        low_variance_mae_ratios=[r['candidate_vs_fp64']['mae']/r['old_vs_fp64']['mae'] for r in low],
        emulator_exact_cases=sum(r['emulator_exact'] for r in real+syn),inspection=check['inspection'],
        full_ner_candidate=False,business_cases=0,benchmark_calls=0,accepted_for_release=False,
        note='New independent FP64 formula criterion plus real CPU regression. Legacy synthetic CPU-equality results remain separate; this does not reverse E116 failure. CPU original graph features, not TRT hidden activations. No complete NER/Softmax/mask/accuracy/performance/release claim.')


def command(*args):return subprocess.check_output(args,text=True).strip()


def archive(stage):
    if stage!=p.STAGE:raise ValueError('Unexpected stage')
    for name in ('summary.json','final_audit.json','manifest.json','technical_evidence.tar.gz','archive_meta.json'):old.fresh(stage/name)
    meta=p.plugin_metadata(stage);plan,build,check=(n.read(stage/name) for name in ('plan.json','build.json','check.json'))
    if any(r[k]!=v for r in (build,check) for k,v in meta.items()) or check['build_sha256']!=q.digest(stage/'build.json'):raise ValueError('Provenance drift')
    if q.digest(stage/'stable.engine')!=build['engine_sha256']:raise ValueError('Candidate engine changed')
    prior_plan=n.read(p.PRIOR/'plan.json');prior_check=n.read(p.PRIOR/'check.json')
    expected=[(r['image_sha256'],r['image_type'],j) for r in prior_plan['prior_real_rows'] for j in range(25)]
    if [(r['image_sha256'],r['image_type'],r['norm_index']) for r in check['real_rows']]!=expected:raise ValueError('Original coverage differs')
    for r,previous in zip(check['real_rows'],prior_check['real_rows']):
        if r['input_sha256']!=previous['input_sha256'] or r['old_output_sha256']!=previous['variants']['fp32']['output_sha256']:raise ValueError('Original control drift')
    for r,previous in zip(check['synthetic_rows'][:12],prior_check['synthetic_rows']):
        v=previous['variants']['fp32']
        if r['input_sha256']!=v['input_sha256'] or r['old_output_sha256']!=v['output_sha256']:raise ValueError('Old synthetic control drift')
    result=summarize(plan,build,check,n.read(stage/'layers.json'))
    prior=n.read(p.PRIOR/'final_audit.json');cfg_path=q.RELEASE/'edge_deploy/jetson/deployment.json';cfg=n.read(cfg_path)
    engines={k:q.digest(q.RELEASE/v['path']) for k,v in cfg['engines'].items()}
    plugins={v['path']:q.digest(q.RELEASE/v['path']) for r in cfg['engines'].values() for v in r.get('plugins',[])}
    if engines!=prior['engines'] or plugins!=prior['plugins'] or q.digest(cfg_path)!=prior['deployment_sha256']:raise ValueError('Release changed')
    packages=sorted(tuple(line.split()) for line in command('dpkg-query','-W','-f=${Package}\t${Version}\n','nvidia-l4t-*').splitlines() if line.strip())
    pkgsha=hashlib.sha256(json.dumps(packages,separators=(',',':')).encode()).hexdigest()
    mount=command('findmnt','-n','-o','TARGET,SOURCE,FSTYPE,OPTIONS','-T',str(q.BASE))
    systemd=command('systemctl','show','autoocr-jetson.service','-p','MainPID','-p','ActiveState','-p','UnitFileState');listeners=command('ss','-ltn','sport = :5051')
    if pkgsha!=prior['l4t_package_rows_sha256'] or mount.split()!=prior['mount'].split() or set(systemd.splitlines())!={'MainPID=0','ActiveState=inactive','UnitFileState=disabled'} or len(listeners.splitlines())!=1:raise ValueError('BSP/mount/service changed')
    processes=[int(line.split()[0]) for line in command('ps','-eo','pid,args').splitlines() if any(s in line for s in ('/ner_norm_stable.py ','/build_grid_plugin.py '))]
    if processes:raise ValueError('Experiment still running')
    q.save(stage/'final_audit.json',dict(deployment_sha256=q.digest(cfg_path),engines=engines,plugins=plugins,mount=mount,systemd=systemd,listeners=listeners,
        l4t_package_rows_sha256=pkgsha,previous_audit_sha256=p.AUDIT_SHA,experiment_process_ids=processes,memory=q.memory(),storage=shutil.disk_usage(q.BASE)._asdict(),
        note='Original seven models/BSP/mount unchanged; HTTP inactive. No activation/count, power or clocks changes. Idle snapshot not peak.'))
    result['reports_sha256']={name:q.digest(stage/name) for name in ('plan.json','build.json','check.json','libner_stable.build.json')};q.save(stage/'summary.json',result)
    paths={name:stage/name for name in FILES};paths['previous_final_audit.json']=p.PRIOR/'final_audit.json';rows=[]
    for name,path in paths.items():
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(q.BASE/'staging') or Path(name).name!=name:raise ValueError('Unsafe evidence')
        rows.append(dict(name=name,bytes=path.stat().st_size,sha256=q.digest(path)))
    q.save(stage/'manifest.json',dict(files=rows,note='Technical allowlist only, no model/plugin binaries, images, token arrays, tensors, document text or license.'))
    with (stage/'technical_evidence.tar.gz').open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as t:
            for name,path in paths.items():t.add(path,arcname=name,recursive=False)
            t.add(stage/'manifest.json',arcname='manifest.json',recursive=False)
    out=dict(bytes=(stage/'technical_evidence.tar.gz').stat().st_size,sha256=q.digest(stage/'technical_evidence.tar.gz'),members=len(paths)+1)
    q.save(stage/'archive_meta.json',out);print(json.dumps(out));print(json.dumps(result))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',type=Path,required=True)
    archive(ap.parse_args().stage.resolve())
