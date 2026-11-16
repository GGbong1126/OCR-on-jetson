"""E116 immutable, technical-only evidence and independent summary."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import ner_norm_probe as p

n,q,old=p.n,p.q,p.old
SCRIPTS=('ner_norm_probe.py','ner_layernorm_plugin.cu','build_grid_plugin.py','archive_ner_norm.py','diagnose_ner_norm.py','test_diagnose_ner_norm.py',
    'test_ner_norm_probe.py','test_archive_ner_norm.py','ner_mixed_trial.py','layout_hotspot_mixed.py','optimize_rec_fp16.py','quantize_yolo_ptq.py')
FILES=SCRIPTS+('plan.json','prepare.log','tests.log','tests_final.log','libner_norm.build.json','plugin_build.log','engines.json','build.log',
    'check.json','check.log','layers_fp32.json','layers_half_boundary.json','diagnostic.json','diagnostic.log','summary.json','final_audit.json')


def ranges(rows):
    return dict(cases=len(rows),allclose=sum(r['allclose'] for r in rows),mae_min=min(r['mae'] for r in rows),
        mae_max=max(r['mae'] for r in rows),max_abs_error=max(r['max_abs_error'] for r in rows))


def summarize(plan,build,check,layers):
    if plan['status']!='prepared' or plan['rules']!=p.RULES or build['status']!='built_not_validated' or check['rules']!=p.RULES:raise ValueError('Plan/build/rules differ')
    if set(build['variants'])!={'fp32','half_boundary'} or set(layers)!=set(build['variants']):raise ValueError('Missing variant')
    if build['variants']['fp32']['flags']!=1024 or build['variants']['half_boundary']['flags']!=1025:raise ValueError('Build flags differ')
    for name in layers:
        if p.inspect(layers[name])!=check['inspection'][name]:raise ValueError('Inspector differs')
        if not build['variants'][name]['plugin_serialization_roundtrip']:raise ValueError('Serialization differs')
    if len(plan['norms'])!=25 or len(plan['prior_real_rows'])!=6:raise ValueError('Plan coverage differs')
    expected=[(r['image_sha256'],r['image_type'],j) for r in plan['prior_real_rows'] for j in range(25)]
    real=check['real_rows'];synthetic=check['synthetic_rows']
    if [(r['image_sha256'],r['image_type'],r['norm_index']) for r in real]!=expected:raise ValueError('Real coverage differs')
    wanted=[(l,k) for l in (2,128,512) for k in ('normal','constant','near_constant','alternating_large')]
    if [(r['length'],r['kind']) for r in synthetic]!=wanted:raise ValueError('Synthetic coverage differs')
    for r in real+synthetic:
        if set(r['variants'])!=set(layers):raise ValueError('Variant coverage differs')
        for v in r['variants'].values():
            if not v['repeat_exact'] or not v['reference']['all_finite']:raise ValueError('Invalid/repeat outputs')
    for r in real:
        if not all(v['original_output']['all_finite'] for v in r['variants'].values()):raise ValueError('Invalid original outputs')
    stats={}
    for name in layers:
        execution=check['execution'][name]
        if execution['calls']!=324 or execution['errors']:raise ValueError('Execution counts/errors differ')
        stats[name]=dict(same_real_input=ranges([r['variants'][name]['reference'] for r in real]),
            original_real_output=ranges([r['variants'][name]['original_output'] for r in real]),
            synthetic_same_input=ranges([r['variants'][name]['reference'] for r in synthetic]),
            failed_synthetic=[dict(length=r['length'],kind=r['kind']) for r in synthetic if not r['variants'][name]['reference']['allclose']])
    real_ok=all(v['same_real_input']['allclose']==150 for v in stats.values());control=stats['fp32']['original_real_output']['allclose']==150
    synth_ok=all(v['synthetic_same_input']['allclose']==12 for v in stats.values())
    status='isolated_passed_not_integrated' if real_ok and control and synth_ok else 'isolated_numeric_not_accepted'
    if (check['status'],check['real_same_input_passed'],check['original_fp32_control_passed'],check['synthetic_same_input_passed'])!=(status,real_ok,control,synth_ok):raise ValueError('Stored gate differs')
    return dict(status=status,original_norms=25,real_ner_inputs=6,real_norm_cases=150,synthetic_cases=12,plugin_calls=648,
        inspection=check['inspection'],variants=stats,input_abs_max=max(r['input_abs_max'] for r in real),
        full_ner_candidate=False,business_cases=0,benchmark_calls=0,accepted_for_release=False,
        note='FP32-only kernel plus visible float plugin interface; half input explicitly restored to float. CPU original-graph features, not TRT hidden activations. Same-rounded-input comparison separates boundary rounding from normalization arithmetic. Not Softmax/mask/whole-NER protection, business accuracy or speedup.')


def command(*args):return subprocess.check_output(args,text=True).strip()


def archive(stage):
    if stage!=p.STAGE:raise ValueError('Unexpected stage')
    for name in ('summary.json','final_audit.json','manifest.json','technical_evidence.tar.gz','archive_meta.json'):old.fresh(stage/name)
    meta=p.plugin_metadata(stage);plan,build,check=(n.read(stage/name) for name in ('plan.json','engines.json','check.json'))
    if any(r[k]!=v for r in (build,check) for k,v in meta.items()) or check['engine_report_sha256']!=q.digest(stage/'engines.json'):raise ValueError('Provenance drift')
    for name,row in build['variants'].items():
        if q.digest(stage/(name+'.engine'))!=row['engine_sha256']:raise ValueError('Engine drift')
    result=summarize(plan,build,check,{name:n.read(stage/('layers_'+name+'.json')) for name in build['variants']})
    diagnostic=n.read(stage/'diagnostic.json')
    if diagnostic['status']!='diagnosed_not_acceptance' or diagnostic['check_sha256']!=q.digest(stage/'check.json') or diagnostic['diagnostic_sha256']!=q.digest(stage/'diagnose_ner_norm.py') or any(diagnostic[k]!=v for k,v in meta.items()):raise ValueError('Diagnostic drift')
    if [r['length'] for r in diagnostic['rows']]!=[2,128,512] or any(v['calls']!=3 or v['errors'] for v in diagnostic['execution'].values()):raise ValueError('Diagnostic coverage differs')
    for spec in plan['norms']:p.validate_spec(spec)
    prior=n.read(p.PRIOR/'final_audit.json');cfg_path=q.RELEASE/'edge_deploy/jetson/deployment.json';cfg=n.read(cfg_path)
    engines={k:q.digest(q.RELEASE/v['path']) for k,v in cfg['engines'].items()}
    plugins={v['path']:q.digest(q.RELEASE/v['path']) for r in cfg['engines'].values() for v in r.get('plugins',[])}
    if engines!=prior['engines'] or plugins!=prior['plugins'] or q.digest(cfg_path)!=prior['deployment_sha256']:raise ValueError('Release changed')
    packages=sorted(tuple(line.split()) for line in command('dpkg-query','-W','-f=${Package}\t${Version}\n','nvidia-l4t-*').splitlines() if line.strip())
    pkgsha=hashlib.sha256(json.dumps(packages,separators=(',',':')).encode()).hexdigest()
    mount=command('findmnt','-n','-o','TARGET,SOURCE,FSTYPE,OPTIONS','-T',str(q.BASE))
    systemd=command('systemctl','show','autoocr-jetson.service','-p','MainPID','-p','ActiveState','-p','UnitFileState');listeners=command('ss','-ltn','sport = :5051')
    if pkgsha!=prior['l4t_package_rows_sha256'] or mount.split()!=prior['mount'].split() or set(systemd.splitlines())!={'MainPID=0','ActiveState=inactive','UnitFileState=disabled'} or len(listeners.splitlines())!=1:raise ValueError('BSP/mount/service changed')
    processes=[int(line.split()[0]) for line in command('ps','-eo','pid,args').splitlines() if any(s in line for s in ('/ner_norm_probe.py ','/build_grid_plugin.py '))]
    if processes:raise ValueError('Experiment still running')
    q.save(stage/'final_audit.json',dict(deployment_sha256=q.digest(cfg_path),engines=engines,plugins=plugins,mount=mount,systemd=systemd,listeners=listeners,
        l4t_package_rows_sha256=pkgsha,previous_audit_sha256=p.AUDIT_SHA,experiment_process_ids=processes,memory=q.memory(),storage=shutil.disk_usage(q.BASE)._asdict(),
        note='Original seven models/BSP/mount unchanged; HTTP inactive. No activation/count, power or clocks changes. Idle snapshot not peak.'))
    result['reports_sha256']={name:q.digest(stage/name) for name in ('plan.json','engines.json','check.json','libner_norm.build.json','diagnostic.json')};q.save(stage/'summary.json',result)
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
