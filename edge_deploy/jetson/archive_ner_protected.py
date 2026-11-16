"""E119 full-FP32 structural control evidence; no candidate promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import ner_protected_fp32 as p

n,q,old,e=p.n,p.q,p.old,p.e
SCRIPTS=('ner_protected_fp32.py','evaluate_ner_mixed.py','ner_norm_probe.py','ner_softmax_probe.py','ner_mixed_trial.py',
    'layout_hotspot_mixed.py','optimize_rec_fp16.py','quantize_yolo_ptq.py','archive_ner_protected.py',
    'test_ner_protected_fp32.py','test_evaluate_ner_mixed.py','test_archive_ner_protected.py')
FILES=SCRIPTS+('tests.log','tests_final.log','plan.json','prepare.log','build.json','build.log','check.json','check.log',
    'layers.json','quality.log','summary.json','final_audit.json')


def execution(r,calls):
    if r['errors'] or r['calls']!=calls:raise ValueError('Execution counts/errors differ')


def summarize(plan,build,check,quality,layers):
    if plan['status']!='prepared' or plan['rules']!=p.RULES or build['status']!='built_not_validated' or build['flags']!=1024:raise ValueError('Plan/build differs')
    if check['status']!='finite_checked_not_business_accepted' or check['rules']!=p.RULES:raise ValueError('Incomplete numeric check')
    if [r['name'] for r in build['boundaries']]!=p.plugin_names():raise ValueError('Build coverage differs')
    ins=p.inspect(layers)
    if ins!=check['inspection']:raise ValueError('Inspection differs')
    real,synthetic=check['rows'],check['boundaries']
    if len(real)!=6 or [n.describe(r) for r in real]!=plan['rows']:raise ValueError('Original real coverage differs')
    if [dict(length=r['length'],kind=r['kind'],segment=r['segment'],inputs=r['inputs']) for r in synthetic]!=plan['boundaries'] or len(synthetic)!=40:raise ValueError('Boundary coverage differs')
    if any(not r['repeat_exact'] or not r['metrics']['all_finite'] for r in real+synthetic):raise ValueError('Unstable or invalid outputs')
    execution(check['candidate_execution'],92);execution(check['boundary_reference_execution'],40)
    prepassed=ins['precision_boundary_verified'] and all(r['metrics']['active_token_labels_equal'] for r in real+synthetic)
    result=dict(status='structure_or_labels_not_accepted',inspection=ins,build_s=build['build_s'],engine_bytes=build['engine_bytes'],
        real_cases=6,real_labels_equal=sum(r['metrics']['active_token_labels_equal'] for r in real),real_allclose=sum(r['metrics']['allclose_diagnostic'] for r in real),
        real_max_abs_error=max(r['metrics']['max_abs_error'] for r in real),boundary_cases=40,
        boundary_active_labels_equal=sum(r['metrics']['active_token_labels_equal'] for r in synthetic),boundary_all_labels_equal=sum(r['all_token_labels_equal'] for r in synthetic),
        boundary_allclose=sum(r['metrics']['allclose_diagnostic'] for r in synthetic),boundary_max_abs_error=max(r['metrics']['max_abs_error'] for r in synthetic),
        candidate_probe_calls=92,reference_boundary_calls=40,business_cases=0,business_executions=0,benchmark_calls=0,accepted_for_release=False,
        note='Complete FP32 structural control, not quantization or a benchmark. 37 protected plugins explicitly visible; other fusion internals not enumerated. Existing development images, no GT. Original release retained.')
    if quality is None:
        if prepassed:raise ValueError('Business report missing despite passed prerequisite')
        return result
    if not prepassed or quality['status'] not in ('quality_consistent_not_benchmarked','not_accepted'):raise ValueError('Unexpected business run/status')
    rows=quality['rows']
    if [dict(image_sha256=r['image_sha256'],image_type=r['image_type']) for r in rows]!=plan['business_pairs'] or len(rows)!=26:raise ValueError('Business coverage differs')
    counts={b:0 for b in ('reference','candidate')}
    for i,r in enumerate(rows):
        if r['order']!=list(('reference','candidate') if i%2==0 else ('candidate','reference')):raise ValueError('Business order differs')
        if r['passed']!=e.quality_passed(r) or r['same_ner_inputs']!=(r['inputs']['reference']==r['inputs']['candidate']):raise ValueError('Stored business decision differs')
        for b in counts:
            size=len(r['inputs'][b]);counts[b]+=size
            if not size or len(r['ner_output_sha256'][b])!=size or r['entities']['postprocess_calls_'+b]!=1:raise ValueError('Missing NER/entity trace')
    execution(quality['candidate_execution'],counts['candidate']);execution(quality['original_execution']['ner'],counts['reference']+1)
    for r in quality['original_execution'].values():
        if r['errors'] or r.get('cpu_fallback',{}).get('errors',0):raise ValueError('Original execution error')
    passed=sum(r['passed'] for r in rows);status='quality_consistent_not_benchmarked' if passed==26 else 'not_accepted'
    if quality['status']!=status or quality['business_passed_count']!=passed or not quality['precision_boundary_verified']:raise ValueError('Stored business status differs')
    result.update(status=status,business_cases=26,business_executions=52,business_passed_count=passed,ner_calls=counts,
        failed_case_indices=[i for i,r in enumerate(rows) if not r['passed']],same_input_cases=sum(r['same_ner_inputs'] for r in rows),
        token_labels_equal_cases=sum(r['token_labels_equal'] for r in rows),
        strict_equal_counts={k:sum(r['strict_equal'][k] for r in rows) for k in rows[0]['strict_equal']},
        entity_equal_counts={k:sum(r['entities'][k] for r in rows) for k in ('raw_entity_spans_equal','cleaned_entity_spans_equal','threshold_membership_equal')},
        max_aligned_entity_score_delta=max(r['entities']['max_aligned_entity_score_delta'] for r in rows),
        field_equal_counts={k:sum(r['fields']['api_special_text_equal'][k] for r in rows) for k in ('title','category_no','page_no')},
        extra_cpu_fallback_cases=sum(r['cpu_fallback_calls']['reference']!=r['cpu_fallback_calls']['candidate'] for r in rows))
    return result


def command(*args):return subprocess.check_output(args,text=True).strip()


def archive(stage):
    if stage!=p.STAGE:raise ValueError('Unexpected stage')
    for name in ('summary.json','final_audit.json','manifest.json','technical_evidence.tar.gz','archive_meta.json'):old.fresh(stage/name)
    meta=p.metadata(stage,True);plan,build,check,layers=(n.read(stage/name) for name in ('plan.json','build.json','check.json','layers.json'))
    quality=n.read(stage/'quality.json') if (stage/'quality.json').exists() else None
    for r in ([check,quality] if quality is not None else [check]):
        if any(r[k]!=v for k,v in meta.items()):raise ValueError('Source/build provenance drift')
    if quality is not None and quality['check_sha256']!=q.digest(stage/'check.json'):raise ValueError('Quality source drift')
    data=q.load_plan(q.BASE/'staging/quant_yolo_e92_r1')
    if plan['business_pairs']!=[dict(image_sha256=r['sha256'],image_type=k) for r in data['calibration']+data['holdout'] for k in (3,1)]:raise ValueError('Dataset drift')
    result=summarize(plan,build,check,quality,layers)
    prior=n.read(p.PRIOR/'final_audit.json');cfg_path=q.RELEASE/'edge_deploy/jetson/deployment.json';cfg=n.read(cfg_path)
    engines={k:q.digest(q.RELEASE/v['path']) for k,v in cfg['engines'].items()}
    plugins={v['path']:q.digest(q.RELEASE/v['path']) for r in cfg['engines'].values() for v in r.get('plugins',[])}
    if engines!=prior['engines'] or plugins!=prior['plugins'] or q.digest(cfg_path)!=prior['deployment_sha256']:raise ValueError('Original release changed')
    packages=sorted(tuple(line.split()) for line in command('dpkg-query','-W','-f=${Package}\t${Version}\n','nvidia-l4t-*').splitlines() if line.strip())
    pkgsha=hashlib.sha256(json.dumps(packages,separators=(',',':')).encode()).hexdigest()
    mount=command('findmnt','-n','-o','TARGET,SOURCE,FSTYPE,OPTIONS','-T',str(q.BASE))
    systemd=command('systemctl','show','autoocr-jetson.service','-p','MainPID','-p','ActiveState','-p','UnitFileState');listeners=command('ss','-ltn','sport = :5051')
    if pkgsha!=prior['l4t_package_rows_sha256'] or mount.split()!=prior['mount'].split() or set(systemd.splitlines())!={'MainPID=0','ActiveState=inactive','UnitFileState=disabled'} or len(listeners.splitlines())!=1:raise ValueError('BSP/mount/service changed')
    processes=[int(line.split()[0]) for line in command('ps','-eo','pid,args').splitlines() if '/ner_protected_fp32.py ' in line]
    if processes:raise ValueError('Experiment still running')
    q.save(stage/'final_audit.json',dict(deployment_sha256=q.digest(cfg_path),engines=engines,plugins=plugins,mount=mount,systemd=systemd,listeners=listeners,
        l4t_package_rows_sha256=pkgsha,previous_audit_sha256=p.AUDIT_SHA,experiment_process_ids=processes,memory=q.memory(),storage=shutil.disk_usage(q.BASE)._asdict(),
        note='Original seven engines/plugin/BSP/mount unchanged, HTTP inactive. No activation/count, power or clocks changes. Idle snapshot not service peak.'))
    reports=['plan.json','build.json','check.json','layers.json']+(['quality.json'] if quality is not None else [])
    result['reports_sha256']={name:q.digest(stage/name) for name in reports};q.save(stage/'summary.json',result)
    paths={name:stage/name for name in FILES};paths['previous_final_audit.json']=p.PRIOR/'final_audit.json'
    first=q.BASE/'staging/ner_protected_e119_r1'
    paths['first_attempt_prepare.log']=first/'prepare.log'
    paths['first_attempt_script.py']=first/'ner_protected_fp32.py'
    if quality is not None:paths['quality.json']=stage/'quality.json'
    rows=[]
    for name,path in paths.items():
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(q.BASE/'staging') or Path(name).name!=name:raise ValueError('Unsafe evidence')
        rows.append(dict(name=name,bytes=path.stat().st_size,sha256=q.digest(path)))
    q.save(stage/'manifest.json',dict(files=rows,note='Technical allowlist only, no engines/plugins/weights/images/token arrays/tensors/text/license.'))
    with (stage/'technical_evidence.tar.gz').open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as t:
            for name,path in paths.items():t.add(path,arcname=name,recursive=False)
            t.add(stage/'manifest.json',arcname='manifest.json',recursive=False)
    meta=dict(bytes=(stage/'technical_evidence.tar.gz').stat().st_size,sha256=q.digest(stage/'technical_evidence.tar.gz'),members=len(paths)+1)
    q.save(stage/'archive_meta.json',meta);print(json.dumps(meta));print(json.dumps(result))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',type=Path,required=True);archive(ap.parse_args().stage.resolve())
