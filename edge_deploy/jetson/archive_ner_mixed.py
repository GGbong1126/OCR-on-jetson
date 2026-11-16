"""E115 technical-only archive and independently recomputable acceptance summary."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import evaluate_ner_mixed as e
import probe_ner_masks as masks

n,q,old=e.n,e.q,e.old
PREVIOUS_SHA='eb16295e9d13cf391649a9831e85d65750f151537c2777d7b811b5e5a0ea0e92'
SCRIPTS=('ner_mixed_trial.py','evaluate_ner_mixed.py','probe_ner_masks.py','archive_ner_mixed.py',
    'test_ner_mixed_trial.py','test_evaluate_ner_mixed.py','test_archive_ner_mixed.py',
    'layout_hotspot_mixed.py','optimize_rec_fp16.py','quantize_yolo_ptq.py')
FILES=SCRIPTS+('tests.log','tests_complete.log','tests_final.log','plan.json','prepare.log','build.json','build.log',
    'check.json','check.log','layers.json','quality.json','quality.log','mask_probe.json','mask_probe.log',
    'summary.json','final_audit.json','constant_audit.json')


def execution(row,calls):
    if row['errors'] or row['calls']!=calls:raise ValueError('Execution counts/errors differ')


def summary(plan,build,check,quality,probe,layers):
    if plan['status']!='prepared' or plan['rules']!=n.RULES or build['status']!='built_not_validated' or build['flags']!=1025:raise ValueError('Plan/build differ')
    if check['status']!='finite_checked_not_business_accepted' or quality['status'] not in ('quality_consistent_not_benchmarked','not_accepted'):raise ValueError('Incomplete checks')
    for key in ('source_sha256','deployment_sha256','prior_archive_sha256','prior_audit_sha256','engine_sha256','dataset_plan_sha256','artifacts_sha256','script_sha256','helpers_sha256'):
        if any(r[key]!=plan[key] for r in (build,check,quality,probe)):raise ValueError('Provenance differs')
    for key in ('plan_sha256','candidate_engine_sha256'):
        if any(r[key]!=build[key] for r in (check,quality,probe)):raise ValueError('Build provenance differs')
    inspection=e.inspect(layers,plan['graph']['protected_nodes'])
    if inspection!=check['inspection'] or quality['precision_boundary_verified']!=inspection['precision_boundary_verified']:raise ValueError('Inspector differs')
    if len(check['rows'])!=6 or [n.describe(r) for r in check['rows']]!=plan['rows']:raise ValueError('Original inputs differ')
    if [r['length'] for r in check['boundaries']]!=[2,128,512]:raise ValueError('Boundary coverage differs')
    for r in check['rows']:
        if not r['repeat_exact'] or not r['metrics']['all_finite']:raise ValueError('Unstable real outputs')
    if any(not r['metrics']['all_finite'] for r in check['boundaries']):raise ValueError('Invalid boundary outputs')
    execution(check['candidate_execution'],15);execution(check['boundary_reference_execution'],3)
    if probe['status']!='synthetic_masks_finite_not_accuracy_validated' or [(r['length'],r['segment']) for r in probe['rows']]!=[(l,s) for l in (8,128,512) for s in (0,1)]:raise ValueError('Mask coverage differs')
    if any(not r['metrics']['all_finite'] or not r['repeat_exact'] for r in probe['rows']):raise ValueError('Invalid masks')
    execution(probe['reference_execution'],6);execution(probe['candidate_execution'],12)
    rows=quality['rows'];pairs=[(r['image_sha256'],r['image_type']) for r in rows]
    if len(rows)!=26 or len(set(pairs))!=26 or len({v[0] for v in pairs})!=13 or any({k for h,k in pairs if h==im}!={3,1} for im,_ in pairs):raise ValueError('Business coverage differs')
    counts={b:0 for b in ('reference','candidate')}
    for i,r in enumerate(rows):
        if r['order']!=list(('reference','candidate') if i%2==0 else ('candidate','reference')):raise ValueError('Order differs')
        if r['passed']!=e.quality_passed(r) or r['same_ner_inputs']!=(r['inputs']['reference']==r['inputs']['candidate']):raise ValueError('Stored gate differs')
        for b in counts:
            size=len(r['inputs'][b]);counts[b]+=size
            if not size or len(r['ner_output_sha256'][b])!=size or r['entities']['postprocess_calls_'+b]!=1:raise ValueError('Missing NER/entity trace')
    execution(quality['candidate_execution'],counts['candidate']);execution(quality['original_execution']['ner'],counts['reference']+1)
    for stats in quality['original_execution'].values():
        if stats['errors'] or stats.get('cpu_fallback',{}).get('errors',0):raise ValueError('Original runtime errors')
    passed=sum(r['passed'] for r in rows)
    expected='quality_consistent_not_benchmarked' if passed==26 and inspection['precision_boundary_verified'] else 'not_accepted'
    if quality['business_passed_count']!=passed or quality['status']!=expected:raise ValueError('Quality status differs')
    return dict(status=expected,original_images=13,business_cases=26,business_executions=52,ner_calls=counts,
        business_passed_count=passed,failed_case_indices=[i for i,r in enumerate(rows) if not r['passed']],
        same_input_cases=sum(r['same_ner_inputs'] for r in rows),token_labels_equal_cases=sum(r['token_labels_equal'] for r in rows),
        strict_equal_counts={k:sum(r['strict_equal'][k] for r in rows) for k in rows[0]['strict_equal']},
        entity_equal_counts={k:sum(r['entities'][k] for r in rows) for k in ('raw_entity_spans_equal','cleaned_entity_spans_equal','threshold_membership_equal')},
        max_aligned_entity_score_delta=max(r['entities']['max_aligned_entity_score_delta'] for r in rows),
        field_equal_counts={k:sum(r['fields']['api_special_text_equal'][k] for r in rows) for k in ('title','category_no','page_no')},
        extra_cpu_fallback_cases=sum(r['cpu_fallback_calls']['reference']!=r['cpu_fallback_calls']['candidate'] for r in rows),
        real_probe_calls=6,real_probe_label_equal=sum(r['metrics']['active_token_labels_equal'] for r in check['rows']),
        real_probe_allclose=sum(r['metrics']['allclose_diagnostic'] for r in check['rows']),
        real_probe_max_abs_error=max(r['metrics']['max_abs_error'] for r in check['rows']),
        synthetic_mask_cases=6,synthetic_mask_label_equal=sum(r['metrics']['active_token_labels_equal'] for r in probe['rows']),
        inspection=inspection,engine_bytes=build['engine_bytes'],build_s=build['build_s'],benchmark_calls=0,accepted_for_release=False,
        note='Existing development images, no GT/accuracy or performance claim. Opaque fusion prevents proof of the declared internal precision boundary. Native Session stats precision string is inherited legacy metadata, not a measurement of candidate arithmetic. Original release retained.')


def command(*args):return subprocess.check_output(args,text=True).strip()


def archive(stage):
    if stage!=q.BASE/'staging/ner_mixed_e115_r1':raise ValueError('Unexpected stage')
    for name in ('summary.json','final_audit.json','constant_audit.json','manifest.json','technical_evidence.tar.gz','archive_meta.json'):old.fresh(stage/name)
    meta=e.metadata(stage)
    plan,build,check,quality,probe,layers=(n.read(stage/name) for name in ('plan.json','build.json','check.json','quality.json','mask_probe.json','layers.json'))
    for r in (check,quality,probe):
        if any(r[k]!=v for k,v in meta.items()):raise ValueError('Current metadata drift')
    if quality['check_sha256']!=q.digest(stage/'check.json') or probe['check_sha256']!=q.digest(stage/'check.json') or probe['probe_sha256']!=q.digest(masks.__file__):raise ValueError('Probe provenance drift')
    data=q.load_plan(q.BASE/'staging/quant_yolo_e92_r1')
    expected=[(r['sha256'],k) for r in data['calibration']+data['holdout'] for k in (3,1)]
    if [(r['image_sha256'],r['image_type']) for r in quality['rows']]!=expected:raise ValueError('Dataset drift')
    result=summary(plan,build,check,quality,probe,layers)
    import onnx
    from onnx import numpy_helper
    graph=onnx.load(str(n.SOURCE)).graph
    found=[node for node in graph.node if node.name=='/inner/bert/Constant_3']
    if len(found)!=1 or found[0].op_type!='Constant':raise ValueError('Mask constant changed')
    value=numpy_helper.to_array(next(a.t for a in found[0].attribute if a.name=='value'))
    if value.size!=1:raise ValueError('Not a scalar')
    q.save(stage/'constant_audit.json',dict(source_sha256=n.SOURCE_SHA,node_name=found[0].name,dtype=str(value.dtype),value=float(value.item()),
        note='Original attention-mask scalar only; unchanged. Build warning does not prove output NaN, and finite output does not prove internal precision.'))
    previous=n.PRIOR/'final_audit.json'
    if q.digest(previous)!=PREVIOUS_SHA:raise ValueError('Prior audit changed')
    prior=n.read(previous);cfg_path=q.RELEASE/'edge_deploy/jetson/deployment.json';cfg=n.read(cfg_path)
    engines={k:q.digest(q.RELEASE/v['path']) for k,v in cfg['engines'].items()}
    plugins={v['path']:q.digest(q.RELEASE/v['path']) for r in cfg['engines'].values() for v in r.get('plugins',[])}
    if engines!=prior['engines'] or plugins!=prior['plugins'] or q.digest(cfg_path)!=prior['deployment_sha256']:raise ValueError('Release changed')
    packages=sorted(tuple(line.split()) for line in command('dpkg-query','-W','-f=${Package}\t${Version}\n','nvidia-l4t-*').splitlines() if line.strip())
    pkgsha=hashlib.sha256(json.dumps(packages,separators=(',',':')).encode()).hexdigest()
    mount=command('findmnt','-n','-o','TARGET,SOURCE,FSTYPE,OPTIONS','-T',str(q.BASE))
    systemd=command('systemctl','show','autoocr-jetson.service','-p','MainPID','-p','ActiveState','-p','UnitFileState');listeners=command('ss','-ltn','sport = :5051')
    if pkgsha!=prior['l4t_package_rows_sha256'] or mount.split()!=prior['mount'].split() or set(systemd.splitlines())!={'MainPID=0','ActiveState=inactive','UnitFileState=disabled'} or len(listeners.splitlines())!=1:raise ValueError('BSP/mount/service changed')
    processes=[int(line.split()[0]) for line in command('ps','-eo','pid,args').splitlines() if any(v in line for v in ('/ner_mixed_trial.py ','/evaluate_ner_mixed.py ','/probe_ner_masks.py '))]
    if processes:raise ValueError('Experiment still running')
    q.save(stage/'final_audit.json',dict(deployment_sha256=q.digest(cfg_path),engines=engines,plugins=plugins,mount=mount,systemd=systemd,listeners=listeners,
        l4t_package_rows_sha256=pkgsha,previous_audit_sha256=PREVIOUS_SHA,experiment_process_ids=processes,memory=q.memory(),storage=shutil.disk_usage(q.BASE)._asdict(),
        note='Original release/BSP/mount unchanged. No HTTP, activation/count, clocks or power changes. Idle snapshot not service peak.'))
    result['reports_sha256']={name:q.digest(stage/name) for name in ('plan.json','build.json','check.json','quality.json','mask_probe.json','layers.json','constant_audit.json')}
    q.save(stage/'summary.json',result)
    paths={name:stage/name for name in FILES};paths['previous_final_audit.json']=previous;rows=[]
    for name,path in paths.items():
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(q.BASE/'staging') or Path(name).name!=name:raise ValueError('Unsafe evidence')
        rows.append(dict(name=name,bytes=path.stat().st_size,sha256=q.digest(path)))
    q.save(stage/'manifest.json',dict(files=rows,note='Technical allowlist only; no models/images/tensors/token arrays/document text/license.'))
    with (stage/'technical_evidence.tar.gz').open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as t:
            for name,path in paths.items():t.add(path,arcname=name,recursive=False)
            t.add(stage/'manifest.json',arcname='manifest.json',recursive=False)
    m=dict(bytes=(stage/'technical_evidence.tar.gz').stat().st_size,sha256=q.digest(stage/'technical_evidence.tar.gz'),members=len(paths)+1)
    q.save(stage/'archive_meta.json',m);print(json.dumps(m));print(json.dumps({k:v for k,v in result.items() if k not in ('inspection','reports_sha256')}))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',type=Path,required=True)
    archive(ap.parse_args().stage.resolve())
