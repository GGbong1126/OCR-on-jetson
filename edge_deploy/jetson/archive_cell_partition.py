"""E114 independently recomputable summary, immutable technical-only archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import evaluate_cell_partition as e

c,p,q=e.c,e.p,e.q
PREVIOUS_SHA='0196788078487cc5cc977a293aba161a11fda197b3c0aa880f089a8975bf3022'
SCRIPTS=('cell_partition.py','evaluate_cell_partition.py','archive_cell_partition.py',
    'diagnose_cell_partition.py','test_diagnose_cell_partition.py',
    'probe_cell_hotspot.py','layout_hotspot_mixed.py','optimize_rec_fp16.py','quantize_yolo_ptq.py',
    'test_cell_partition.py','test_evaluate_cell_partition.py','test_archive_cell_partition.py')
FILES=SCRIPTS+('tests.log','tests_complete.log','tests_diagnostic.log','geometry_diagnostic.json','geometry_diagnostic.log','plan.json','prepare.log',
    'build_prefix.json','build_prefix.log','build_suffix.json','build_suffix.log',
    'check.json','check.log','layers_prefix.json','layers_suffix.json','layers_fp32.json','layers_mixed.json',
    'quality.json','quality.log','summary.json','final_audit.json')
FAILED=('cell_partition.py','test_cell_partition.py','tests.log','plan.json','prepare.log','prepare_cli_error.log')
META=('prior_metadata','prior_archive_sha256','script_sha256','prior_check_sha256','plan_sha256')


def healthy(execution,counts):
    if execution['errors'] or execution['calls']!=counts:raise ValueError('Composed count/error drift')
    wanted=dict(prefix=sum(counts.values()),suffix=sum(counts.values()),**counts)
    if set(execution['parts'])!=set(wanted):raise ValueError('Incomplete composed execution')
    for name,n in wanted.items():
        if execution['parts'][name]['calls']!=n or execution['parts'][name]['errors']:raise ValueError('Native count/error drift')


def ranges(rows):
    nonempty=[r for r in rows if r['reference_effective_count'] and r['candidate_effective_count']==r['reference_effective_count']]
    return dict(calls=len(rows),geometry_passed=sum(bool(c.geometry_passed(r)) for r in rows),
        reference_effective_boxes=sum(r['reference_effective_count'] for r in rows),
        candidate_effective_boxes=sum(r['candidate_effective_count'] for r in rows),
        raw_allclose_calls=sum(all(v['allclose'] for v in r['raw'].values()) for r in rows),
        min_ordered_iou=min((r['min_ordered_iou'] for r in nonempty),default=None),
        max_coordinate_delta_pixels=max((r['max_coordinate_delta_pixels'] for r in nonempty),default=None),
        max_score_delta_diagnostic=max((r['max_score_delta_diagnostic'] for r in nonempty),default=None))


def summarize(plan,builds,check,quality,layers):
    if plan['status']!='partition_prepared' or plan['rules']!=c.RULES:raise ValueError('Plan differs')
    if check['status']!='full_cell_formats_checked_not_business_accepted' or quality['status']!='quality_checked_not_promoted':raise ValueError('Incomplete check')
    for r in [check,quality]+list(builds.values()):
        if any(r[k]!=check[k] for k in META) or any(r[k]!=plan[k] for k in META[:-1]):raise ValueError('Provenance differs')
    if check['rules']!=c.RULES or quality['rules']!=c.RULES:raise ValueError('Rules drift')
    if set(builds)!={'prefix','suffix'} or any(b['status']!='built_not_validated' or b['flags']!=0 for b in builds.values()):raise ValueError('Invalid builds')
    for name,info in layers.items():
        inspected=p.inspect(info,name) if name in ('fp32','mixed') else c.inspect_fp32(info,18 if name=='suffix' else 0)
        if inspected!=check['inspection'][name]:raise ValueError('Inspector drift')
    if set(layers)!={'fp32','mixed','prefix','suffix'} or len(check['rows'])!=6:raise ValueError('Check coverage differs')
    healthy(check['execution'],{'fp32':12,'mixed':12})
    check_summaries={}
    for v in ('fp32','mixed'):
        rows=[r['variants'][v] for r in check['rows']]
        if any(not r['repeat_exact'] or r['geometry_passed']!=c.geometry_passed(r) for r in rows):raise ValueError('Check repeat/geometry differs')
        check_summaries[v]=ranges(rows)
        key='control_geometry_passed' if v=='fp32' else 'mixed_geometry_passed'
        if check[key]!=(check_summaries[v]['geometry_passed']==6):raise ValueError('Stored check totals differ')
    pairs=[{k:r[k] for k in ('image_sha256','image_type')} for r in quality['rows']]
    if len(pairs)!=26 or pairs!=plan['image_cases'] or len({(r['image_sha256'],r['image_type']) for r in pairs})!=26:
        raise ValueError('Business coverage differs')
    summaries={};counts={v:0 for v in e.BRANCHES}
    for i,row in enumerate(quality['rows']):
        order=list(e.BRANCHES[i%3:]+e.BRANCHES[:i%3])
        if row['order']!=order:raise ValueError('Business order differs')
        for v in counts:
            n=len(row['inputs'][v]);counts[v]+=n
            if len(row['output_sha256'][v])!=n:raise ValueError('Output coverage differs')
        if row['original_cell_calls']!=len(row['inputs']['reference']):raise ValueError('Original calls differ')
    healthy(quality['candidate_execution'],{v:counts[v] for v in ('fp32','mixed')})
    if not counts['reference'] or quality['original_execution']['cell']['calls']!=counts['reference']+1:raise ValueError('Original cell call accounting differs')
    for r in quality['original_execution'].values():
        if r['errors'] or r.get('cpu_fallback',{}).get('errors',0):raise ValueError('Original execution errors')
    for key in ('reference_vs_fp32','reference_vs_mixed','fp32_vs_mixed'):
        refname,outname=key.split('_vs_');rows=[r['comparisons'][key] for r in quality['rows']]
        for outer,r in zip(quality['rows'],rows):
            same=outer['inputs'][refname]==outer['inputs'][outname]
            if r['same_original_input']!=same or (same and len(r['cell_comparisons'])!=len(outer['inputs'][refname])):raise ValueError('Inputs/geometry coverage differs')
            if r['passed']!=e.row_passed(r):raise ValueError('Stored business gate differs')
        n=sum(r['passed'] for r in rows)
        if n!=quality['totals'][key]:raise ValueError('Stored totals differ')
        summaries[key]=dict(combined_passed=n,business_only_passed=sum(bool(e.business_passed(r)) for r in rows),
            strict_equal_counts={k:sum(r['strict_equal'][k] for r in rows) for k in rows[0]['strict_equal']},
            failed_case_indices=[i for i,r in enumerate(rows) if not r['passed']],
            cell=ranges([v for r in rows for v in r['cell_comparisons']]))
    passed=all(v['combined_passed']==26 for v in summaries.values()) and check['control_geometry_passed'] and check['mixed_geometry_passed']
    return dict(status='quality_passed_not_benchmarked' if passed else 'quality_not_accepted',
        original_images=13,image_type_cases=26,business_executions=78,cell_calls_by_branch=counts,
        isolated_full_cell=check_summaries,full_business=summaries,benchmark_calls=0,accepted_for_release=False,
        note='FP32 partition control distinguishes graph/tactic changes from local half changes. Existing development data, no GT/accuracy or speedup claim. Explicit host copies and two extra micro outputs remain. Original release retained.')


def command(*args):return subprocess.check_output(args,text=True).strip()


def archive(stage):
    if stage!=q.BASE/'staging/cell_partition_e114_r2':raise ValueError('Unexpected stage')
    for name in ('summary.json','final_audit.json','manifest.json','technical_evidence.tar.gz','archive_meta.json'):c.old.fresh(stage/name)
    meta=c.metadata(stage);builds=c.require_builds(stage)
    plan,check,quality=(p.read(stage/n) for n in ('plan.json','check.json','quality.json'))
    if any(check[k]!=v or quality[k]!=v for k,v in meta.items()) or quality['check_sha256']!=q.digest(stage/'check.json') or quality['evaluator_sha256']!=q.digest(e.__file__):raise ValueError('Current provenance differs')
    layers={n:p.read(stage/('layers_'+n+'.json')) for n in ('prefix','suffix','fp32','mixed')}
    result=summarize(plan,builds,check,quality,layers)
    diagnostic=p.read(stage/'geometry_diagnostic.json')
    if diagnostic['status']!='diagnosed_not_acceptance' or len(diagnostic['rows'])!=6 or any(diagnostic[k]!=v for k,v in meta.items()) or diagnostic['check_sha256']!=q.digest(stage/'check.json') or diagnostic['diagnostic_sha256']!=q.digest(stage/'diagnose_cell_partition.py'):raise ValueError('Diagnostic changed')
    healthy(diagnostic['execution'],{'fp32':0,'mixed':6})
    result['reports_sha256']={n:q.digest(stage/n) for n in ('plan.json','check.json','quality.json','build_prefix.json','build_suffix.json')}
    q.save(stage/'summary.json',result)
    previous=c.PRIOR/'final_audit.json'
    if q.digest(previous)!=PREVIOUS_SHA:raise ValueError('Prior audit changed')
    old=p.read(previous);cfg_path=q.RELEASE/'edge_deploy/jetson/deployment.json';cfg=p.read(cfg_path)
    engines={k:q.digest(q.RELEASE/v['path']) for k,v in cfg['engines'].items()}
    plugins={v['path']:q.digest(q.RELEASE/v['path']) for r in cfg['engines'].values() for v in r.get('plugins',[])}
    if engines!=old['engines'] or plugins!=old['plugins'] or q.digest(cfg_path)!=old['deployment_sha256']:raise ValueError('Release changed')
    packages=sorted(tuple(line.split()) for line in command('dpkg-query','-W','-f=${Package}\t${Version}\n','nvidia-l4t-*').splitlines() if line.strip())
    pkgsha=hashlib.sha256(json.dumps(packages,separators=(',',':')).encode()).hexdigest()
    mount=command('findmnt','-n','-o','TARGET,SOURCE,FSTYPE,OPTIONS','-T',str(q.BASE))
    systemd=command('systemctl','show','autoocr-jetson.service','-p','MainPID','-p','ActiveState','-p','UnitFileState')
    listeners=command('ss','-ltn','sport = :5051')
    if pkgsha!=old['l4t_package_rows_sha256'] or mount.split()!=old['mount'].split() or set(systemd.splitlines())!={'MainPID=0','ActiveState=inactive','UnitFileState=disabled'} or len(listeners.splitlines())!=1:raise ValueError('BSP/mount/service changed')
    processes=[int(line.split()[0]) for line in command('ps','-eo','pid,args').splitlines() if any(s in line for s in ('/cell_partition.py ','/evaluate_cell_partition.py '))]
    if processes:raise ValueError('Experiment still running')
    q.save(stage/'final_audit.json',dict(deployment_sha256=q.digest(cfg_path),engines=engines,plugins=plugins,mount=mount,
        systemd=systemd,listeners=listeners,l4t_package_rows_sha256=pkgsha,previous_audit_sha256=PREVIOUS_SHA,
        experiment_process_ids=processes,memory=q.memory(),storage=shutil.disk_usage(q.BASE)._asdict(),
        note='Original release/BSP/mount unchanged. No HTTP/activation/count/clocks/power changes. Idle snapshot not service peak.'))
    paths={n:stage/n for n in FILES}
    paths.update({'failed_r1_'+n:q.BASE/'staging/cell_partition_e114_r1'/n for n in FAILED})
    paths['previous_final_audit.json']=previous
    rows=[]
    for name,path in paths.items():
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(q.BASE/'staging') or Path(name).name!=name:raise ValueError('Unsafe evidence')
        rows.append(dict(name=name,bytes=path.stat().st_size,sha256=q.digest(path)))
    q.save(stage/'manifest.json',dict(files=rows,note='Technical allowlist only; no models, images, tensors, OCR text or license.'))
    with (stage/'technical_evidence.tar.gz').open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as t:
            for n,path in paths.items():t.add(path,arcname=n,recursive=False)
            t.add(stage/'manifest.json',arcname='manifest.json',recursive=False)
    m=dict(bytes=(stage/'technical_evidence.tar.gz').stat().st_size,sha256=q.digest(stage/'technical_evidence.tar.gz'),members=len(paths)+1)
    q.save(stage/'archive_meta.json',m);print(json.dumps(m));print(json.dumps(result))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',type=Path,required=True)
    archive(ap.parse_args().stage.resolve())
