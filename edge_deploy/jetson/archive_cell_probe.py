"""E113 independent local-safe summary and board-only technical archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import probe_cell_hotspot as p

q=p.q
PREVIOUS=q.BASE/'staging/layout_semantic_e112_r1/final_audit.json'
PREVIOUS_SHA='a651aee557999300ac3be76cce90c9504977dffdcde3207a0aa4ab3acd85bafb'
FILES=('probe_cell_hotspot.py','archive_cell_probe.py','layout_hotspot_mixed.py',
       'optimize_rec_fp16.py','quantize_yolo_ptq.py','test_probe_cell_hotspot.py',
       'test_archive_cell_probe.py','tests.log','tests_complete.log','ranges.json','ranges.log',
       'build_fp32.json','build_fp32.log','build_mixed.json','build_mixed.log',
       'check.json','check.log','layers_fp32.json','layers_mixed.json','summary.json','final_audit.json')
FAILED=('probe_cell_hotspot.py','ranges.json','ranges.log','tests.log')


def summarize(ranges, builds, check, inspectors):
    p.range_gate(ranges)
    if check['status']!='isolated_boundary_compared_not_full_model_accepted' or check['rules']!=p.RULES:
        raise ValueError('Incomplete block check')
    meta=('source_sha256','engine_sha256','deployment_sha256','plugin_sha256','dataset_plan_sha256','script_sha256','helpers_sha256')
    if any(check[k]!=ranges[k] for k in meta):raise ValueError('Range/check provenance differs')
    n=len(ranges['rows'])
    if len(check['rows'])!=n or set(builds)!={'fp32','mixed'} or set(inspectors)!=set(builds):raise ValueError('Incomplete evidence')
    if len(ranges['pages'])!=6 or sum(r['cell_calls'] for r in ranges['pages'])!=n:raise ValueError('Cell coverage differs')
    pairs={(r['image_sha256'],r['image_type']) for r in ranges['pages']}
    images={h for h,_ in pairs}
    if len(images)!=3 or pairs!={(h,k) for h in images for k in (3,1)}:raise ValueError('Page coverage differs')
    for variant,b in builds.items():
        if b['status']!='built_not_validated' or b['variant']!=variant or b['rules']!=p.RULES or any(b[k]!=ranges[k] for k in meta):
            raise ValueError('Build provenance differs')
        if p.inspect(inspectors[variant],variant)!=check['inspection'][variant]:raise ValueError('Actual format evidence differs')
        if check['execution'][variant]['calls']!=2*n or check['execution'][variant]['errors']:
            raise ValueError('Execution count/errors differ')
    controls=0; mixed=0; values={name:[] for name in p.OUTPUTS}
    for index,(r,c) in enumerate(zip(ranges['rows'],check['rows'])):
        keys=('image_sha256','image_type','call_index','inputs','original_output_sha256','feature_sha256')
        if any(r[k]!=c[k] for k in keys) or r['call_index']!=index or not c['repeat_exact']:raise ValueError('Input/output drift')
        if set(c['mixed_vs_fp32'])!=set(p.OUTPUTS):raise ValueError('Missing comparison')
        controls+=all(v['allclose_diagnostic'] for v in c['variants']['fp32'].values())
        mixed+=all(v['allclose_diagnostic'] for v in c['mixed_vs_fp32'].values())
        for name in p.OUTPUTS:
            metric=c['mixed_vs_fp32'][name]
            if not all(p.np.isfinite(metric[k]) and metric[k]>=0 for k in ('mae','max_abs_error')):raise ValueError('Invalid numeric metric')
            values[name].append(metric)
    if controls!=check['control_allclose_cases'] or mixed!=check['mixed_allclose_cases']:raise ValueError('Stored counts differ')
    return dict(status='isolated_cell_boundary_diagnosed_not_business_accepted',
        original_pages=6,real_cell_calls=n,candidate_calls=2*n,control_calls=2*n,
        control_allclose_cases=controls,mixed_allclose_cases=mixed,
        error_ranges={name:{k:[min(r[k] for r in rows),max(r[k] for r in rows)] for k in ('mae','max_abs_error')} for name,rows in values.items()},
        both_original_outputs_consumed=ranges['consumer']['both_model_outputs_consumed'],
        graph_nodes=ranges['graph']['graph_nodes'],live_nodes=ranges['graph']['live_nodes'],
        full_model_candidate_built=False,business_candidate_cases=0,benchmark_calls=0,accepted_for_release=False,
        note='Only original CPU-prefix features and isolated GPU block controls. Actual FP32 SiLU/boundaries checked. No hidden original GPU feature equivalence, GT, business correctness or speedup claimed.')


def command(*args):return subprocess.check_output(args,text=True).strip()


def archive(stage):
    if stage!=q.BASE/'staging/cell_probe_e113_r2':raise ValueError('Unexpected stage')
    for n in ('summary.json','final_audit.json','manifest.json','technical_evidence.tar.gz','archive_meta.json'):p.old.fresh(stage/n)
    ranges=p.require_ranges(stage);check=p.read(stage/'check.json')
    builds={v:p.read(stage/('build_'+v+'.json')) for v in ('fp32','mixed')}
    layers={v:p.read(stage/('layers_'+v+'.json')) for v in builds}
    for v,b in builds.items():
        if b['ranges_sha256']!=q.digest(stage/'ranges.json') or b['candidate_engine_sha256']!=q.digest(stage/(v+'.engine')):
            raise ValueError('Build artifacts changed')
        if check['build_sha256'][v]!=q.digest(stage/('build_'+v+'.json')):raise ValueError('Check provenance changed')
    if check['ranges_sha256']!=q.digest(stage/'ranges.json'):raise ValueError('Ranges changed')
    result=summarize(ranges,builds,check,layers)
    evidence=['ranges.json','check.json']+['%s_%s.json'%(kind,v) for kind in ('build','layers') for v in builds]
    result['reports_sha256']={n:q.digest(stage/n) for n in evidence}
    q.save(stage/'summary.json',result)
    if q.digest(PREVIOUS)!=PREVIOUS_SHA:raise ValueError('Previous audit differs')
    old=p.read(PREVIOUS);cfg_path=q.RELEASE/'edge_deploy/jetson/deployment.json';cfg=p.read(cfg_path)
    engines={k:q.digest(q.RELEASE/v['path']) for k,v in cfg['engines'].items()}
    plugins={v['path']:q.digest(q.RELEASE/v['path']) for r in cfg['engines'].values() for v in r.get('plugins',[])}
    for new,k in ((engines,'engines'),(plugins,'plugins'),(q.digest(cfg_path),'deployment_sha256')):
        if new!=old[k]:raise ValueError('Original release changed')
    norm=lambda v:sorted(tuple(line.split()) for line in v.splitlines() if line.strip())
    packages=norm(command('dpkg-query','-W','-f=${Package}\t${Version}\n','nvidia-l4t-*'))
    pkgsha=hashlib.sha256(json.dumps(packages,separators=(',',':')).encode()).hexdigest()
    if pkgsha!=old['l4t_package_rows_sha256']:raise ValueError('BSP changed')
    mount=command('findmnt','-n','-o','TARGET,SOURCE,FSTYPE,OPTIONS','-T',str(q.BASE))
    systemd=command('systemctl','show','autoocr-jetson.service','-p','MainPID','-p','ActiveState','-p','UnitFileState')
    listeners=command('ss','-ltn','sport = :5051')
    if mount.split()!=old['mount'].split() or set(systemd.splitlines())!={'MainPID=0','ActiveState=inactive','UnitFileState=disabled'} or len(listeners.splitlines())!=1:
        raise ValueError('Mount/service differs')
    processes=[int(line.split()[0]) for line in command('ps','-eo','pid,args').splitlines() if 'probe_cell_hotspot.py ' in line]
    if processes:raise ValueError('Cell experiment still running')
    audit=dict(deployment_sha256=q.digest(cfg_path),engines=engines,plugins=plugins,mount=mount,systemd=systemd,
        listeners=listeners,l4t_package_rows_sha256=pkgsha,previous_audit_sha256=PREVIOUS_SHA,
        experiment_process_ids=processes,memory=q.memory(),storage=shutil.disk_usage(q.BASE)._asdict(),
        note='Original release, BSP, mount unchanged; no HTTP, activation, counter, clocks or power changes. Idle snapshot not service peak.')
    q.save(stage/'final_audit.json',audit)
    paths={n:stage/n for n in FILES}
    paths.update({'failed_r1_'+n:q.BASE/'staging/cell_probe_e113_r1'/n for n in FAILED})
    entries=[]
    for name,path in paths.items():
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(q.BASE/'staging') or Path(name).name!=name:
            raise ValueError('Unsafe/missing technical evidence')
        entries.append(dict(name=name,bytes=path.stat().st_size,sha256=q.digest(path)))
    q.save(stage/'manifest.json',dict(files=entries,note='Explicit technical allowlist; no model, image, tensor, recognized text or license.'))
    with (stage/'technical_evidence.tar.gz').open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as t:
            for name,path in paths.items():t.add(path,arcname=name,recursive=False)
            t.add(stage/'manifest.json',arcname='manifest.json',recursive=False)
    meta=dict(bytes=(stage/'technical_evidence.tar.gz').stat().st_size,sha256=q.digest(stage/'technical_evidence.tar.gz'),members=len(paths)+1)
    q.save(stage/'archive_meta.json',meta);print(json.dumps(meta));print(json.dumps(result))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',type=Path,required=True)
    archive(ap.parse_args().stage.resolve())
