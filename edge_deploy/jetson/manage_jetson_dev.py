"""E129 technical evidence: scoped deployment, HTTP acceptance, immutable audit.

Installation/start/stop are explicit administrator commands, not done here.
No pictures, recognized values, credentials or licensing data in saved reports.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from urllib.request import build_opener, ProxyHandler, Request
import serve_jetson_dev as dev
import health_jetson_dev as health
import health_rec_wide as core
import smoke_business as smoke

STAGE=dev.BASE/'staging/jetson_dev_e129_r1'
PRIOR=dev.BASE/'staging/rec_service_e127_r2'
AUDIT_SHA='798f271c8359e8365807f3dcbce4c91d164278e2a18bb77e97141c86d278c457'
DELIVERY=('serve_jetson_dev.py','health_jetson_dev.py','health_rec_wide.py',
          'start_jetson_dev.sh','autoocr-jetson-dev.service')
SOURCES=DELIVERY+('manage_jetson_dev.py','smoke_business.py','test_jetson_dev.py','test_rec_service.py',
    'test_rec_wide_adapter.py','serve_rec_wide.py','rec_wide_adapter.py','health_jetson.py',
    'validate_rec_service.py','start_rec_wide.sh','autoocr-jetson-wide.service','REC_WIDE_SERVICE.md')


def read(path): return json.loads(Path(path).read_text())
def save(path,data):
    with Path(path).open('x') as stream: json.dump(data,stream,indent=2,allow_nan=False)
def cmd(*args): return subprocess.check_output(args,text=True).strip()
def signature(value): return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def licensing_metadata():
    paths={'original_license':dev.BASE/'runtime/license.dat','dev_license':dev.RUNTIME/'license.dat',
        'counter':Path('/home/xchq/.cache/sys_config_d/sys_kernel_event.log'),
        'used_keys':Path('/home/xchq/.cache/sys_config_d/IconCache_ver.db'),
        'old_counter':Path('/home/xchq/Documents/Syslib/syslicense.log'),
        'old_keys':Path('/home/xchq/Documents/Syslib/sys_ver_control.dat')}
    return {name:dict(exists=p.exists(), size=p.stat().st_size if p.exists() else None,
                     mtime_ns=p.stat().st_mtime_ns if p.exists() else None) for name,p in paths.items()}


def audit():
    if dev.sha(PRIOR/'final_audit.json')!=AUDIT_SHA: raise ValueError('Prior audit changed')
    before=read(PRIOR/'final_audit.json'); cfg=dev.RELEASE/'edge_deploy/jetson/deployment.json'; c=read(cfg)
    packages=sorted(tuple(line.split()) for line in cmd('dpkg-query','-W','-f=${Package}\t${Version}\n','nvidia-l4t-*').splitlines() if line.strip())
    state=dict(deployment_sha256=dev.sha(cfg), engines={k:dev.sha(dev.RELEASE/v['path']) for k,v in c['engines'].items()},
        plugins={p['path']:dev.sha(dev.RELEASE/p['path']) for row in c['engines'].values() for p in row.get('plugins',[])},
        package={name:dev.sha(dev.OVERLAY/name) for name in before['package']},
        mount=cmd('findmnt','-n','-o','TARGET,SOURCE,FSTYPE,OPTIONS','-T',str(dev.BASE)),
        l4t_package_rows_sha256=hashlib.sha256(json.dumps(packages,separators=(',',':')).encode()).hexdigest(),
        systemd=cmd('systemctl','show','autoocr-jetson.service','-p','MainPID','-p','ActiveState','-p','UnitFileState'))
    for key in state:
        if (state[key].split()!=before[key].split() if key=='mount' else state[key]!=before[key]):
            raise ValueError('Original deployment changed: '+key)
    state.update(source_sha256={name:dev.sha(dev.RELEASE/name) for name in ('server.py','server_config.py','jetson_source_manifest.json')},
        licensing_metadata=licensing_metadata(), listeners=cmd('ss','-H','-ltn','sport = :5051'),
        dev_service=cmd('systemctl','show','autoocr-jetson-dev.service','-p','MainPID','-p','ActiveState','-p','UnitFileState'),
        storage=shutil.disk_usage(dev.BASE)._asdict())
    return state


def prepare():
    if dev.PACKAGE.exists() or (STAGE/'plan.json').exists(): raise ValueError('Fresh delivery required')
    before=audit()
    if before['listeners']: raise ValueError('Port occupied')
    dev.PACKAGE.mkdir()
    for name in DELIVERY:
        with (STAGE/name).open('rb') as src,(dev.PACKAGE/name).open('xb') as dest: shutil.copyfileobj(src,dest)
    manifest=dict(kind=dev.MODE,files={name:dev.sha(dev.PACKAGE/name) for name in DELIVERY})
    save(dev.PACKAGE/'manifest.json',manifest)
    save(STAGE/'delivery_manifest.json',manifest)
    save(STAGE/'plan.json',dict(status='prepared_development_no_authorization',initial_audit=before,
        sources={n:dev.sha(STAGE/n) for n in SOURCES},delivery=manifest, user_authorized_no_limits=True))
    print('Independent dev package prepared; original license/counters untouched')


def call(endpoint,payload=None):
    data=None if payload is None else json.dumps(payload,ensure_ascii=False).encode()
    opener=build_opener(ProxyHandler({}),core.NoRedirect())
    req=Request('http://127.0.0.1:5051'+endpoint,data=data,headers={'Content-Type':'application/json'})
    with opener.open(req,timeout=240) as response:
        raw=response.read(24*1024*1024+1)
        if len(raw)>24*1024*1024: raise ValueError('Response too large')
        if response.status!=200 or response.headers.get('X-AutoOCR-Mode')!=dev.MODE: raise ValueError('Wrong dev HTTP service')
        return json.loads(raw)


def run_smoke(round_name):
    out=STAGE/(round_name+'.json')
    if out.exists(): raise ValueError('New report required')
    before=call('/healthz')
    if not health.ready(before): raise ValueError('Dev service not ready')
    for endpoint in ('/OCR','/Extract'):
        r=call(endpoint,{})
        if r.get('status')!='400' or r.get('isSuccess') is not False: raise ValueError('Validation contract changed')
    if call('/healthz')['models_execution']!=before['models_execution']: raise ValueError('Invalid payload ran models')
    cases=read(dev.BASE/'staging/rec_soak_e126_r1/plan.json')['cases'][:26]
    if len(cases)!=26 or len({r['image_sha256'] for r in cases})!=13: raise ValueError('Unexpected 13-image basis')
    if round_name=='smoke_restart':
        wanted={r['image_sha256'] for r in cases if r['wide_calls']>0}
        wanted.add(cases[0]['image_sha256'])
        cases=[r for r in cases if r['image_sha256'] in wanted]
    wanted={r['image_sha256'] for r in cases}; files={}
    for p in (dev.BASE/'testpic').rglob('*'):
        if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png'):
            digest=dev.sha(p)
            if digest in wanted: files[digest]=p
    if set(files)!=wanted: raise ValueError('Existing test pictures missing')
    rows=[]
    for case in cases:
        prior=call('/healthz'); start=time.monotonic()
        ocr=call('/OCR',dict(img=base64.b64encode(files[case['image_sha256']].read_bytes()).decode(),imageType=str(case['image_type'])))
        stats=smoke.ocr_statistics(ocr)
        if stats['markdown_body_characters']<=0: raise ValueError('No real body for Extract')
        middle=call('/healthz')
        extracted=call('/Extract',dict(ocrResult=ocr['msg'],topFields=[dict(name='title',type='String',description='title')],
            nerFields=[dict(name='姓名',description='姓名')]))
        counts=smoke.extract_statistics(extracted); counts.pop('note',None)
        after=call('/healthz')
        if not health.ready(after) or after['worker_pid']!=before['worker_pid']: raise ValueError('Worker drift/unhealthy')
        delta={k:after['models_execution'][k]['calls']-prior['models_execution'][k]['calls'] for k in core.ROLES}
        if delta['ner']<=0 or middle['models_execution']['ner']['calls']!=prior['models_execution']['ner']['calls']:
            raise ValueError('OCR/NER execution scope wrong')
        if (case['image_type']==1 and delta['enhance']<=0) or (case['image_type']==3 and delta['enhance']!=0):
            raise ValueError('Enhancement route wrong')
        wide=after['models_execution']['rec']['wide_execution']['calls']-prior['models_execution']['rec']['wide_execution']['calls']
        if wide!=case['wide_calls']: raise ValueError('Wide route count drift')
        rows.append(dict(image_sha256=case['image_sha256'],image_type=case['image_type'],ocr_sha256=signature(ocr),
            extract_sha256=signature(extracted),ocr_statistics=stats,extract_statistics=counts,
            model_calls=delta,wide_calls=wide,single_pair_wall_s=time.monotonic()-start))
        print('HTTP OCR + Extract passed',len(rows),'/',len(cases),flush=True)
    if round_name=='smoke_restart':
        first=read(STAGE/'smoke_initial.json')
        if before['worker_pid']==first['before']['worker_pid']: raise ValueError('Worker did not restart')
        index={(r['image_sha256'],r['image_type']):r for r in first['rows']}
        for row in rows:
            original=index[row['image_sha256'],row['image_type']]
            if any(row[k]!=original[k] for k in ('ocr_sha256','extract_sha256','model_calls','wide_calls')):
                raise ValueError('Post-restart output changed')
    final=audit(); plan=read(STAGE/'plan.json')
    for key in ('licensing_metadata','source_sha256'):
        if final[key]!=plan['initial_audit'][key]: raise ValueError('Original state changed: '+key)
    result=dict(status='development_http_passed_not_gt_or_benchmark',before=before,after=call('/healthz'),
        rows=rows,invalid_payloads_rejected=2,authorization_limits_disabled=True,licensing_files_unchanged=True,
        final_audit=final)
    save(out,result)
    print(json.dumps(dict(status=result['status'],cases=len(rows),wide_calls=sum(r['wide_calls'] for r in rows))))


def archive():
    plan=read(STAGE/'plan.json'); final=audit()
    for name,digest in plan['sources'].items():
        if dev.sha(STAGE/name)!=digest: raise ValueError('Sources changed')
    for name,digest in plan['delivery']['files'].items():
        if dev.sha(dev.PACKAGE/name)!=digest: raise ValueError('Delivery changed')
    for key in ('licensing_metadata','source_sha256'):
        if final[key]!=plan['initial_audit'][key]: raise ValueError('Original state changed')
    for name in ('smoke_initial','smoke_restart'):
        if read(STAGE/(name+'.json'))['status']!='development_http_passed_not_gt_or_benchmark': raise ValueError('Smoke incomplete')
    if 'ActiveState=active' not in final['dev_service'] or 'UnitFileState=enabled' not in final['dev_service']:
        raise ValueError('Dev service not persistent/ready')
    save(STAGE/'final_audit.json',final)
    names=SOURCES+('plan.json','delivery_manifest.json','smoke_initial.json','smoke_restart.json','final_audit.json','tests.log','preflight.log','unit_verify.log')
    paths={name:STAGE/name for name in names}
    save(STAGE/'evidence_manifest.json',dict(files=[dict(name=n,bytes=p.stat().st_size,sha256=dev.sha(p)) for n,p in paths.items()]))
    with (STAGE/'technical_evidence.tar.gz').open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as t:
            for n,p in paths.items(): t.add(p,arcname=n,recursive=False)
            t.add(STAGE/'evidence_manifest.json',arcname='evidence_manifest.json',recursive=False)
    meta=dict(bytes=(STAGE/'technical_evidence.tar.gz').stat().st_size,sha256=dev.sha(STAGE/'technical_evidence.tar.gz'),
        members=len(paths)+1,audit_sha256=dev.sha(STAGE/'final_audit.json'))
    save(STAGE/'archive_meta.json',meta); print(json.dumps(meta))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','smoke_initial','smoke_restart','archive'))
    args=parser.parse_args()
    if Path(__file__).resolve().parent!=STAGE: raise ValueError('Isolated E129 stage required')
    if args.mode.startswith('smoke_'): run_smoke(args.mode)
    else: {'prepare':prepare,'archive':archive}[args.mode]()
