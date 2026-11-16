"""Finite actual-HTTP comparison, select owner count, deploy isolated OCR only."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from urllib.request import build_opener,ProxyHandler,Request

B=Path('/home/xchq/work/autoocr'); S=B/'staging/threaded_e167_r1'
OLD=B/'overlays/batch-ocr-e161-r1'; PREVIOUS=B/'overlays/batch-ocr-plan2-e164-r1'
NEW=B/'overlays/batch-ocr-threaded-e167-r1'; PY=B/'envs/business-py310/bin/python'
OUT=S/'control_r1.json'; OPENER=build_opener(ProxyHandler({}))


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def health():
    with OPENER.open('http://127.0.0.1:5052/healthz',timeout=10) as response: return json.load(response)
def alive(pid):
    try: return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError: return False
def units():
    return {u:subprocess.check_output(['systemctl','show',u,'-p','MainPID','-p','ActiveState','-p','UnitFileState'],text=True).splitlines()
            for u in ('autoocr-jetson-dev.service','gdm.service','ssh.service','frpc-jetson.service')}
def tree(pid):
    parents=dict(tuple(map(int,l.split())) for l in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines())
    found={pid}
    while True:
        new=found|{p for p,pp in parents.items() if pp in found}
        if new==found: return sorted(found)
        found=new
def snapshot(pid):
    mem=dict(l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines())
    rows={}
    for p in tree(pid):
        try:
            path=Path('/proc')/str(p); status=dict(l.split(':',1) for l in (path/'status').read_text().splitlines())
            rows[str(p)]=dict(rss_kib=int(status.get('VmRSS','0 kB').split()[0]),threads=int(status['Threads']),fd=len(list((path/'fd').iterdir())))
        except FileNotFoundError:
            if p==pid: raise
    return dict(memory_kib={k:int(mem[k].split()[0]) for k in ('MemAvailable','NvMapMemUsed','SwapFree')},processes=rows)
def idle():
    for _ in range(100):
        h=health(); session=h['runtime']['session']
        if not h['busy'] and session['images']==session['registered_bytes']==session['active_reserved']==0 and h['front_reservation_bytes']==0:
            assert h['status']=='ready' and h['models']['ner']['calls']==1
            return h
        time.sleep(.1)
    raise RuntimeError('Service did not return all credits')
def stop(pid,folder):
    assert idle()['status']=='ready'
    assert os.readlink(Path('/proc')/str(pid)/'cwd')==str(folder/'app')
    owned=tree(pid); assert len(owned)==4
    os.kill(pid,signal.SIGTERM); end=time.monotonic()+75
    while any(alive(p) for p in owned) and time.monotonic()<end: time.sleep(.1)
    assert not any(alive(p) for p in owned),'Unknown GPU cleanup: no next model set'
    mem=snapshot(os.getpid())['memory_kib']; assert mem['MemAvailable']>8*1024**2
def start(folder,slots,label):
    env=dict(os.environ)
    for k in ('AUTOOCR_REC_SLOTS','AUTOOCR_REC_THREADS','AUTOOCR_BATCH_DET','AUTOOCR_BATCH_JPEG','AUTOOCR_BATCH_RUNTIME_BASE'): env.pop(k,None)
    if slots: env.update(AUTOOCR_REC_SLOTS=str(slots),AUTOOCR_REC_THREADS='1')
    with (S/(label+'.log')).open('xb') as log:
        child=subprocess.Popen(['bash',str(folder/'app/start_batch.sh')],env=env,cwd=folder/'app',stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    end=time.monotonic()+260
    while time.monotonic()<end and child.poll() is None:
        try:
            h=health()
            if h['status']=='ready' and not h['busy']: return child.pid
        except (OSError,ValueError): pass
        time.sleep(.2)
    raise RuntimeError('Startup failed; inspect retained runtime, do not kill GPU')


def inventory():
    from PIL import Image
    files=sorted(p for p in (B/'testpic').iterdir() if p.is_file() and p.suffix.lower()=='.jpg'); rows=[]; invalid=[]
    for p in files:
        try:
            with Image.open(p) as im: w,h=im.size; fmt=im.format
            rows.append(dict(file=p.name,width=w,height=h,bytes=p.stat().st_size,format=fmt))
        except Exception as e: invalid.append(dict(file=p.name,error=type(e).__name__))
    original=json.loads((B/'staging/det_batch_e153_r1/plan.json').read_text())['rows']
    assert all(sha(B/'testpic'/r['file'])==r['id'] for r in original)
    chosen=[r['file'] for r in original]
    extra=[r for r in rows if r['file'] not in chosen and r['bytes']<2*1024**2 and max(r['width'],r['height'])<=10000
           and r['width']*r['height']<=4000000]
    # Three new aspect/area representatives, no text inspection or selection by performance.
    for key in (lambda r:r['width']/r['height'],lambda r:r['height']/r['width'],lambda r:r['width']*r['height']):
        candidate=max((r for r in extra if r['file'] not in chosen),key=key)
        chosen.append(candidate['file'])
    selected=[]
    for name in chosen:
        p=B/'testpic'/name; r=next(r for r in rows if r['file']==name)
        selected.append(dict(r,image_sha256=sha(p)))
    report=dict(total=len(files),header_valid=len(rows),invalid=invalid,headers=rows,selected=selected)
    (S/'inventory_r1.json').write_text(json.dumps(report,indent=2)+'\n')
    return selected


def request(pages,golden,learn=False):
    begin=time.monotonic(); images=[]
    for i,(row,kind) in enumerate(pages):
        payload=(B/'testpic'/row['file']).read_bytes(); assert hashlib.sha256(payload).hexdigest()==row['image_sha256']
        images.append(dict(imageId='image'+str(i),imageType=kind,imageBase64=base64.b64encode(payload).decode()))
    wire=json.dumps(dict(images=images)).encode(); prepared=time.monotonic()
    req=Request('http://127.0.0.1:5052/OCRBatch',data=wire,headers={'Content-Type':'application/json'})
    with OPENER.open(req,timeout=330) as response: result=json.load(response)
    elapsed=time.monotonic()-begin
    assert result['status']=='ok' and len(result['results'])==len(pages)
    records=[]
    for index,((row,kind),value) in enumerate(zip(pages,result['results'])):
        assert value['status']=='ok' and value['image_id']=='image'+str(index)
        key=row['image_sha256']+':'+str(kind)
        digest=hashlib.sha256(json.dumps(value['data'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if learn: golden[key]=digest
        assert golden[key]==digest,('Public OCR output drift',row['file'],kind)
        records.append(dict(image_sha256=row['image_sha256'],image_type=kind,public_sha256=digest,equal=True))
    h=idle()
    for model in h['models'].values(): assert model.get('errors',0)==0
    return dict(seconds=elapsed,prepare_seconds=prepared-begin,pages=len(pages),records=records,execution=result.get('execution',{}))


def cycle(rows,golden,reverse=False,learn=False):
    pages=[(row,kind) for kind in (3,1) for row in rows]
    if reverse: pages.reverse()
    groups=[]
    for i in range(0,len(pages),4): groups.append(request(pages[i:i+4],golden,learn))
    return dict(seconds=sum(r['seconds'] for r in groups),pages=len(pages),requests=groups)


def main(resume=False):
    assert resume or (not OUT.exists() and not NEW.exists())
    state=dict(status='running',phase='preflight',arms=[],selected=None); pid=None; folder=None; safe=True
    def save(): OUT.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    frozen=json.loads((B/'staging/multistream_e163_r1/preflight_r2.json').read_text())
    initial=dict(boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),units=units())
    if resume:
        prior=json.loads(OUT.read_text())
        assert prior['status']=='failed' and len(prior['arms'])==1 and prior['preflight']==initial
        assert not alive(prior['old_master_pid']) and not any(alive(p) for p in tree(prior['old_master_pid']))
        from socket import socket
        with socket() as sock: assert sock.connect_ex(('127.0.0.1',5052))!=0
        assert snapshot(os.getpid())['memory_kib']['MemAvailable']>8*1024**2
        state.update(prior); state.pop('error',None); state.update(status='running',phase='resume_after_observer_race')
    base_hashes=json.loads((PREVIOUS/'installation.json').read_text())['files_sha256']
    assert all(sha(PREVIOUS/n)==h for n,h in base_hashes.items())
    def originals():
        for root,key in ((B/'releases/jetson-r2','originals'),(B/'overlays/rec-wide-e124-r1','wide'),(OLD,'stable_source_sha256')):
            assert all(sha(root/n)==v for n,v in frozen[key].items())
        assert all(sha(PREVIOUS/n)==h for n,h in base_hashes.items())
        assert initial==dict(boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),units=units())
    originals()
    if resume:
        rows=json.loads((S/'inventory_r1.json').read_text())['selected']
        assert all(sha(B/'testpic'/r['file'])==r['image_sha256'] for r in rows)
        installation=json.loads((NEW/'installation.json').read_text())
        assert all(sha(NEW/n)==v for n,v in installation['files_sha256'].items())
    else:
        rows=inventory(); shutil.copytree(PREVIOUS,NEW)
        for name in ('app/serve_backend.py','multistream/threaded_rec.py','multistream/test_threaded_rec.py'):
            shutil.copy2(S/name,NEW/name)
        shutil.copy2(S/'start_threaded.sh',NEW/'start_threaded.sh')
        installation=dict(base=str(PREVIOUS),base_files_sha256=base_hashes,
            files_sha256={str(p.relative_to(NEW)):sha(p) for p in NEW.rglob('*') if p.is_file() and p.name!='installation.json'})
        (NEW/'installation.json').write_text(json.dumps(installation,indent=2)+'\n')
    env=dict(os.environ,AUTOOCR_THREAD_SOURCE=str(NEW),AUTOOCR_BATCH_APP=str(NEW/'app/autoocr_batch_app'))
    for cmd in ([str(PY),'-B','-m','unittest','discover','-s',str(NEW/'multistream'),'-p','test_threaded_rec.py','-v'],
                [str(PY),'-B',str(S/'check_pipeline.py')]):
        assert subprocess.run(cmd,env=env).returncode==0,'CPU/real IPC preflight failed; stable service untouched'
    masters=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal(): continue
        try:
            if os.readlink(p/'cwd')==str(OLD/'app') and b'gunicorn' in (p/'cmdline').read_bytes(): masters.append(int(p.name))
        except (OSError,PermissionError): pass
    parent=dict(tuple(map(int,l.split())) for l in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines())
    roots=[p for p in masters if parent[p] not in masters]
    if not resume:
        assert len(roots)==1; pid=roots[0]; folder=OLD
    else: assert not roots
    if not resume:
        state.update(preflight=initial,old_master_pid=pid,source_sha256=installation['files_sha256'],test_count=len(rows),
                     scripts_sha256={p.name:sha(p) for p in S.glob('*.py')})
    else:
        state['resumed_scripts_sha256']={p.name:sha(p) for p in S.glob('*.py')}
    save()
    golden=json.loads((S/'reference_hashes.json').read_text()) if resume else {}
    historical=json.loads((B/'staging/multistream_e164_r1/bench_s0_r18.json').read_text())
    historic={r['image_sha256']+':'+str(r['image_type']):r['candidate_sha256'] for r in historical['rows']}
    try:
        for index,slots in enumerate((0,1,2,4,0)):
            if index<len(state['arms']): continue
            if index:
                if pid is not None:
                    safe=False; stop(pid,folder); pid=None; safe=True
                folder=NEW if slots else OLD; safe=False
                pid=start(folder,slots,'run_'+str(index)); safe=True
            state['phase']='http_arm_'+str(index); state['current_master_pid']=pid; save()
            warm=cycle(rows,golden,learn=(index==0))
            if index==0:
                assert all(golden[k]==v for k,v in historic.items() if k in golden)
                (S/'reference_hashes.json').write_text(json.dumps(golden,indent=2)+'\n')
            before=snapshot(pid); passes=[]
            for repeat in range(3):
                passes.append(cycle(rows,golden,reverse=(repeat==1)))
            after=snapshot(pid); h=idle()
            b=before['processes']; a=after['processes']; assert set(a)==set(b)
            gates=dict(no_fd_growth=all(a[p]['fd']<=b[p]['fd']+4 for p in a),
                       no_thread_growth=all(a[p]['threads']<=b[p]['threads']+2 for p in a),
                       warm_rss_growth=sum(a[p]['rss_kib'] for p in a)<=sum(b[p]['rss_kib'] for p in b)+128*1024,
                       memory_reserve=after['memory_kib']['MemAvailable']>=1024**2,
                       swap_growth=before['memory_kib']['SwapFree']-after['memory_kib']['SwapFree']<=128*1024)
            assert all(gates.values()),gates
            if slots:
                t=h['models']['rec']['threaded']; assert t['owners']==t['alive']==slots and t['calls']>0 and t['errors']==0 and t['context_shared']
            arm=dict(index=index,slots=slots,pid=pid,warmup=warm,passes=passes,resource_before=before,resource_after=after,gates=gates,
                     final_health=h,pages_per_second=sum(p['pages'] for p in passes)/sum(p['seconds'] for p in passes))
            (S/('arm_%d.json'%index)).write_text(json.dumps(arm,indent=2)+'\n')
            state['arms'].append(dict(index=index,slots=slots,pages_per_second=arm['pages_per_second'],gates=gates)); save()
            print('Completed HTTP arm',index,slots,arm['pages_per_second'],flush=True)
        candidates=[a for a in state['arms'] if a['slots']]
        best=max(a['pages_per_second'] for a in candidates)
        selected=min(a['slots'] for a in candidates if a['pages_per_second']>=.98*best)
        state['selected']=selected; state['phase']='deploy_selected'; save()
        safe=False; stop(pid,folder); pid=None; safe=True
        (NEW/'selection.json').write_text(json.dumps(dict(selected_rec_threads=selected,rule='smallest within 2% of fastest valid end-to-end HTTP candidate',arms=state['arms']),indent=2)+'\n')
        folder=NEW; safe=False; pid=start(folder,selected,'selected_start'); safe=True
        final=cycle(rows,golden); h=idle(); originals()
        state.update(status='selected_threaded_ocr_deployed',phase='complete',selected_master_pid=pid,final_checks=final,final_health=h,
                     final_snapshot=snapshot(pid),originals_unchanged=True)
        save()
    except Exception as error:
        state.update(error=repr(error),status='failed',phase='safe_recovery' if safe else 'unknown_cleanup_retain'); save()
        if safe:
            if pid is not None: stop(pid,folder)
            pid=start(OLD,0,'failure_restore'); state.update(restored_master_pid=pid,restored_health=idle()); save()
        raise


if __name__=='__main__':
    assert sys.argv[1:] in ([],['--resume'])
    main(resume=bool(sys.argv[1:]))
