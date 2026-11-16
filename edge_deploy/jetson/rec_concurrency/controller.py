"""Freeze identities, normal stop, finite REC tests, always safe E162 restore."""
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler, Request

B=Path('/home/xchq/work/autoocr'); S=B/'staging/rec_concurrency_e166_r1'
STABLE=B/'overlays/batch-ocr-e161-r1'; SOURCE=B/'overlays/batch-ocr-plan2-e164-r1'
PYTHON=B/'envs/business-py310/bin/python'; OUT=S/'control_r1.json'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def alive(pid):
    try: return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError: return False
def units():
    return {u:subprocess.check_output(['systemctl','show',u,'-p','ActiveState','-p','SubState','-p','MainPID','-p','UnitFileState'],text=True).splitlines()
            for u in ('autoocr-jetson-dev.service','gdm.service','ssh.service','frpc-jetson.service')}
def applications():
    result=[]
    for line in subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines():
        pid,command=line.strip().split(None,1); words=command.split()
        if any(w.endswith(('/serve_backend.py','/autoocr_batch_app','/probe.py','/threads.py','/positive.py')) for w in words[:4]):
            result.append(dict(pid=int(pid),command=command))
    return result


def main(followup=False):
    output_control=S/'control_threads_r1.json' if followup else OUT
    assert not output_control.exists()
    state=dict(status='running',phase='preflight',restored=False,arms=[])
    def save(): output_control.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    opener=build_opener(ProxyHandler({}))
    def health():
        with opener.open('http://127.0.0.1:5052/healthz',timeout=8) as r: return json.load(r)
    frozen=json.loads((B/'staging/multistream_e163_r1/preflight_r2.json').read_text())
    source=json.loads((SOURCE/'installation.json').read_text())['files_sha256']
    def fingerprints():
        for root,key in ((B/'releases/jetson-r2','originals'),(B/'overlays/rec-wide-e124-r1','wide'),(STABLE,'stable_source_sha256')):
            assert all(sha(root/n)==v for n,v in frozen[key].items())
        assert sha(B/'staging/det_batch_e153_r1/det_batch2.engine')==frozen['n2_sha256']
        assert all(sha(SOURCE/n)==v for n,v in source.items())
    fingerprints()
    initial=dict(boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),units=units())
    env=dict(os.environ,LD_LIBRARY_PATH=str(B/'lib/cuda10.2')+':/usr/local/cuda-10.2/targets/aarch64-linux/lib',
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONNOUSERSITE='1')
    for key in ('CUDA_LAUNCH_BLOCKING','CUDA_DEVICE_MAX_CONNECTIONS','AUTOOCR_CUPTI_LIBRARY'): env.pop(key,None)
    continuing=followup and not applications()
    state.update(preflight=initial,scripts_sha256={p.name:sha(p) for p in S.glob('*.py')})
    if continuing:
        prior=json.loads(OUT.read_text())
        assert prior['phase']=='unsafe_exit_retain_resources' and initial==prior['preflight']
        assert all(not alive(pid) for pid in prior['owned_pids'])
        assert 'Application received signal 139' in (S/'timeline_r1/graph_w32.csv').read_text()
        # nvprof's child has exited, not quarantined. Check driver reclamation and a new empty context, no reset.
        mem=dict(l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines())
        assert int(mem['NvMapMemUsed'].split()[0])<1024**2
        code="import ctypes as c,json; x=c.CDLL('libcudart.so.10.2'); x.cudaSetDevice.argtypes=[c.c_int]; a=x.cudaSetDevice(0); b=x.cudaDeviceSynchronize(); print(json.dumps([a,b])); assert a==b==0"
        result=subprocess.run([str(PYTHON),'-B','-c',code],env=env,capture_output=True,text=True)
        assert result.returncode==0 and json.loads(result.stdout)==[0,0]
        state.update(continued_after_profiler_child_exit=True,empty_context_cuda_codes=[0,0],
                     original_controller_status=prior['status'],old_master_pid=prior['old_master_pid'])
    else:
        h=health(); assert h['status']=='ready' and not h['busy'] and h['runtime']['session']['images']==0
        masters=[]
        for proc in Path('/proc').iterdir():
            if not proc.name.isdecimal(): continue
            try:
                if os.readlink(proc/'cwd')==str(STABLE/'app') and b'gunicorn' in (proc/'cmdline').read_bytes(): masters.append(int(proc.name))
            except (FileNotFoundError,PermissionError): pass
        parents=dict(tuple(map(int,l.split())) for l in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines())
        roots=[p for p in masters if parents[p] not in masters]; assert len(roots)==1; pid=roots[0]
        raw=(Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
        assert b'AUTOOCR_BATCH_DET=1' in raw and b'AUTOOCR_BATCH_JPEG=1' in raw
        assert not any(x.startswith(b'AUTOOCR_REC_SLOTS=') for x in raw)
        owned={pid}
        while True:
            new=owned|{p for p,pp in parents.items() if pp in owned}
            if new==owned: break
            owned=new
        assert len(owned)==4 and len(applications())==2
        state.update(health_before=h,old_master_pid=pid,owned_pids=sorted(owned))
    assert subprocess.run(['sudo','-n','true'],stdin=subprocess.DEVNULL).returncode==0,'New authorized profiler credential precheck'
    save()
    result=subprocess.run([str(PYTHON),'-B','-m','unittest','discover','-s',str(S),'-p','test_*.py','-v'],cwd=S)
    assert result.returncode==0
    if not continuing:
        assert not health()['busy'] and os.readlink(Path('/proc')/str(pid)/'cwd')==str(STABLE/'app')
        state['phase']='normal_stop_e162'; save(); started=time.monotonic(); os.kill(pid,signal.SIGTERM)
        while any(alive(p) for p in owned) and time.monotonic()-started<60: time.sleep(.1)
        assert not any(alive(p) for p in owned) and not applications(),'No new GPU beside unknown old processes'
        state['normal_stop_seconds']=time.monotonic()-started
    state['phase']='ordinary_rec_matrix'; save()
    safe=True; error=None
    try:
        order=([('threads',w) for w in (32,160,1287,2081)] if followup else
               [('slot',160)]+[(m,w) for m in ('resident','graph') for w in (32,160,1287,2081)])
        for mode,width in order:
            assert not applications()
            output=S/('perf_%s_w%d.json'%(mode,width))
            with output.with_suffix('.log').open('xb') as log:
                command=([str(PYTHON),'-u','-B',str(S/'threads.py'),'--width',str(width),'--output',str(output)] if followup else
                         [str(PYTHON),'-u','-B',str(S/'probe.py'),'--mode',mode,'--width',str(width),'--slots','16','--output',str(output)])
                child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
                state['child_pid']=child.pid; save(); safe=False; code=child.wait()
            r=json.loads(output.read_text()) if output.exists() else {}
            safe=r.get('closed') is True and r.get('safe_quiescence_confirmed') is True and not applications()
            state['arms'].append(dict(mode=mode,width=width,returncode=code,status=r.get('status'),closed=r.get('closed'))); save()
            assert safe,'Unknown candidate safety; retain resources'
            if mode!='graph': assert code==0 and r['status']==('threaded_rec_outputs_consistent' if followup else 'rec_outputs_consistent')
            print('Ordinary',mode,width,code,flush=True)
        state['phase']='own_process_gpu_timeline'; save(); safe=False
        with (S/('profile_threads.log' if followup else 'profile.log')).open('xb') as log:
            code=subprocess.call(['sudo','-n',str(PYTHON),'-u','-B',str(S/('profile_threads.py' if followup else 'profile.py'))],stdin=subprocess.DEVNULL,stdout=log,stderr=log)
        folder=S/('timeline_threads_r1' if followup else 'timeline_r1')
        pr=json.loads((folder/'profile.json').read_text()) if (folder/'profile.json').exists() else {}
        safe=not applications() and all(r.get('closed') for r in pr.get('rows',[]))
        state['profile_returncode']=code; save()
        assert safe and code==0 and pr.get('status')=='finite_profile_matrix_finished'
        from analyze import analyze
        analyses=[]
        for row in pr['rows']:
            if 'output' not in row: continue
            path=Path(row['output']); r=json.loads(path.read_text())
            if r['status'] not in ('rec_outputs_consistent','threaded_rec_outputs_consistent','positive_control_completed_NOT_REC'):
                analyses.append(dict(mode=row['mode'],width=row['width'],status='failed_preserved')); continue
            assert r['trace']['stopped']
            summary=analyze(path.with_suffix('.csv').read_text(),r['trace'])
            analyses.append(dict(mode=row['mode'],width=row['width'],**summary))
        (S/('analysis_threads_r1.json' if followup else 'analysis_r1.json')).write_text(json.dumps(analyses,indent=2)+'\n')
        state['max_observed_rec_kernel_streams']=max(r.get('max_rec_streams_over_threshold',0) for r in analyses if r['mode']!='positive_control_NOT_REC')
    except Exception as failure:
        error=repr(failure); state['error']=error
    finally:
        state['phase']='restore_e162' if safe else 'unsafe_exit_retain_resources'; save()
        if safe:
            assert not applications(); fingerprints()
            assert initial==dict(boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),units=units())
            for key in ('AUTOOCR_REC_SLOTS','AUTOOCR_BATCH_DET','AUTOOCR_BATCH_JPEG','AUTOOCR_BATCH_RUNTIME_BASE'): env.pop(key,None)
            with (S/('restored_threads_e162.log' if followup else 'restored_e162.log')).open('xb') as log:
                master=subprocess.Popen(['bash',str(STABLE/'app/start_batch.sh')],env=env,cwd=STABLE/'app',
                                        stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            state['restored_master_pid']=master.pid; save(); end=time.monotonic()+250
            while time.monotonic()<end and master.poll() is None:
                try:
                    h=health()
                    if h['status']=='ready' and not h['busy']: break
                except (OSError,ValueError): pass
                time.sleep(.2)
            else: raise RuntimeError('E162 restore failed')
            raw=(Path('/proc')/str(master.pid)/'environ').read_bytes().split(b'\0')
            assert b'AUTOOCR_BATCH_DET=1' in raw and b'AUTOOCR_BATCH_JPEG=1' in raw and not any(v.startswith(b'AUTOOCR_REC_SLOTS=') for v in raw)
            golden=json.loads((B/'staging/multistream_e164_r1/bench_s0_r18.json').read_text())
            golden={(r['image_sha256'],r['image_type']):r['candidate_sha256'] for r in golden['rows']}
            plan=json.loads((B/'staging/det_batch_e153_r1/plan.json').read_text()); images=[]; keys=[]
            for row in plan['rows'][:2]:
                data=(Path(plan['images'])/row['file']).read_bytes(); assert hashlib.sha256(data).hexdigest()==row['id']
                for kind in (3,1):
                    images.append(dict(imageId='image'+str(len(images)),imageType=kind,imageBase64=base64.b64encode(data).decode()))
                    keys.append((row['id'],kind))
            req=Request('http://127.0.0.1:5052/OCRBatch',data=json.dumps(dict(images=images)).encode(),headers={'Content-Type':'application/json'})
            with opener.open(req,timeout=320) as response: value=json.load(response)
            assert value['status']=='ok' and len(value['results'])==4
            checks=[]
            for item,key in zip(value['results'],keys):
                digest=hashlib.sha256(json.dumps(item['data'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
                assert item['status']=='ok' and digest==golden[key]
                checks.append(dict(image_sha256=key[0],image_type=key[1],public_sha256=digest,equal=True))
            for _ in range(60):
                h=health()
                if not h['busy'] and h['runtime']['session']['images']==h['runtime']['session']['registered_bytes']==0: break
                time.sleep(.1)
            assert h['status']=='ready' and not h['busy'] and h['runtime']['session']['images']==h['runtime']['session']['registered_bytes']==0
            fingerprints(); assert units()==initial['units']
            state.update(restored=True,restored_checks=checks,restored_health=h,phase='complete',
                         status='rec_matrix_completed_e162_restored' if error is None else 'test_failed_e162_restored')
            save()
    assert error is None and state['restored']


if __name__=='__main__':
    assert sys.argv[1:] in ([],['--followup'])
    main(followup=bool(sys.argv[1:]))
