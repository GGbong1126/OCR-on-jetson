"""One finite own-project root diagnostic service; never shadow stdlib profile."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from settings import B,S,SOURCE,PY,TOOL,e167 as c


def wait_until(test,seconds):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        try:
            value=test()
            if value: return value
        except (OSError,ValueError): pass
        time.sleep(.1)
    raise TimeoutError('Diagnostic startup/gate deadline; retain evidence')


def main():
    assert os.geteuid()==0
    folder=S/'trace'; folder.mkdir(mode=0o700)
    runtime=folder/'runtime'; runtime.mkdir(mode=0o700)
    with socket.socket() as sock: assert sock.connect_ex(('127.0.0.1',5052))!=0
    mem=dict(l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines())
    assert int(mem['MemAvailable'].split()[0])>=8*1024**2
    env=dict(os.environ,HOME=str(folder/'home'),XDG_CACHE_HOME=str(folder/'cache'),
        CUDA_CACHE_PATH=str(folder/'cuda_cache'),TMPDIR=str(folder/'tmp'),
        LD_LIBRARY_PATH=str(TOOL/'targets/aarch64-linux/lib')+':'+str(B/'lib/cuda10.2')+':/usr/local/cuda-10.2/targets/aarch64-linux/lib',
        PYTHONPATH=str(S)+':'+str(SOURCE/'app')+':'+str(SOURCE/'ipc'),
        AUTOOCR_ENABLE_DEV_NO_AUTH='1',AUTOOCR_REC_THREADS='1',AUTOOCR_REC_SLOTS='4',
        AUTOOCR_BATCH_DET='1',AUTOOCR_BATCH_JPEG='1',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    for key in ('CUDA_LAUNCH_BLOCKING','CUDA_DEVICE_MAX_CONNECTIONS','PYTHONHOME','GUNICORN_CMD_ARGS'): env.pop(key,None)
    for n in ('home','cache','cuda_cache','tmp'): (folder/n).mkdir(mode=0o700)
    result=dict(status='starting',closed=False,extract_requests=0,system_changed=False)
    backend=cpp=http=None; logs=[]; safe=False
    def save(): (folder/'profile.json').write_text(json.dumps(result,indent=2)+'\n')
    def launch(cmd,label):
        stream=(folder/(label+'.log')).open('xb'); logs.append(stream)
        return subprocess.Popen(cmd,env=env,cwd=SOURCE/'app',stdin=subprocess.DEVNULL,
              stdout=stream,stderr=stream,start_new_session=True)
    save()
    try:
        backend=launch([str(TOOL/'bin/nvprof'),'--profile-from-start','off','--concurrent-kernels','on',
           '--print-gpu-trace','--csv','--normalized-time-unit','us','--log-file',str(folder/'gpu.csv'),
           str(PY),'-u','-B',str(S/'backend_trace.py'),'--runtime',str(runtime)],'backend')
        def ready():
            if backend.poll() is not None: raise RuntimeError('Profile backend exited; inspect log')
            p=runtime/'status.json'
            r=json.loads(p.read_text()) if p.exists() else {}
            if r.get('status') in ('error','quarantined','stopped'): raise RuntimeError('Profile backend failed')
            return r if r.get('status')=='ready' else None
        marker=wait_until(ready,260); result['backend_pid']=marker['pid']; save()
        cpp=launch([str(SOURCE/'app/autoocr_batch_app'),str(runtime/'backend.sock'),str(runtime/'front.sock'),
                     '--rec-slots=4','--det-batch','--jpeg'],'cpp')
        wait_until(lambda:(runtime/'front.sock').exists(),10)
        (folder/'server.json').write_text(json.dumps(dict(backend_pid=marker['pid'],cpp_pid=cpp.pid)))
        http=launch([str(B/'envs/business-py310/bin/gunicorn'),'--config','/dev/null','--chdir',str(SOURCE/'app'),
            '--bind','127.0.0.1:5052','--workers','1','--worker-class','gthread','--threads','4',
            '--timeout','600','--worker-connections','8','--graceful-timeout','40','--backlog','4',
            '--access-logfile','/dev/null','http_entry:create_app()'],'http')
        wait_until(lambda:c.health()['status']=='ready',30)
        rows=json.loads((c.S/'inventory_r1.json').read_text())['selected']
        golden=json.loads((c.S/'reference_hashes.json').read_text())
        result.update(status='warming',http_pid=http.pid,cpp_pid=cpp.pid); save()
        result['warmup']=c.cycle(rows,golden); save()
        def gate(command,phase):
            c.idle(); p=folder/'command.json'; q=p.with_suffix('.new')
            q.write_text(json.dumps(dict(command=command))); os.replace(q,p)
            wait_until(lambda:json.loads((folder/'gate.json').read_text())['phase']==phase,10)
        gate('start','on'); result['status']='profiling_real_http'; save()
        result['measured']=c.cycle(rows,golden); result['final_health']=c.idle(); save()
        gate('stop','stopped'); result['status']='sample_completed'; save()
    except Exception as error:
        result.update(status='failed',error=repr(error)); save(); raise
    finally:
        # Stop only these own idle project children; never kill/reset an unknown GPU.
        try:
            if http is not None and http.poll() is None:
                http.terminate(); result['http_exit']=http.wait(timeout=60)
            if cpp is not None and cpp.poll() is None:
                cpp.terminate(); result['cpp_exit']=cpp.wait(timeout=20)
            if backend is not None:
                p=runtime/'status.json'
                marker=json.loads(p.read_text()) if p.exists() else {}
                pid=marker.get('pid')
                if pid and c.alive(pid): os.kill(pid,signal.SIGTERM)
                result['profiler_exit']=backend.wait(timeout=180)
                p=folder/'backend_exit.json'; exit_report=json.loads(p.read_text()) if p.exists() else {}
                result['backend_exit']=exit_report
                safe=exit_report.get('status')=='stopped' and exit_report.get('close_error') is False
                if not pid and backend.returncode is not None: safe=True
            else: safe=True
        finally:
            result['closed']=safe
            if safe and result['status']=='sample_completed': result['status']='real_http_trace_closed'
            save()
            for log in logs: log.close()
            for p in folder.rglob('*'):
                assert p.resolve().is_relative_to(folder)
                if not p.is_symlink(): os.chown(p,1000,1000)
            os.chown(folder,1000,1000)
    assert result['status']=='real_http_trace_closed' and result['profiler_exit']==0,result
    print(json.dumps(dict(status=result['status'],pages=result['measured']['pages'])),flush=True)


if __name__=='__main__': main()
