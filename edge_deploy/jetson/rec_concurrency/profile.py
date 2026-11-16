"""New E166 authorization: sudo only this finite own-project nvprof matrix."""
import json
import os
from pathlib import Path
import subprocess
import sys

B=Path('/home/xchq/work/autoocr'); S=B/'staging/rec_concurrency_e166_r1'
TOOL=B/'staging/multistream_e164_r1/profiler/extracted/usr/local/cuda-10.2'


def main():
    assert os.geteuid()==0 and len(sys.argv)==1
    folder=S/'timeline_r1'; folder.mkdir(mode=0o700)
    assert folder.resolve().is_relative_to(S.resolve())
    env=dict(os.environ,HOME=str(folder/'home'),XDG_CACHE_HOME=str(folder/'cache'),
             CUDA_CACHE_PATH=str(folder/'cuda_cache'),TMPDIR=str(folder/'tmp'),
             LD_LIBRARY_PATH=str(TOOL/'targets/aarch64-linux/lib')+':'+str(B/'lib/cuda10.2')+':/usr/local/cuda-10.2/targets/aarch64-linux/lib',
             AUTOOCR_CUPTI_LIBRARY=str(TOOL/'targets/aarch64-linux/lib/libcupti.so.10.2.175'),
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONNOUSERSITE='1')
    for key in ('CUDA_LAUNCH_BLOCKING','CUDA_DEVICE_MAX_CONNECTIONS'): env.pop(key,None)
    for name in ('home','cache','cuda_cache','tmp'): (folder/name).mkdir(mode=0o700)
    result=dict(status='running',rows=[],new_e166_authorization=True,system_configuration_changed=False)
    try:
        # Same slots for all modes; fixed synthetic shapes, not an overall throughput contest.
        for mode,width in [('slot',160),('resident',32),('resident',160),('resident',1287),('resident',2081),
                           ('graph',32),('graph',160),('graph',1287),('graph',2081)]:
            name='%s_w%d'%(mode,width); output=folder/(name+'.json')
            perf=S/('perf_'+name+'.json')
            prior=json.loads(perf.read_text())
            if prior['status']!='rec_outputs_consistent':
                result['rows'].append(dict(mode=mode,width=width,skipped='ordinary mode failed, preserved',closed=prior['closed']))
                continue
            assert not output.exists()
            cmd=[str(TOOL/'bin/nvprof'),'--profile-from-start','off','--concurrent-kernels','on',
                 '--print-gpu-trace','--csv','--normalized-time-unit','us','--log-file',str(output.with_suffix('.csv')),
                 str(B/'envs/business-py310/bin/python'),'-u','-B',str(S/'probe.py'),
                 '--mode',mode,'--width',str(width),'--slots','16','--trace','--output',str(output)]
            with output.with_suffix('.log').open('xb') as log:
                code=subprocess.call(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
            r=json.loads(output.read_text()) if output.exists() else {}
            row=dict(mode=mode,width=width,returncode=code,status=r.get('status'),closed=r.get('closed'),output=str(output))
            result['rows'].append(row)
            (folder/'profile.json').write_text(json.dumps(result,indent=2))
            print('Profile',name,code,r.get('status'),flush=True)
            assert r.get('closed') and r.get('safe_quiescence_confirmed'),'Unsafe child, no next GPU process'
        result['status']='finite_profile_matrix_finished'
    finally:
        (folder/'profile.json').write_text(json.dumps(result,indent=2))
        # Only fresh project-owned subtree; never touch symlink targets or original sources.
        for p in folder.rglob('*'):
            assert p.resolve().is_relative_to(folder)
            if not p.is_symlink(): os.chown(p,1000,1000)
        os.chown(folder,1000,1000)


if __name__=='__main__': main()
