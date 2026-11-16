"""Finite E166 follow-up own-process timeline: threaded REC plus positive control."""
import json
import os
from pathlib import Path
import subprocess
import sys

from profile import B,S,TOOL


def main():
    assert os.geteuid()==0 and len(sys.argv)==1
    folder=S/'timeline_threads_r1'; folder.mkdir(mode=0o700)
    env=dict(os.environ,HOME=str(folder/'home'),XDG_CACHE_HOME=str(folder/'cache'),
             CUDA_CACHE_PATH=str(folder/'cuda_cache'),TMPDIR=str(folder/'tmp'),
             LD_LIBRARY_PATH=str(TOOL/'targets/aarch64-linux/lib')+':'+str(B/'lib/cuda10.2')+':/usr/local/cuda-10.2/targets/aarch64-linux/lib',
             AUTOOCR_CUPTI_LIBRARY=str(TOOL/'targets/aarch64-linux/lib/libcupti.so.10.2.175'),
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONNOUSERSITE='1')
    for key in ('CUDA_LAUNCH_BLOCKING','CUDA_DEVICE_MAX_CONNECTIONS'): env.pop(key,None)
    for name in ('home','cache','cuda_cache','tmp'): (folder/name).mkdir(mode=0o700)
    result=dict(status='running',rows=[],new_e166_authorization=True,system_configuration_changed=False)
    try:
        for width in (None,32,160,1287,2081):
            name='positive_control_NOT_REC' if width is None else 'threads_w'+str(width)
            output=folder/(name+'.json')
            script=S/('positive.py' if width is None else 'threads.py')
            if width is not None:
                prior=json.loads((S/('perf_threads_w'+str(width)+'.json')).read_text())
                assert prior['status']=='threaded_rec_outputs_consistent' and prior['closed']
            cmd=[str(TOOL/'bin/nvprof'),'--profile-from-start','off','--concurrent-kernels','on',
                 '--print-gpu-trace','--csv','--normalized-time-unit','us','--log-file',str(output.with_suffix('.csv')),
                 str(B/'envs/business-py310/bin/python'),'-u','-B',str(script),'--trace','--output',str(output)]
            if width is not None: cmd.extend(('--width',str(width)))
            with output.with_suffix('.log').open('xb') as log:
                code=subprocess.call(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
            r=json.loads(output.read_text()) if output.exists() else {}
            result['rows'].append(dict(mode='positive_control_NOT_REC' if width is None else 'threads',width=width,
                                      returncode=code,status=r.get('status'),closed=r.get('closed'),output=str(output)))
            (folder/'profile.json').write_text(json.dumps(result,indent=2))
            assert r.get('closed') and r.get('safe_quiescence_confirmed'),'Unknown GPU safety, no next process'
            print('Profile',name,code,r.get('status'),flush=True)
        result['status']='finite_profile_matrix_finished'
    finally:
        (folder/'profile.json').write_text(json.dumps(result,indent=2))
        for p in folder.rglob('*'):
            assert p.resolve().is_relative_to(folder)
            if not p.is_symlink(): os.chown(p,1000,1000)
        os.chown(folder,1000,1000)


if __name__=='__main__': main()
