"""E120 first-attempt CUDA child diagnostic; no raw data or deployment changes."""
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import benchmark_ner_protected as b


def environment():
    return dict(cwd=os.getcwd(),environment={k:os.environ.get(k) for k in
        ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','LD_LIBRARY_PATH','LD_PRELOAD','CUDA_CACHE_PATH')})


def child(label,stage):
    import ctypes as ct
    from trt82_backend import Cuda
    r=dict(label=label,pid=os.getpid(),**environment())
    try:Cuda();r['cuda_ok']=True
    except Exception as e:r.update(cuda_ok=False,error=str(e))
    r['mapped_libraries']=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
        if '/' in line and any(s in line for s in ('libcuda','libnvinfer','libnvrm','libnvos'))})
    b.q.save(stage/('spawn_'+label+'.json'),r);print(json.dumps(r),flush=True)


if __name__=='__main__':
    a=b.h.arguments(('diagnose',));path=a.stage/'spawn_diagnostic.json';b.old.fresh(path)
    for label in ('before','after'):b.old.fresh(a.stage/('spawn_'+label+'.json'))
    ctx=mp.get_context('spawn');result=dict(before=environment())
    p=ctx.Process(target=child,args=('before',a.stage));p.start();p.join(30)
    if p.is_alive():p.terminate();p.join(10);raise RuntimeError('Probe timeout')
    if p.exitcode:raise RuntimeError('Probe failed')
    rows,pages,execution=b.n.capture(a);del rows;gc.collect()
    result.update(after=environment(),execution=execution)
    p=ctx.Process(target=child,args=('after',a.stage));p.start();p.join(30)
    if p.is_alive():p.terminate();p.join(10);raise RuntimeError('Probe timeout')
    if p.exitcode:raise RuntimeError('Probe failed')
    b.q.save(path,result)
