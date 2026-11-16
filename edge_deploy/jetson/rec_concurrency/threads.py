"""Optional E166 follow-up: independent owner threads, one primary CUDA context."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
from queue import Queue
import socket
import sys
import threading
import time

import numpy as np
from probe import B,SOURCE,NARROW,WIDE,HASHES,memory,tokens,mapping


def main():
    p=argparse.ArgumentParser(); p.add_argument('--width',type=int,choices=(32,160,1287,2081),required=True)
    p.add_argument('--trace',action='store_true'); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); out=args.output.resolve()
    assert out.is_relative_to(B/'staging/rec_concurrency_e166_r1') and out.parent.is_dir() and not out.exists()
    with socket.socket() as sock: assert sock.connect_ex(('127.0.0.1',5052))!=0
    assert memory()['MemAvailable']>=8*1024**2
    sys.path[:0]=[str(SOURCE/'multistream'),str(B/'releases/jetson-r2/edge_deploy/jetson')]
    from trt82_backend import Session,Cuda
    from trt_async import Slot
    events=Queue(maxsize=32); commands=[]; workers=[]; closed=[]
    report=dict(status='running',kind='threaded_resident_rec',width=args.width,batch=1,synthetic=True,
                passes=[],closed=False,trace_requested=args.trace,memory_before=memory())
    def save(): out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def worker(index,queue):
        session=slot=None; safe=False
        try:
            wide=args.width>1536; session=Session(WIDE if wide else NARROW,HASHES[int(wide)],'thread_rec_'+str(index))
            slot=Slot(session)
            feed={'x':np.random.RandomState(166+index).uniform(-1,1,(1,3,48,args.width)).astype(np.float32)}
            ref=session.run(None,feed)[0]; assert np.isfinite(ref).all()
            for _ in range(3):
                slot.submit(('warm',index),feed); slot.cuda.call('cudaStreamSynchronize',slot.stream); slot.collect()
            slot.submit(('resident',index),feed); slot.cuda.call('cudaStreamSynchronize',slot.stream)
            ptrs=[slot.device[i].value for i in range(session.engine.num_bindings)]
            identity=mapping([slot]) if args.trace else None
            events.put(dict(kind='ready',index=index,mapping=identity))
            while True:
                command=queue.get()
                if command is None: break
                barrier,repeat=command; barrier.wait(timeout=120)
                calls=[]
                for _ in range(repeat):
                    begin=time.monotonic()
                    assert session.context.execute_async_v2(ptrs,slot.stream.value)
                    submitted=time.monotonic(); slot.cuda.call('cudaStreamSynchronize',slot.stream)
                    calls.append(dict(start=begin,submitted=submitted,finished=time.monotonic()))
                finished=time.monotonic()
                a=slot.arrays[session.outputs[0].index]
                slot.cuda.call('cudaMemcpyAsync',slot.host[session.outputs[0].index],slot.device[session.outputs[0].index],a.nbytes,2,slot.stream)
                slot.cuda.call('cudaStreamSynchronize',slot.stream)
                check=dict(all_finite=bool(np.isfinite(a).all()),allclose=bool(np.allclose(a,ref,rtol=1e-3,atol=1e-4)),
                           ctc_indices_equal=tokens(a)==tokens(ref))
                assert all(check.values())
                events.put(dict(kind='finished',index=index,finished=finished,calls=calls,check=check))
        except Exception as e:
            events.put(dict(kind='failed',index=index,error=repr(e)))
        finally:
            try:
                if slot is not None: slot.quiesce()
                safe=True
                if slot is not None: slot.close()
                if session is not None: session.close()
                closed.append(index)
            finally:
                if not safe:
                    print('QUARANTINED owner',index,flush=True)
                    while True: time.sleep(1)
    save(); cuda=None; trace_on=False
    try:
        identities=[]
        for i in range(16):
            if memory()['MemAvailable']<3*1024**2: break
            queue=Queue(maxsize=1); commands.append(queue)
            thread=threading.Thread(target=worker,args=(i,queue),name='rec-owner-'+str(i)); workers.append(thread); thread.start()
            ready=events.get(timeout=180); assert ready['kind']=='ready',ready
            identities.append(ready['mapping'])
        assert workers
        report.update(allocated_slots=len(workers),memory_loaded=memory())
        cuda=Cuda()
        if args.trace:
            assert len({m['context_id'] for m in identities})==1,'All threads must use same primary CUDA context'
            report['trace']=dict(context_id=identities[0]['context_id'],stream_ids={str(i):next(iter(m['stream_ids'].values())) for i,m in enumerate(identities)})
            cuda.call('cudaProfilerStart'); trace_on=True
        counts=([len(workers)] if args.trace else [n for n in (1,2,4,8,16) if n<=len(workers)])
        for direction in (('trace',) if args.trace else ('forward','reverse')):
            for count in (list(reversed(counts)) if direction=='reverse' else counts):
                start=[]; barrier=threading.Barrier(count+1,action=lambda:start.append(time.monotonic()))
                repeat=6 if args.trace else 12
                for q in commands[:count]: q.put((barrier,repeat))
                barrier.wait(timeout=120)
                rows=[events.get(timeout=180) for _ in range(count)]
                assert all(r['kind']=='finished' and all(r['check'].values()) for r in rows),rows
                elapsed=max(r['finished'] for r in rows)-start[0]
                for row in rows:
                    for call in row['calls']:
                        for key in call: call[key]=(call[key]-start[0])*1000
                    row['finished']=(row['finished']-start[0])*1000
                report['passes'].append(dict(direction=direction,slots=count,rounds_per_worker=repeat,
                    seconds=elapsed,inferences_per_second=count*repeat/elapsed,workers=rows))
                save(); print('Threaded',args.width,direction,count,flush=True)
        if trace_on: cuda.call('cudaProfilerStop'); trace_on=False; report['trace']['stopped']=True
        report['status']='threaded_rec_outputs_consistent'
    except Exception as e:
        report.update(status='failed',error=repr(e)); raise
    finally:
        if trace_on and cuda is not None: cuda.call('cudaProfilerStop')
        for q,t in zip(commands,workers):
            if t.is_alive(): q.put(None,timeout=120)
        for t in workers: t.join(timeout=180)
        report['closed']=not any(t.is_alive() for t in workers) and len(closed)==len(workers)
        report['safe_quiescence_confirmed']=report['closed']; report['memory_end']=memory(); save()
        if not report['closed']:
            print('QUARANTINED threads; no kill/reset',flush=True)
            while True: time.sleep(1)


if __name__=='__main__': main()
