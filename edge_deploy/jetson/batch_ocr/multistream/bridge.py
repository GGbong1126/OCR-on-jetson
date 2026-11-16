"""Bounded rec fan-out/fan-in; I/O registry owner, one GPU owner, one CTC worker."""
from collections import deque
from dataclasses import dataclass
import math
import threading
import time

import numpy as np

from stage_session import encoded, require, StateError
from stage2 import QUEUE_BYTES, QUEUE_ITEMS
from worker_bridge import AlgorithmAPI, Request, Tensor, WorkerBridge


@dataclass(eq=False)
class RecRequest(Request):
    job: object=None
    group: int=0
    index: int=0
    decode: object=None
    value: object=None
    removed: bool=False
    queued_credit: bool=True
    decoding: bool=False


class MultiAPI(AlgorithmAPI):
    def recognize(self,image,boxes,make,decode):
        b=self.bridge; j=self.job; requests=[]; results=[None]*len(boxes)
        with b.cv:
            j.rec_group=getattr(j,'rec_group',0)+1; group=j.rec_group
        try:
            h,w=image.shape[:2]
            for index,box in enumerate(boxes):
                self.checkpoint()
                if len(box)!=4: continue
                x1,y1=max(0,box[:,0].min()),max(0,box[:,1].min())
                x2,y2=min(w,box[:,0].max()),min(h,box[:,1].max())
                if x2<=x1 or y2<=y1: continue
                width=max(32,max(1,min(3200,int(math.ceil(48*((x2-x1)/float(y2-y1)))))))
                size=width*48*3*4; start=time.monotonic()
                with b.cv:
                    b.cv.wait_for(lambda:j.cancelled or (b.rec_preparing_queued<QUEUE_ITEMS and b.rec_queue_bytes+4*size<=QUEUE_BYTES))
                    require(not j.cancelled,j.cancelled or 'STATE')
                    b.rec_preparing_queued+=1; b.rec_queue_bytes+=4*size
                    b.rec_high_items=max(b.rec_high_items,b.rec_preparing_queued)
                    b.rec_high_bytes=max(b.rec_high_bytes,b.rec_queue_bytes)
                try:
                    tensors=make(box); require(type(tensors) is tuple and sum(len(t.data) for t in tensors)==size)
                    request=RecRequest('rec',tensors,job=j,group=group,index=index,decode=decode)
                except Exception:
                    with b.cv: b.rec_preparing_queued-=1; b.rec_queue_bytes-=4*size; b.notify()
                    raise
                with b.cv:
                    b.rec_queue_bytes-=3*size
                    b.rec_requests.append(request); requests.append(request); b.notify()
                    j.timings['rec_prepare_and_backpressure_ms']=j.timings.get('rec_prepare_and_backpressure_ms',0)+(time.monotonic()-start)*1000
            # Closing the producer set, not an empty queue, defines completion.
        finally:
            with b.cv:
                b.cv.wait_for(lambda:all(r.removed for r in requests))
        self.checkpoint()
        for r in requests: results[r.index]=r.value
        return results


class MultiBridge(WorkerBridge):
    def __init__(self,*args,**kwargs):
        self.rec_requests=[]; self.rec_submit={}; self.rec_done={}; self.rec_decodes=deque()
        self.rec_queue_bytes=self.rec_preparing_queued=self.rec_high_items=self.rec_high_bytes=0
        self.rec_high_inflight=self.rec_calls=self.cpu_route_calls=0
        self.rec_timing={}; self.det_timing={}; self.decoder_closed=False; self.cpu_route_closed=False
        self.cpu_work=None; self.runtime=None
        super().__init__(*args,**kwargs)
        self.decoder=threading.Thread(target=self.decode_loop,name='ocr-ctc')
        self.cpu_route=threading.Thread(target=self.cpu_route_loop,name='ocr-rec-cpu-route')
        try:
            self.decoder.start(); self.cpu_route.start()
        except Exception:
            with self.cv:
                self.decoder_closed=self.decoder_closed or self.decoder.ident is None
                self.cpu_route_closed=self.cpu_route_closed or self.cpu_route.ident is None
            while not self.close(1):
                if self.closed: break
            raise

    def cpu_loop(self):
        while True:
            with self.cv:
                self.cv.wait_for(lambda:self.pending or self.closing)
                if not self.pending: return
                job=self.pending.popleft()
            result=b''; error=''
            try:
                api=MultiAPI(self,job); api.checkpoint()
                result=self.algorithm(job.page.payload,job.page.image_type,api)
                from stage_contract import validate_result
                require(type(result) is bytes); validate_result(result)
            except StateError as failure: error=failure.code
            except Exception: error='STATE'
            finally:
                with self.cv:
                    job.result=result if not error else b''; job.error=error; job.finished=True; self.notify()
                result=b''; job=api=None

    def decode_loop(self):
        try:
            while True:
                with self.cv:
                    self.cv.wait_for(lambda:self.rec_decodes or (self.closing and not self.jobs))
                    if not self.rec_decodes: return
                    r=self.rec_decodes.popleft()
                value=None; failure=False; begin=time.monotonic()
                try:
                    if not r.job.cancelled:
                        require(len(r.outputs)==1)
                        t=r.outputs[0]; value=r.decode(np.frombuffer(t.data,np.float32).reshape(t.shape))
                except Exception: failure=True
                finally:
                    with self.cv:
                        if failure: r.job.cancelled='STATE'
                        r.value=value; r.outputs=(); r.consumed=True
                        j=r.job; j.timings['rec_ctc_ms']=j.timings.get('rec_ctc_ms',0)+(time.monotonic()-begin)*1000
                        self.notify()
                    r=None; value=None; t=None
        finally:
            with self.cv: self.decoder_closed=True; self.notify()

    def cpu_route_loop(self):
        try:
            while True:
                with self.cv:
                    self.cv.wait_for(lambda:self.cpu_work is not None or self.closing)
                    if self.cpu_work is None: return
                    r=self.cpu_work
                values=(); failed=False
                try: values=self.runtime.run_cpu(r)
                except Exception: failed=True
                with self.cv:
                    r.finished=time.monotonic()
                    self.rec_done[5]=(True,failed,values,{})
                    self.cpu_work=None; self.notify()
                r=None; values=()
        finally:
            with self.cv: self.cpu_route_closed=True; self.notify()

    def gpu_loop(self):
        runtime=None; active={}; det=None; seen_task=None; gpu_fault=False; fault_safe=False
        try: runtime=self.runtime_factory(); self.runtime=runtime
        except Exception:
            with self.cv: self.init_error=True
        finally:
            with self.cv: self.initialized=True; self.notify()
        try:
            while True:
                with self.cv:
                    task=self.task
                    if task is None: seen_task=None
                    work=[(k,r) for k,r in self.rec_submit.items() if k not in active and k not in self.rec_done]
                    if self.closing and task is None and not self.rec_submit and self.cpu_work is None: break
                if runtime is None:
                    with self.cv: self.cv.wait(.001)
                    continue
                try:
                    if self.faulted and not gpu_fault:
                        gpu_fault=True
                        runtime.synchronize(); fault_safe=True; active.clear(); det=None
                    if task is not None and task is not seen_task and not gpu_fault:
                        batch,requests=task
                        if requests[0].model=='det':
                            begin=time.monotonic()
                            for r in requests: r.started=begin
                            det=runtime.submit_det(task,requests); seen_task=task; self.gpu_calls+=1
                        elif not active and det is None and self.cpu_work is None:
                            begin=time.monotonic()
                            for r in requests: r.started=begin
                            values=(runtime.run(requests[0].model,requests[0].tensors),)
                            for r in requests: r.finished=time.monotonic()
                            self.gpu_host_ms+=(time.monotonic()-begin)*1000
                            with self.cv: self.gpu_result=(True,False,values); self.notify()
                            self.gpu_calls+=1; seen_task=task; values=()
                    barrier=task is not None and task[1][0].model!='det'
                    for slot_id,r in work:
                        if gpu_fault or barrier: continue
                        with self.cv:
                            if self.rec_submit.get(slot_id) is not r or slot_id in self.rec_done: continue
                            if r.job.cancelled and not (slot_id==5 and self.cpu_work is r):
                                self.rec_done[slot_id]=(True,False,(),{}); self.notify(); continue
                        if slot_id==5:
                            with self.cv:
                                if self.cpu_work is None and self.rec_submit.get(5) is r and 5 not in self.rec_done:
                                    self.cpu_work=r; r.started=time.monotonic(); self.cpu_route_calls+=1
                                    self.gpu_calls+=1; self.notify()
                            continue
                        r.started=time.monotonic(); active[slot_id]=runtime.submit_rec(slot_id,r,r)
                        self.gpu_calls+=1; self.rec_calls+=1
                        self.rec_high_inflight=max(self.rec_high_inflight,len(active))
                    if det is not None and det.ready():
                        values,timing=runtime.collect(det,'det'); requests=seen_task[1]
                        for key,value in timing.items(): self.det_timing[key]=self.det_timing.get(key,0)+value
                        t=values[0]
                        if len(requests)==1: outputs=(values,)
                        else:
                            array=np.frombuffer(t.data,t.dtype).reshape(t.shape)
                            outputs=tuple((Tensor(t.name,tuple(array[i:i+1].shape),array[i:i+1].tobytes(),t.dtype),) for i in range(2))
                            array=None
                        if len(requests)==2: runtime.batch_calls+=1
                        for r in requests: r.finished=time.monotonic()
                        with self.cv: self.gpu_result=(True,False,outputs); self.notify()
                        det=None; values=outputs=(); t=None
                    for slot_id,slot in list(active.items()):
                        if not slot.ready(): continue
                        values,timing=runtime.collect(slot,'rec')
                        tensors=values
                        with self.cv:
                            r=self.rec_submit[slot_id]; r.finished=time.monotonic()
                            self.rec_done[slot_id]=(True,False,tensors,timing); self.notify()
                        del active[slot_id]; values=tensors=()
                except Exception:
                    gpu_fault=True; safe=False
                    try: runtime.synchronize(); safe=True
                    except Exception: pass
                    fault_safe=safe
                    with self.cv:
                        if self.task is not None: self.gpu_result=(safe,True,())
                        for k in self.rec_submit:
                            if k!=5: self.rec_done[k]=(safe,True,(),{})
                        if safe: active.clear(); det=None
                        self.notify()
                if gpu_fault:
                    with self.cv:
                        retry=self.retry; self.retry=False
                    if retry:
                        try: runtime.synchronize(); fault_safe=True; active.clear(); det=None
                        except Exception: fault_safe=False
                    with self.cv:
                        if fault_safe:
                            # Accepted commands can arrive behind the failed one before I/O latches it.
                            if self.task is not None and (self.gpu_result is None or not self.gpu_result[0]):
                                self.gpu_result=(True,True,()); self.notify()
                            for k,r in self.rec_submit.items():
                                if k==5 and self.cpu_work is r: continue
                                if k not in self.rec_done or not self.rec_done[k][0]:
                                    self.rec_done[k]=(True,True,(),{}); self.notify()
                        self.cv.wait(.01)
                else:
                    with self.cv: self.cv.wait(.001)
        finally:
            try:
                if runtime is not None: runtime.synchronize(); runtime.close()
            except Exception: self.close_error=True
            with self.cv: self.gpu_closed=True; self.notify()

    def step(self):
        s=self.io.session; s._owned()
        with self.cv:
            for slot_id,(safe,failed,outputs,timing) in list(self.rec_done.items()):
                r=self.rec_submit[slot_id]
                if failed and not self.faulted:
                    self.faulted=True; s.gpu_failed=True
                    for page in s.pages(): s.fail_page(page,'GPU_FAILED')
                    self.pending_fault=s.meta('model.failed',call_ids=[r.call_id],code='GPU_FAILED')
                if not safe: continue
                try:
                    if not self.io.stopped and not r.job.page.error: self.io.event_room()
                    meta=s.complete_rec(slot_id,tuple(t.data for t in outputs))
                    if meta is not None: self.io.emit(meta)
                except StateError as e:
                    if e.code=='BUSY': continue
                    raise
                r.outputs=outputs if not r.job.page.error else ()
                for key,value in timing.items(): self.rec_timing[key]=self.rec_timing.get(key,0)+value
                del self.rec_done[slot_id]; del self.rec_submit[slot_id]; self.notify()
            for r in list(self.rec_requests):
                page=r.job.page
                if r.job.cancelled and not page.error: s.fail_page(page,r.job.cancelled)
                if page.error: r.job.cancelled=page.error
                call=s.rec_calls.get(r.call_id)
                if page.error and (call is None or call.state!='running') and not r.decoding:
                    r.outputs=(); r.consumed=True
                if r.consumed:
                    if call is not None:
                        if call.state=='running': continue
                        if not self.io.stopped and getattr(call,'slot',None) is not None:
                            try: self.io.event_room()
                            except StateError as error:
                                if error.code=='BUSY': continue
                                raise
                        meta=s.release_rec(r.call_id)
                        if meta is not None and not self.io.stopped: self.io.emit(meta)
                    if r.started:
                        for key,value in (('rec_queue_ms',r.started-r.created),('rec_call_wall_ms',r.finished-r.started)):
                            r.job.timings[key]=r.job.timings.get(key,0)+max(0,value)*1000
                    self.release_credit(r); r.tensors=r.outputs=(); r.decode=None; r.removed=True
                    self.rec_requests.remove(r); self.notify(); continue
                if page.error: continue
                if not r.call_id:
                    try:
                        self.io.event_room(); meta=s.ready_rec(page,r.tensors,r.group,r.index,r.index)
                        self.io.emit(meta); r.call_id=meta['call_id']
                    except StateError as error:
                        if error.code=='BUSY': continue
                        s.fail_page(page,error.code); continue
                elif call.state=='resumed' and not r.decoding:
                    if len(self.rec_decodes)>=s.rec_slots+1: continue
                    r.resumed=r.decoding=True; self.rec_decodes.append(r); self.notify()
            for slot_id,batch in s.rec_running.items():
                if slot_id not in self.rec_submit:
                    r=next(r for r in self.rec_requests if r.call_id==batch.calls[0].meta['call_id'])
                    self.release_credit(r); self.rec_submit[slot_id]=r; self.notify()
        super().step()

    def release_credit(self,r):
        if r.queued_credit:
            self.rec_preparing_queued-=1; self.rec_queue_bytes-=sum(len(t.data) for t in r.tensors)
            r.queued_credit=False; require(self.rec_preparing_queued>=0 and self.rec_queue_bytes>=0)

    def request_quiescence(self):
        self.io.session._owned()
        with self.cv:
            require((self.gpu_result is not None and not self.gpu_result[0]) or
                    any(not value[0] for value in self.rec_done.values()))
            self.retry=True; self.notify()

    def close(self,timeout=5):
        self.io.session._owned()
        if self.closed: return not self.close_error
        self.stop(); end=time.monotonic()+timeout
        while time.monotonic()<end:
            self.io.poll(.002); self.step()
            if not self.jobs and self.task is None and self.gpu_closed and self.decoder_closed and self.cpu_route_closed:
                for t in (*self.cpu,self.gpu,self.decoder,self.cpu_route):
                    if t.ident is not None: t.join(max(0,end-time.monotonic()))
                if any(t.is_alive() for t in (*self.cpu,self.gpu,self.decoder,self.cpu_route)): return False
                self.io.selector.unregister(self.reader); self.reader.close(); self.writer.close()
                self.closed=True; return not self.close_error
        return False

    def stats(self):
        d=super().stats()
        with self.cv:
            d.update(rec_tasks=len(self.rec_requests),rec_queue_bytes=self.rec_queue_bytes,
                     rec_preparing_queued=self.rec_preparing_queued,rec_high_items=self.rec_high_items,
                     rec_high_bytes=self.rec_high_bytes,rec_high_inflight=self.rec_high_inflight,
                     rec_calls=self.rec_calls,cpu_route_calls=self.cpu_route_calls,rec_event_totals=dict(self.rec_timing),
                     det_event_totals=dict(self.det_timing))
        return d
