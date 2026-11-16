"""Two CPU algorithm workers, one GPU owner, and bounded owner-thread handoffs.

No HTTP, models, or automatic execution-error fallback. The runtime adapter must
confirm quiescence before buffers/context can be freed, including on exceptions.
"""
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
import selectors
import socket
import sys
import threading
import time

import ipc_wire as wire
from stage_contract import LIMITS, validate_result
from stage_session import StateError, require


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple
    data: bytes
    dtype: str='float32'

    def __post_init__(self):
        require(type(self.name) is str and 1<=len(self.name)<=64)
        require(type(self.dtype) is str and self.dtype in ('float32','int32'))
        require(type(self.shape) is tuple and 1<=len(self.shape)<=4,'LIMIT')
        size=4
        for n in self.shape:
            require(type(n) is int and n>0 and n<=LIMITS['max_tensor_bytes']//size,'LIMIT')
            size*=n
        require(type(self.data) is bytes and len(self.data)==size,'LIMIT')


@dataclass(eq=False)
class Request:
    model: str
    tensors: tuple
    call_id: int=0
    outputs: tuple=()
    resumed: bool=False
    consumed: bool=False
    created: float=field(default_factory=time.monotonic)
    started: float=0
    finished: float=0


@dataclass(eq=False)
class Job:
    page: object
    request: object=None
    cancelled: str=''
    finished: bool=False
    result: bytes=b''
    error: str=''
    timings: dict=field(default_factory=dict)


class AlgorithmAPI:
    def __init__(self, bridge, job): self.bridge=bridge; self.job=job

    def checkpoint(self):
        with self.bridge.cv:
            require(not self.job.cancelled,self.job.cancelled or 'STATE')

    @contextmanager
    def infer(self, model, tensors):
        b=self.bridge; j=self.job
        require(type(tensors) is tuple and 1<=len(tensors)<=3)
        require(all(type(t) is Tensor for t in tensors))
        require(all(t.dtype=='float32' for t in tensors))
        require(sum(len(t.data) for t in tensors)<=LIMITS['max_tensor_bytes'],'LIMIT')
        request=Request(model,tensors)
        with b.cv:
            require(not j.cancelled,j.cancelled or 'STATE'); require(j.request is None)
            j.request=request; b.notify()
            b.cv.wait_for(lambda:request.resumed or j.cancelled)
            require(not j.cancelled,j.cancelled or 'STATE')
            outputs=request.outputs
            for key,value in (('model_queue_ms',request.started-request.created),
                              ('model_host_call_ms',request.finished-request.started)):
                j.timings[key]=j.timings.get(key,0)+max(0,value)*1000
        try:
            yield outputs
        finally:
            outputs=()
            with b.cv:
                request.consumed=True; b.notify()
                b.cv.wait_for(lambda:j.request is not request or j.cancelled)


class WorkerBridge:
    def __init__(self, io, algorithm, runtime_factory):
        io.session._owned()
        require(sys.platform=='linux')
        self.io=io; self.algorithm=algorithm; self.runtime_factory=runtime_factory
        self.cv=threading.Condition(); self.jobs={}; self.pending=deque()
        self.task=None; self.gpu_result=None; self.retry=False; self.faulted=False
        self.pending_fault=None; self.initialized=False; self.init_error=False
        self.closing=self.closed=False; self.gpu_closed=False; self.close_error=False
        self.gpu_calls=0; self.high_jobs=0; self.gpu_host_ms=0
        self.reader,self.writer=socket.socketpair(socket.AF_UNIX,socket.SOCK_STREAM)
        self.reader.setblocking(False); self.writer.setblocking(False)
        io.selector.register(self.reader,selectors.EVENT_READ,self.drain_wakeup)
        self.cpu=[threading.Thread(target=self.cpu_loop,name='ocr-cpu-'+str(i)) for i in range(2)]
        self.gpu=threading.Thread(target=self.gpu_loop,name='ocr-gpu')
        started=[]
        try:
            for thread in [*self.cpu,self.gpu]: thread.start(); started.append(thread)
        except Exception:
            with self.cv: self.closing=True; self.notify()
            for thread in started: thread.join()
            io.selector.unregister(self.reader); self.reader.close(); self.writer.close()
            raise

    def notify(self):
        self.cv.notify_all()
        try: self.writer.send(b'x')
        except (BlockingIOError,InterruptedError): pass

    def drain_wakeup(self):
        while True:
            try:
                if not self.reader.recv(4096): break
            except (BlockingIOError,InterruptedError): break

    def cpu_loop(self):
        while True:
            with self.cv:
                self.cv.wait_for(lambda:self.pending or self.closing)
                if not self.pending: return
                job=self.pending.popleft()
            result=b''; error=''
            try:
                api=AlgorithmAPI(self,job); api.checkpoint()
                result=self.algorithm(job.page.payload,job.page.image_type,api)
                require(type(result) is bytes,'STATE'); validate_result(result)
            except StateError as failure: error=failure.code
            except Exception: error='STATE'
            finally:
                with self.cv:
                    job.result=result if not error else b''; job.error=error
                    job.finished=True; self.notify()
                result=b''; job=None; api=None

    def gpu_loop(self):
        runtime=None
        try:
            runtime=self.runtime_factory()
        except Exception:
            with self.cv: self.init_error=True
        finally:
            with self.cv: self.initialized=True; self.notify()
        while True:
            with self.cv:
                self.cv.wait_for(lambda:(self.task is not None and self.gpu_result is None) or
                                self.retry or (self.closing and self.task is None))
                if self.closing and self.task is None: break
                task=self.task; retry=self.retry; self.retry=False
            outputs=(); failed=False; safe=False
            if not retry:
                try:
                    self.gpu_calls+=1
                    requests=task[1]
                    started=time.monotonic()
                    for request in requests: request.started=started
                    outputs=((runtime.run(requests[0].model,requests[0].tensors),) if len(requests)==1 else
                             runtime.run_batch(tuple((r.model,r.tensors) for r in requests)))
                    require(type(outputs) is tuple and len(outputs)==len(requests))
                    for values in outputs:
                        require(type(values) is tuple and 1<=len(values)<=16)
                        require(all(type(t) is Tensor for t in values))
                        require(sum(len(t.data) for t in values)<=LIMITS['max_tensor_bytes'],'LIMIT')
                except Exception: failed=True; outputs=()
            else: failed=True
            try: runtime.synchronize(); safe=True
            except Exception: failed=True; outputs=()
            if not retry:
                finished=time.monotonic(); self.gpu_host_ms+=(finished-started)*1000
                for request in requests: request.finished=finished
            with self.cv:
                self.gpu_result=(safe,failed,outputs); self.notify()
                # An unsafe task stays owned here until an explicit synchronization retry.
                self.cv.wait_for(lambda:self.task is not task or self.retry)
            task=None; outputs=()
        try:
            if runtime is not None: runtime.close()
        except Exception: self.close_error=True
        finally:
            runtime=None
            with self.cv: self.gpu_closed=True; self.notify()

    def _emit_fault(self):
        if self.pending_fault is not None:
            if not self.io.stopped:
                self.io.event_room(); self.io.emit(self.pending_fault)
            self.pending_fault=None

    def step(self):
        """Called by the I/O owner after poll, never from a worker thread."""
        s=self.io.session; s._owned(); require(not self.closed); s.expire()
        with self.cv:
            if self.init_error and not self.io.stopped: self.io.stop('GPU_INIT_FAILED')
            if self.gpu_result is not None:
                safe,failed,outputs=self.gpu_result
                batch,requests=self.task
                if failed and not self.faulted:
                    self.faulted=True; self.pending_fault=s.gpu_fault(batch)
                if safe:
                    if failed or all(c.page.error for c in batch.calls):
                        s.complete(batch,[() for _ in batch.calls])
                        for request in requests: request.outputs=()
                    else:
                        try: self.io.model_complete(batch,[tuple(t.data for t in values) for values in outputs])
                        except StateError as error:
                            if error.code!='BUSY': raise
                        else:
                            for call,request,values in zip(batch.calls,requests,outputs):
                                request.outputs=() if call.page.error else values
                            self.task=None; self.gpu_result=None
                    if failed or all(c.page.error for c in batch.calls):
                        self.task=None; self.gpu_result=None
                    self.notify()
            try: self._emit_fault()
            except StateError as error:
                if error.code!='BUSY': raise
            for job in list(self.jobs.values()):
                page=job.page; request=job.request
                if page.error: job.cancelled=page.error; self.notify()
                if request and request.consumed and page.call and page.call.state=='resumed':
                    s.release_call(page); request.tensors=request.outputs=()
                    job.request=None; self.notify(); request=None
                if job.finished:
                    if job.error and not page.error: s.fail_page(page,job.error)
                    if page.call and page.call.state=='running': continue
                    if page.error:
                        if not self.io.stopped:
                            try: self.io.event_room()
                            except StateError as error:
                                if error.code=='BUSY': continue
                                raise
                        s.retire(page)
                    else:
                        try:
                            self.io.event_room(); s.finish(page,job.result)
                        except StateError as error:
                            if error.code=='BUSY': continue
                            s.fail_page(page,error.code); continue
                    if not self.io.stopped:
                        fields=dict(image_id=page.image_id,status='ok')
                        if page.error:
                            fields.update(status='cancelled' if page.error=='CANCELLED' else 'error',code=page.error)
                        self.io.emit(s.meta('image.done',**fields))
                    if request: request.tensors=request.outputs=()
                    job.request=None; job.result=b''; self.jobs.pop(page.image_id); self.notify()
                    continue
                if request and not page.error:
                    if not request.call_id:
                        try:
                            meta=self.io.model_ready(page,request.model,[(t.name,t.shape,t.data) for t in request.tensors])
                            request.call_id=meta['call_id']
                        except (StateError,wire.ProtocolError) as error:
                            if isinstance(error,StateError) and error.code=='BUSY': continue
                            s.fail_page(page,'STATE'); job.cancelled=page.error; self.notify()
                    elif page.call and page.call.state=='resumed' and not request.resumed:
                        request.resumed=True; self.notify()
            if s.running is not None and self.task is None:
                requests=[]
                for call in s.running.calls:
                    job=self.jobs[call.page.image_id]
                    require(job.request is not None and call.meta['call_id']==job.request.call_id)
                    requests.append(job.request)
                self.task=(s.running,tuple(requests)); self.notify()
            if self.initialized and not self.closing and not s.gpu_failed and not self.io.stopped:
                for page in s.pages():
                    if len(self.jobs)>=2: break
                    if page.state=='queued':
                        s.activate(page); job=Job(page); self.jobs[page.image_id]=job
                        self.pending.append(job); self.high_jobs=max(self.high_jobs,len(self.jobs)); self.notify()

    def request_quiescence(self):
        """Retry only synchronization on the GPU owner, never inference/reset/fallback."""
        self.io.session._owned()
        with self.cv:
            require(self.task is not None and self.gpu_result is not None and not self.gpu_result[0])
            self.gpu_result=None; self.retry=True; self.notify()

    def stop(self):
        self.io.session._owned(); self.io.stop('WORKERS_STOPPED')
        with self.cv:
            self.closing=True
            for job in self.jobs.values(): job.cancelled=job.page.error or 'DISCONNECTED'
            self.notify()

    def close(self, timeout=5):
        self.io.session._owned()
        if self.closed: return not self.close_error
        self.stop(); end=time.monotonic()+timeout
        while time.monotonic()<end:
            self.io.poll(.01); self.step()
            if not self.jobs and self.task is None and self.gpu_closed:
                for thread in [*self.cpu,self.gpu]: thread.join(max(0,end-time.monotonic()))
                if any(t.is_alive() for t in [*self.cpu,self.gpu]): return False
                self.io.selector.unregister(self.reader); self.reader.close(); self.writer.close()
                self.closed=True; return not self.close_error
        return False

    def stats(self):
        self.io.session._owned()
        with self.cv:
            return dict(jobs=len(self.jobs),high_jobs=self.high_jobs,pending=len(self.pending),
                        gpu_calls=self.gpu_calls,gpu_host_ms=self.gpu_host_ms,gpu_task=self.task is not None,
                        quarantined=self.gpu_result is not None and not self.gpu_result[0],
                        threads_alive=sum(t.is_alive() for t in [*self.cpu,self.gpu]),
                        initialized=self.initialized,gpu_closed=self.gpu_closed,close_error=self.close_error)
