"""REC owner threads behind the existing bounded stage/2 and CTC lifecycle."""
from queue import Queue
import threading
import time


class Reply:
    def __init__(self):
        self.done=threading.Event(); self.value=None; self.error=None
    def ready(self):
        if not self.done.is_set(): return False
        if self.error is not None: raise RuntimeError('REC owner failed') from self.error
        return True
    def result(self):
        self.done.wait()
        self.ready()
        return self.value


class Owner:
    def __init__(self,factory,index):
        self.queue=Queue(maxsize=1); self.startup=Reply(); self.closed=False; self.failed=False
        self.calls=0; self.resource_stats={}; self.index=index; self.context=None
        self.thread=threading.Thread(target=self.loop,args=(factory,),name='ocr-rec-owner-'+str(index))
        self.thread.start(); self.startup.result()
    def loop(self,factory):
        try:
            resource=factory(self.index)
            self.context=getattr(resource,'context',None); self.resource_stats=resource.stats()
        except Exception as error:
            self.startup.error=error; self.startup.done.set(); self.closed=True; return
        self.startup.done.set()
        while True:
            op,value,reply=self.queue.get()
            try:
                if op=='run':
                    if self.failed: raise RuntimeError('REC owner fault latched')
                    reply.value=resource.run(value); self.calls+=1
                elif op=='sync': resource.synchronize()
                elif op=='close':
                    resource.close(); self.closed=True
                else: raise ValueError('Unknown owner command')
            except Exception as error:
                self.failed=True; reply.error=error
            finally: reply.done.set()
            if self.closed: return
            value=reply=None  # Idle owners must not retain tensors after registry credit is released.
    def command(self,op,value=None):
        if self.closed: raise RuntimeError('REC owner closed')
        reply=Reply(); self.queue.put_nowait((op,value,reply)); return reply


class Pool:
    def __init__(self,count,factory):
        if count not in (1,2,4): raise ValueError('Expected 1/2/4 owners')
        self.owners=[]; self.closed=False
        try:
            for i in range(count): self.owners.append(Owner(factory,i))
            contexts={o.context for o in self.owners if o.context is not None}
            if len(contexts)>1: raise RuntimeError('REC owners must share the CUDA primary context')
        except Exception:
            self.close(); raise
    def submit(self,index,request): return self.owners[index].command('run',request)
    def synchronize(self):
        replies=[o.command('sync') for o in self.owners if not o.closed]
        errors=[]
        for reply in replies:
            try: reply.result()
            except Exception as error: errors.append(error)
        if errors: raise RuntimeError('REC synchronization unconfirmed') from errors[0]
    def close(self):
        if self.closed: return
        self.synchronize()
        replies=[o.command('close') for o in self.owners if not o.closed]
        for reply in replies: reply.result()
        for owner in self.owners: owner.thread.join()
        self.closed=True
    def stats(self):
        return dict(owners=len(self.owners),alive=sum(o.thread.is_alive() for o in self.owners),
                    calls=sum(o.calls for o in self.owners),errors=sum(o.failed for o in self.owners),
                    context_shared=len({o.context for o in self.owners})==1,
                    fixed_io_bytes=sum(o.resource_stats.get('fixed_io_bytes',0) for o in self.owners))


class Resource:
    def __init__(self,index):
        import ctypes as ct
        from pathlib import Path
        from trt82_backend import Session
        from trt_async import Slot
        from rec_wide_adapter import ENGINE_SHA
        b=Path('/home/xchq/work/autoocr'); self.slots=[]; self.sessions=[]
        try:
            w=Session(b/'overlays/rec-wide-e124-r1/rec_wide_fp32.engine',ENGINE_SHA,'thread_rec_wide_'+str(index)); self.sessions.append(w)
            wide=Slot(w); self.slots.append(wide)
            n=Session(b/'releases/jetson-r2/pth/tensorrt/ocr_rec_fp32_w32_1536_r1.engine',
                      'da2ff03a315a410ccd1d795af129c4220905d45ea2732da0d4441c8ea914c6ff','thread_rec_'+str(index)); self.sessions.append(n)
            narrow=Slot(n,storage=wide); self.slots.append(narrow)
            self.routes=dict(narrow=narrow,wide=wide)
            driver=ct.CDLL('libcuda.so.1'); driver.cuCtxGetCurrent.argtypes=[ct.POINTER(ct.c_void_p)]
            driver.cuCtxGetCurrent.restype=ct.c_int; value=ct.c_void_p()
            if driver.cuCtxGetCurrent(ct.byref(value)) or not value.value: raise RuntimeError('Missing primary context')
            self.context=value.value
        except Exception:
            self.close(); raise
    def run(self,request):
        import numpy as np
        from worker_bridge import Tensor
        t=request.tensors[0]; width=t.shape[3]
        if not 32<=width<=2304: raise ValueError('GPU width outside original profile')
        slot=self.routes['narrow' if width<=1536 else 'wide']
        slot.submit(id(request),{'x':np.frombuffer(t.data,np.float32).reshape(t.shape)})
        slot.cuda.call('cudaStreamSynchronize',slot.stream)
        _,values,timings=slot.collect(as_bytes=True)
        if len(values)!=1: raise ValueError('Unexpected REC outputs')
        name=slot.session.outputs[0].name
        return (Tensor(name,*values[0]),),timings
    def synchronize(self):
        for slot in self.slots: slot.quiesce()
    def close(self):
        self.synchronize()
        for slot in reversed(self.slots): slot.close()
        for session in reversed(self.sessions): session.close()
    def stats(self): return dict(fixed_io_bytes=sum(s.host_bytes+s.device_bytes for s in self.slots))


def runtime_class():
    from runtime import AsyncRuntime
    from ocr_worker_adapter import OCRRuntime
    class ThreadedRuntime(AsyncRuntime):
        def __init__(self,*args,rec_slots=1,**kwargs):
            self.rec_pool=None
            super().__init__(*args,rec_slots=1,**kwargs)
            # Keep original sessions for the business factory, not duplicate IO banks.
            for slot in self.rec[0].values():
                slot.quiesce(); slot.close(); self.slots.remove(slot)
            self.rec_slots_count=rec_slots
            try:
                self.rec_pool=Pool(rec_slots,Resource)
                if self.rec_pool.stats()['fixed_io_bytes']+sum(s.host_bytes+s.device_bytes for s in self.slots)>240*1024**2:
                    raise RuntimeError('Fixed IO exceeds original 240MiB budget')
            except Exception:
                self.close(); raise
        def submit_rec(self,slot_id,ticket,request):
            self._owned()
            if not 1<=slot_id<=self.rec_slots_count: raise ValueError('REC slot')
            return self.rec_pool.submit(slot_id-1,request)
        def collect(self,slot,role):
            if isinstance(slot,Reply):
                values,timing=slot.result()
                self.output_dtypes['rec'].update(t.dtype for t in values)
                return values,timing
            return super().collect(slot,role)
        def synchronize(self):
            self._owned()
            if self.rec_pool is not None: self.rec_pool.synchronize()
            for s in self.slots: s.quiesce()
            OCRRuntime.synchronize(self)
        def close(self):
            self._owned()
            if self.rec_pool is not None: self.rec_pool.close()
            for slot in reversed(self.slots): slot.close()
            OCRRuntime.close(self)
        def stats(self):
            value=OCRRuntime.stats(self)
            if self.rec_pool is not None:
                metrics=self.rec_pool.stats(); value['rec'].update(calls=metrics['calls'],errors=metrics['errors'],threaded=metrics)
            return value
    return ThreadedRuntime
