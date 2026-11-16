"""TRT8.2/CUDA10.2 owned async slot. No global barrier on the inference path.

One deserialized engine per simultaneously usable context; profile 0 is never
shared across live contexts of one engine. Buffers are allocated at startup.
"""
import ctypes as ct
import threading
import time

import numpy as np

MAX_TENSOR=64*1024**2
NOT_READY=600  # Verified with the board's CUDA10.2 cudaGetErrorName, not legacy CUDA codes.


class AsyncCuda:
    def __init__(self, cuda):
        self.cuda=cuda
        signatures={
            'cudaStreamCreateWithFlags':[ct.POINTER(ct.c_void_p),ct.c_uint],
            'cudaStreamDestroy':[ct.c_void_p], 'cudaStreamSynchronize':[ct.c_void_p],
            'cudaHostAlloc':[ct.POINTER(ct.c_void_p),ct.c_size_t,ct.c_uint],
            'cudaFreeHost':[ct.c_void_p],
            'cudaMemcpyAsync':[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_int,ct.c_void_p],
            'cudaEventCreateWithFlags':[ct.POINTER(ct.c_void_p),ct.c_uint],
            'cudaEventRecord':[ct.c_void_p,ct.c_void_p],
            'cudaEventQuery':[ct.c_void_p], 'cudaEventDestroy':[ct.c_void_p],
            'cudaEventElapsedTime':[ct.POINTER(ct.c_float),ct.c_void_p,ct.c_void_p]}
        for name,args in signatures.items():
            function=getattr(cuda.lib,name); function.argtypes=args; function.restype=ct.c_int

    def call(self, name, *args): return self.cuda.call(name,*args)

    def ready(self, event):
        code=self.cuda.lib.cudaEventQuery(event)
        if code==NOT_READY: return False
        if code: raise RuntimeError('cudaEventQuery failed: '+str(code))
        return True

    def elapsed(self, first, last):
        value=ct.c_float(); self.call('cudaEventElapsedTime',ct.byref(value),first,last)
        return float(value.value)


class Slot:
    """Owner-thread only; busy remains latched after uncertain CUDA failures."""
    def __init__(self, session, *, cuda=None, owned_session=False, storage=None):
        self.owner=threading.get_ident(); self.session=session
        self.cuda=cuda or AsyncCuda(session.cuda); self.owned_session=owned_session
        self.stream=ct.c_void_p(); self.events=[]; self.host={}; self.device={}; self.caps={}
        self.busy=self.failed=self.closed=False; self.ticket=None; self.arrays={}
        self.calls=0; self.timings={}; self.host_bytes=self.device_bytes=0
        self.storage=storage.storage if storage is not None else self
        self.active=None
        self.cuda.call('cudaSetDevice',0)
        try:
            self.cuda.call('cudaStreamCreateWithFlags',ct.byref(self.stream),1)  # nonblocking
            for _ in range(4):
                event=ct.c_void_p(); self.cuda.call('cudaEventCreateWithFlags',ct.byref(event),0)
                self.events.append(event)
            # Existing OCR engines have one execution input, linear FP32 IO.
            if len(session.inputs)!=1 or any(dtype!=np.dtype('float32') for dtype in session.dtypes.values()):
                raise ValueError('Expected one FP32 input and FP32 outputs')
            for shape in session.profiles[session.inputs[0].name]:
                if not session.context.set_binding_shape(session.inputs[0].index,shape): raise RuntimeError('shape')
                for item in session.inputs+session.outputs:
                    dims=tuple(session.context.get_binding_shape(item.index))
                    size=self.size(dims,session.dtypes[item.index])
                    self.caps[item.index]=max(self.caps.get(item.index,0),size)
            if storage is not None:
                storage.owned()
                if storage.closed or set(self.caps)!=set(storage.caps) or any(self.caps[i]>storage.caps[i] for i in self.caps):
                    raise ValueError('Incompatible shared route storage')
                self.caps=dict(storage.caps); self.host=storage.host; self.device=storage.device
            for index,size in (() if storage is not None else self.caps.items()):
                host=ct.c_void_p(); device=ct.c_void_p()
                self.cuda.call('cudaHostAlloc',ct.byref(host),size,0); self.host[index]=host
                self.host_bytes+=size
                self.cuda.call('cudaMalloc',ct.byref(device),size); self.device[index]=device
                self.device_bytes+=size
        except Exception:
            self.close(); raise

    @staticmethod
    def size(shape,dtype):
        size=np.dtype(dtype).itemsize
        for n in shape:
            if not isinstance(n,(int,np.integer)) or n<=0 or n>MAX_TENSOR//size: raise ValueError('unresolved/oversize shape')
            size*=int(n)
        return size

    def owned(self):
        if threading.get_ident()!=self.owner: raise RuntimeError('CUDA slot owner differs')

    def submit(self,ticket,feeds):
        self.owned()
        if self.closed or self.busy or self.failed or self.storage.closed or self.storage.active is not None or ticket is None:
            raise RuntimeError('Slot unavailable')
        if not self.session.in_profile(feeds): raise ValueError('Outside original profile')
        arrays={}
        for item in self.session.inputs:
            a=feeds[item.name]
            if not isinstance(a,np.ndarray) or a.dtype!=self.session.dtypes[item.index] or not np.isfinite(a).all():
                raise ValueError('Input dtype/finite differs')
            arrays[item.index]=a
        self.busy=True; self.storage.active=self; self.ticket=ticket; self.timings={}
        start=time.monotonic()
        try:
            for item in self.session.inputs:
                if not self.session.context.set_binding_shape(item.index,arrays[item.index].shape): raise RuntimeError('set shape')
            for item in self.session.inputs+self.session.outputs:
                shape=tuple(self.session.context.get_binding_shape(item.index))
                size=self.size(shape,self.session.dtypes[item.index])
                if size>self.caps[item.index]: raise ValueError('Output exceeds preallocated profile bound')
                a=np.ctypeslib.as_array((ct.c_ubyte*size).from_address(self.host[item.index].value))
                self.arrays[item.index]=a.view(self.session.dtypes[item.index]).reshape(shape)
            for index,a in arrays.items(): np.copyto(self.arrays[index],a)
            self.cuda.call('cudaEventRecord',self.events[0],self.stream)
            for item in self.session.inputs:
                size=self.arrays[item.index].nbytes
                self.cuda.call('cudaMemcpyAsync',self.device[item.index],self.host[item.index],size,1,self.stream)
            self.cuda.call('cudaEventRecord',self.events[1],self.stream)
            pointers=[self.device[i].value for i in range(self.session.engine.num_bindings)]
            begin=time.monotonic()
            if not self.session.context.execute_async_v2(pointers,self.stream.value): raise RuntimeError('TensorRT enqueue failed')
            self.timings['enqueue_host_ms']=(time.monotonic()-begin)*1000
            self.cuda.call('cudaEventRecord',self.events[2],self.stream)
            for item in self.session.outputs:
                self.cuda.call('cudaMemcpyAsync',self.host[item.index],self.device[item.index],self.arrays[item.index].nbytes,2,self.stream)
            self.cuda.call('cudaEventRecord',self.events[3],self.stream)
            self.timings['submit_host_ms']=(time.monotonic()-start)*1000
            self.calls+=1; self.session.calls+=1
        except Exception:
            self.failed=True; self.session.errors+=1; raise

    def ready(self):
        self.owned()
        if not self.busy or self.failed: raise RuntimeError('No queryable task')
        try: return self.cuda.ready(self.events[-1])
        except Exception:
            self.failed=True; self.session.errors+=1; raise

    def collect(self,*,as_bytes=False):
        self.owned()
        if not self.ready(): raise RuntimeError('Task not complete')
        try:
            arrays=[self.arrays[item.index] for item in self.session.outputs]
            if not all(np.isfinite(a).all() for a in arrays): raise RuntimeError('Nonfinite GPU output')
            # Registry bytes are copied once, while this slot still owns the pinned storage.
            outputs=[(tuple(a.shape),a.tobytes(),a.dtype.name) if as_bytes else a.copy() for a in arrays]
            timings=dict(self.timings)
            timings.update({key:self.cuda.elapsed(a,b) for key,a,b in (
                ('h2d_event_ms',self.events[0],self.events[1]),
                ('execute_event_ms',self.events[1],self.events[2]),
                ('d2h_event_ms',self.events[2],self.events[3]))})
            ticket=self.ticket
            self.ticket=None; self.arrays.clear(); self.busy=False; self.storage.active=None
            return ticket,outputs,timings
        except Exception:
            self.failed=True; self.session.errors+=1; raise

    def quiesce(self):
        self.owned()
        # A failed synchronize deliberately leaves busy/resources retained.
        if self.stream.value: self.cuda.call('cudaStreamSynchronize',self.stream)
        if self.storage.active is self: self.storage.active=None
        self.busy=False; self.ticket=None; self.arrays.clear()

    def close(self):
        self.owned()
        if self.closed: return
        if self.busy or (self.storage is self and self.active is not None):
            raise RuntimeError('Must confirm GPU quiescence before close')
        if self.storage is self:
            for ptr in self.device.values(): self.cuda.call('cudaFree',ptr)
            for ptr in self.host.values(): self.cuda.call('cudaFreeHost',ptr)
        self.device={}; self.host={}
        for event in self.events: self.cuda.call('cudaEventDestroy',event)
        self.events.clear()
        if self.stream.value: self.cuda.call('cudaStreamDestroy',self.stream); self.stream=ct.c_void_p()
        if self.owned_session: self.session.close()
        self.closed=True

    def stats(self):
        return dict(calls=self.calls,busy=self.busy,failed=self.failed,host_pinned_bytes=self.host_bytes,
                    device_buffer_bytes=self.device_bytes,engine_device_memory_bytes=self.session.engine.device_memory_size,
                    independent_deserialized_engine=True,profile=0)
