import ctypes as ct
from types import SimpleNamespace
import threading
import unittest

import numpy as np

from trt_async import AsyncCuda, MAX_TENSOR, Slot


class FakeCuda:
    def __init__(self):
        self.memory={}; self.calls=[]; self.done=False; self.fail=''; self.next=10
    def call(self,name,*args):
        self.calls.append(name)
        if self.fail==name: raise RuntimeError('injected CUDA failure')
        if name in ('cudaMalloc','cudaHostAlloc'):
            array=ct.create_string_buffer(args[1]); address=ct.addressof(array)
            self.memory[address]=array; args[0]._obj.value=address
        elif name in ('cudaStreamCreateWithFlags','cudaEventCreateWithFlags'):
            self.next+=1; args[0]._obj.value=self.next
        elif name in ('cudaFree','cudaFreeHost'): self.memory.pop(args[0].value)
        elif name=='cudaMemcpyAsync': ct.memmove(args[0],args[1],args[2])
    def ready(self,event): return self.done
    def elapsed(self,a,b): return 1.0


class FakeContext:
    def __init__(self,cuda): self.cuda=cuda; self.shape=(1,3,48,32); self.fail=False
    def set_binding_shape(self,index,shape): self.shape=tuple(shape); return True
    def get_binding_shape(self,index): return self.shape if index==0 else (1,self.shape[-1]//8,5)
    def execute_async_v2(self,pointers,stream):
        if self.fail: return False
        # Synthetic model: the output contains the first input value.
        size=int(np.prod(self.get_binding_shape(1)))
        a=(ct.c_float*size).from_address(pointers[1]); value=ct.c_float.from_address(pointers[0]).value
        for i in range(size): a[i]=value
        return True


def session(cuda):
    return SimpleNamespace(cuda=None,engine=SimpleNamespace(num_bindings=2,device_memory_size=256),
        inputs=[SimpleNamespace(name='x',index=0)],outputs=[SimpleNamespace(name='out',index=1)],
        dtypes={0:np.dtype('float32'),1:np.dtype('float32')},
        profiles={'x':[(1,3,48,32),(1,3,48,64),(1,3,48,128)]},context=FakeContext(cuda),
        calls=0,errors=0,in_profile=lambda f:set(f)=={'x'} and f['x'].shape[:3]==(1,3,48) and 32<=f['x'].shape[3]<=128,
        close=lambda:None)


class SlotTests(unittest.TestCase):
    def setUp(self):
        self.cuda=FakeCuda(); self.session=session(self.cuda); self.slot=Slot(self.session,cuda=self.cuda)
    def tearDown(self):
        self.cuda.fail=''; self.slot.quiesce(); self.slot.close(); self.assertEqual(self.cuda.memory,{})
    def x(self,width=32,value=7): return {'x':np.full((1,3,48,width),value,np.float32)}
    def test_event_gates_collect_and_reuse(self):
        self.slot.submit(('page',1,2),self.x())
        self.assertFalse(self.slot.ready())
        with self.assertRaises(RuntimeError): self.slot.collect()
        with self.assertRaises(RuntimeError): self.slot.submit('b',self.x())
        with self.assertRaises(RuntimeError): self.slot.close()
        self.cuda.done=True; ticket,outputs,timing=self.slot.collect()
        self.assertEqual(ticket,('page',1,2)); self.assertTrue((outputs[0]==7).all())
        self.assertEqual(timing['execute_event_ms'],1.0)
        allocations=self.cuda.calls.count('cudaMalloc')
        self.slot.submit('b',self.x(128,9)); _,second,_=self.slot.collect()
        self.assertTrue((outputs[0]==7).all()); self.assertTrue((second[0]==9).all())
        self.assertEqual(self.cuda.calls.count('cudaMalloc'),allocations)
        self.assertNotIn('cudaDeviceSynchronize',self.cuda.calls)
    def test_inputs_checked_before_submission(self):
        for feeds in (self.x(129),{'x':self.x()['x'].astype('float16')},{'x':self.x(value=np.nan)['x']}):
            with self.assertRaises(ValueError): self.slot.submit('a',feeds)
            self.assertFalse(self.slot.busy)
        self.assertEqual(self.session.calls,0)
    def test_enqueue_fault_retains_resources_until_safe_sync(self):
        self.session.context.fail=True
        with self.assertRaises(RuntimeError): self.slot.submit('a',self.x())
        self.assertTrue(self.slot.failed); self.assertTrue(self.slot.busy); memory=len(self.cuda.memory)
        self.cuda.fail='cudaStreamSynchronize'
        with self.assertRaises(RuntimeError): self.slot.quiesce()
        self.assertTrue(self.slot.busy); self.assertEqual(len(self.cuda.memory),memory)
        with self.assertRaises(RuntimeError): self.slot.close()
    def test_result_nonfinite_is_latched(self):
        self.slot.submit('a',self.x()); self.cuda.done=True
        ct.c_float.from_address(self.slot.host[1].value).value=float('nan')
        with self.assertRaises(RuntimeError): self.slot.collect()
        self.assertTrue(self.slot.failed); self.assertTrue(self.slot.busy)
    def test_owner_thread_required(self):
        errors=[]
        def other():
            try: self.slot.submit('a',self.x())
            except RuntimeError: errors.append(True)
        t=threading.Thread(target=other); t.start(); t.join(); self.assertEqual(errors,[True])
    def test_independent_shape_and_buffer_ownership(self):
        c=FakeCuda(); other=Slot(session(c),cuda=c)
        try:
            self.slot.submit('a',self.x(32,3)); other.submit('b',self.x(128,8))
            self.cuda.done=c.done=True
            self.assertEqual(self.slot.collect()[1][0].shape,(1,4,5))
            self.assertEqual(other.collect()[1][0].shape,(1,16,5))
        finally: other.quiesce(); other.close()
    def test_bounded_shape_product(self):
        for shape in ((0,),(-1,),(MAX_TENSOR,), (8192,8192)):
            with self.assertRaises(ValueError): Slot.size(shape,np.float32)
    def test_close_is_idempotent(self):
        self.slot.close(); count=len(self.cuda.calls); self.slot.close()
        self.assertEqual(len(self.cuda.calls),count)

    def test_cuda102_query_status_not_legacy_code(self):
        wrapper=object.__new__(AsyncCuda)
        lib=SimpleNamespace(cudaEventQuery=lambda _:600)
        wrapper.cuda=SimpleNamespace(lib=lib)
        self.assertFalse(wrapper.ready(None))
        lib.cudaEventQuery=lambda _:0; self.assertTrue(wrapper.ready(None))
        lib.cudaEventQuery=lambda _:34
        with self.assertRaises(RuntimeError): wrapper.ready(None)

    def test_route_storage_reuse_excludes_overlapping_submit_and_double_free(self):
        count=self.cuda.calls.count('cudaMalloc')
        borrowed=Slot(session(self.cuda),cuda=self.cuda,storage=self.slot)
        try:
            self.assertEqual(borrowed.host_bytes,0); self.assertEqual(self.cuda.calls.count('cudaMalloc'),count)
            borrowed.submit('wide',self.x(128,8))
            with self.assertRaises(RuntimeError): self.slot.submit('narrow',self.x())
            with self.assertRaises(RuntimeError): self.slot.close()
            self.cuda.done=True; _,values,_=borrowed.collect()
            self.slot.submit('narrow',self.x(32,3)); self.slot.collect()
            self.assertTrue((values[0]==8).all())
        finally: borrowed.quiesce(); borrowed.close()
        self.assertEqual(len(self.cuda.memory),4)

    def test_registry_bytes_are_independent_before_slot_reuse(self):
        self.slot.submit('a',self.x(128,7)); self.cuda.done=True
        _,values,_=self.slot.collect(as_bytes=True)
        shape,data,dtype=values[0]
        self.assertEqual(type(data),bytes)
        self.slot.submit('b',self.x(128,3)); self.slot.collect(as_bytes=True)
        self.assertTrue((np.frombuffer(data,dtype).reshape(shape)==7).all())


if __name__=='__main__': unittest.main()
