import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'ipc'),str(ROOT/'app')]
import ipc_wire as wire
from batch_front_client import exchange
from bridge import MultiBridge
from stage2 import MultiChannel
from stage_session import encoded
from worker_bridge import Tensor


class Event:
    def __init__(self,value,delay): self.value=value; self.until=time.monotonic()+delay
    def ready(self): return time.monotonic()>=self.until


class Runtime:
    names={'det':('out',),'rec':('out',)}
    def __init__(self,fail=False): self.owner=threading.get_ident(); self.calls=0; self.batch_calls=0; self.closed=False; self.fail=fail
    def block_cpu(self): self.cpu_enter=threading.Event(); self.cpu_gate=threading.Event()
    def check(self): assert threading.get_ident()==self.owner
    def submit_rec(self,slot,ticket,request):
        self.check(); self.calls+=1
        if self.fail: raise RuntimeError('injected GPU submit failure')
        return Event(np.frombuffer(request.tensors[0].data,np.float32)[0],.004 if slot%2 else .001)
    def submit_det(self,ticket,requests): self.check(); return Event(1,.002)
    def collect(self,event,role):
        self.check(); assert event.ready()
        return (Tensor('out',(1,1,1),np.array([event.value],np.float32).tobytes()),),{'execute_event_ms':1.}
    def run(self,model,tensors): self.check(); return (Tensor('out',(1,),bytes(4)),)
    def run_cpu(self,request):
        assert threading.get_ident()!=self.owner
        if hasattr(self,'cpu_gate'):
            self.cpu_enter.set(); assert self.cpu_gate.wait(5)
        return (Tensor('out',(1,1,1),request.tensors[0].data[:4]),)
    def synchronize(self): self.check()
    def close(self): self.check(); self.closed=True


class Backend:
    def __init__(self,folder,slots,*,fail_decode=False,zero=False,fail_gpu=False,cpu=False,mixed=False):
        self.path=Path(folder)/'backend.sock'; self.quit=threading.Event(); self.ready=threading.Event()
        self.errors=[]; self.slots=slots; self.fail_decode=fail_decode; self.zero=zero; self.end=None; self.runtime=None
        self.fail_gpu=fail_gpu
        self.width=2305 if cpu else 32
        self.mixed=mixed
    def __enter__(self):
        self.thread=threading.Thread(target=self.run); self.thread.start(); assert self.ready.wait(5)
        if self.errors: raise self.errors[0]
        return self
    def __exit__(self,*_):
        self.quit.set(); self.thread.join(10)
        assert not self.thread.is_alive() and not self.errors,self.errors
    def algorithm(self,payload,kind,api):
        if self.mixed and api.job.page.image_id=='b':
            assert self.runtime.cpu_enter.wait(3)
            with api.bridge.cv:
                assert api.bridge.cv.wait_for(lambda:any(r.job.page.image_id=='a' and r.index==1 and r.call_id for r in api.bridge.rec_requests),1)
        width=32 if self.mixed and api.job.page.image_id=='b' else self.width
        image=np.zeros((48,width,3),np.uint8)
        count=2 if self.mixed and api.job.page.image_id=='a' else 40
        boxes=[] if self.zero else [np.array([[0,0],[width,0],[width,48],[0,48]]) for _ in range(count)]
        seq=[0]
        def make(_):
            i=seq[0]; seq[0]+=1
            return (Tensor('x',(1,3,48,width),np.full((1,3,48,width),i,np.float32).tobytes()),)
        def decode(array):
            if self.fail_decode: raise ValueError('injected CPU decode')
            time.sleep(.001)
            return int(array.flat[0])
        results=api.recognize(image,boxes,make,decode)
        return encoded(dict(values=results))
    def run(self):
        io=bridge=None
        try:
            with wire.private_listener(self.path) as listener:
                io=MultiChannel(listener,self.slots,transport_bytes=64*1024**2)
                def factory():
                    self.runtime=Runtime(self.fail_gpu)
                    if self.mixed: self.runtime.block_cpu()
                    return self.runtime
                bridge=MultiBridge(io,self.algorithm,factory); self.ready.set()
                while not self.quit.is_set() and not io.stopped: io.poll(.002); bridge.step()
                end=time.monotonic()+8
                while not bridge.close(.1) and time.monotonic()<end: pass
                assert bridge.closed and bridge.decoder_closed and bridge.cpu_route_closed
                self.end=dict(worker=bridge.stats(),session=io.session.stats()); io.close()
        except Exception as error: self.errors.append(error); self.ready.set()


@unittest.skipUnless(sys.platform=='linux' and os.environ.get('AUTOOCR_BATCH_APP'),'Linux compiled C++ candidate required')
class PipelineTests(unittest.TestCase):
    def exercise(self,slots,*,zero=False,fail_decode=False,fail_gpu=False,cpu=False):
        with tempfile.TemporaryDirectory(prefix='multi_') as folder,Backend(folder,slots,zero=zero,fail_decode=fail_decode,fail_gpu=fail_gpu,cpu=cpu) as backend:
            front=Path(folder)/'front.sock'
            cpp=subprocess.Popen([os.environ['AUTOOCR_BATCH_APP'],str(backend.path),str(front),'--rec-slots='+str(slots)],
                                 stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            try:
                end=time.monotonic()+5
                while not front.exists() and cpp.poll() is None and time.monotonic()<end: time.sleep(.01)
                assert front.exists(),cpp.stderr.read().decode()
                if fail_decode or fail_gpu:
                    from stage_session import StateError
                    with self.assertRaises(StateError): exchange(front,[('a',3,b'a'),('b',3,b'b')],remaining_ms=10000)
                    cpp.wait(5); self.assertEqual(cpp.returncode,1)
                    if fail_gpu: self.assertEqual(backend.runtime.calls,1)
                    return
                value=exchange(front,[('a',3,b'a'),('b',3,b'b')],remaining_ms=10000)
                for r in value['results']:
                    if fail_decode: self.assertEqual(r['status'],'error')
                    else: self.assertEqual(r['data']['values'],[] if zero else list(range(40)))
                stats=exchange(front)['execution']; self.assertLessEqual(stats['rec_queue_high'],8)
                self.assertLessEqual(stats['rec_queue_bytes_high'],16*1024**2)
                self.assertLessEqual(stats['rec_high_slots'],slots)
            finally:
                if cpp.poll() is None: cpp.terminate()
                cpp.wait(5); cpp.stderr.close()
        self.assertEqual(backend.end['worker']['rec_tasks'],0)
        self.assertEqual(backend.end['worker']['rec_queue_bytes'],0)
        self.assertEqual(backend.end['session']['registered_bytes'],0)
        return backend.end
    def test_rec1(self): self.exercise(1)
    def test_rec2(self): self.exercise(2)
    def test_rec4_and_bounded_queue_pressure(self): self.exercise(4)
    def test_zero_crop(self): self.exercise(4,zero=True)
    def test_decode_error_cancel_and_clean_stop(self): self.exercise(2,fail_decode=True)
    def test_gpu_fault_latched_no_inference_retry(self): self.exercise(4,fail_gpu=True)
    def test_cpu_width_route_uses_independent_bounded_worker(self):
        end=self.exercise(2,cpu=True)
        self.assertEqual(end['worker']['cpu_route_calls'],80)
        self.assertEqual(end['worker']['rec_high_inflight'],0)
    def test_busy_cpu_route_does_not_head_of_line_block_gpu(self):
        with tempfile.TemporaryDirectory(prefix='multi_') as folder,Backend(folder,2,cpu=True,mixed=True) as backend:
            front=Path(folder)/'front.sock'; results=[]; errors=[]
            cpp=subprocess.Popen([os.environ['AUTOOCR_BATCH_APP'],str(backend.path),str(front),'--rec-slots=2'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            def client():
                try: results.append(exchange(front,[('a',3,b'a'),('b',3,b'b')],remaining_ms=10000))
                except Exception as e: errors.append(e)
            thread=None
            try:
                end=time.monotonic()+5
                while not front.exists() and time.monotonic()<end: time.sleep(.01)
                thread=threading.Thread(target=client); thread.start()
                assert backend.runtime.cpu_enter.wait(3)
                end=time.monotonic()+2
                while backend.runtime.calls==0 and time.monotonic()<end: time.sleep(.005)
                self.assertGreater(backend.runtime.calls,0,'GPU queued behind busy CPU route')
            finally:
                backend.runtime.cpu_gate.set()
                if thread: thread.join(5)
                if cpp.poll() is None: cpp.terminate()
                cpp.wait(5)
            self.assertFalse(errors); self.assertEqual(len(results),1)
            self.assertEqual([len(r['data']['values']) for r in results[0]['results']],[2,40])


if __name__=='__main__': unittest.main()
