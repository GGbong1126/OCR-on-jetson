import json
import os
from pathlib import Path
import selectors
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import ipc_wire as wire
from channel_io import ChannelServer
from stage_session import Session, StateError, encoded
from worker_bridge import Tensor, WorkerBridge


def input_tensor(): return Tensor('x',(1,3,32,32),bytes(12288))


def algorithm(payload, image_type, api):
    with api.infer('det',(input_tensor(),)) as outputs:
        assert struct.unpack('f',outputs[0].data)==(7.,)
    return encoded(dict(synthetic=True,image_id=payload.decode()))


class Runtime:
    def __init__(self):
        self.owner=threading.get_ident(); self.calls=0; self.closed=False
        self.enter=threading.Event(); self.gate=threading.Event(); self.gate.set()
        self.fail_run=self.fail_sync=self.fail_close=False
        self.identities=[]

    def check(self):
        self.identities.append(threading.get_ident()); assert threading.get_ident()==self.owner

    def run(self, model, inputs):
        self.check(); self.calls+=1; self.enter.set()
        assert self.gate.wait(3), 'test gate timed out'
        if self.fail_run: raise RuntimeError('synthetic execution failure')
        assert model=='det' and inputs[0].shape==(1,3,32,32)
        return (Tensor('output',(1,),struct.pack('f',7)),)

    def run_batch(self, requests):
        self.check(); self.calls+=1
        self.enter.set(); assert self.gate.wait(3)
        assert len(requests)==2 and all(model=='det' for model,_ in requests)
        if self.fail_run: raise RuntimeError('synthetic batched failure')
        return tuple((Tensor('output',(1,),struct.pack('f',7)),) for _ in requests)

    def synchronize(self):
        self.check()
        if self.fail_sync: raise RuntimeError('synthetic unsafe device')

    def close(self):
        self.check(); self.closed=True
        if self.fail_close: raise RuntimeError('synthetic close failure')


class OwnerIO:
    """Deterministic event backpressure fixture; production integration tested below."""
    def __init__(self):
        self.session=Session(); self.selector=selectors.DefaultSelector()
        self.stopped=False; self.blocked=False; self.events=[]; self.ids={}

    def event_room(self):
        if self.blocked: raise StateError('BUSY')

    def emit(self, meta): self.event_room(); self.events.append(meta)

    def model_ready(self, page, model, tensors):
        self.event_room(); meta=self.session.ready(page,model,tensors); self.emit(meta); return meta

    def model_complete(self, batch, outputs):
        self.event_room(); meta=self.session.complete(batch,outputs)
        if meta: self.emit(meta)

    def poll(self, timeout):
        for key,_ in self.selector.select(timeout): key.data()

    def stop(self, reason='STOP'):
        if not self.stopped: self.stopped=True; self.session.disconnect()

    def rpc(self, channel, op, payload=b'', **fields):
        self.ids[channel]=self.ids.get(channel,0)+1
        r=self.session.handle(wire.Frame(self.ids[channel],encoded(self.session.meta(op,**fields)),payload),channel)
        return json.loads(r.frame.metadata),r

    def submit(self, name):
        meta,_=self.rpc('input','image.submit',name.encode(),image_id=name,image_type=3,format='encoded')
        assert meta['code']=='OK'


class TensorTests(unittest.TestCase):
    def test_byte_ownership_and_bounded_fp32_product(self):
        self.assertEqual(len(input_tensor().data),12288)
        for shape,data in (((0,),b''),((True,),bytes(4)),((2**30,),b''),((1,),bytearray(4))):
            with self.assertRaises(StateError): Tensor('x',shape,data)

    def test_int32_output_and_no_unknown_dtype(self):
        self.assertEqual(Tensor('count',(1,),bytes(4),'int32').dtype,'int32')
        with self.assertRaises(StateError): Tensor('count',(1,),bytes(4),'float16')


@unittest.skipUnless(sys.platform=='linux','Native AF_UNIX worker wakeups; no Windows TCP emulation')
class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.io=OwnerIO(); self.runtime=None; self.bridge=None

    def start(self, function=algorithm, factory=None):
        def make(): self.runtime=Runtime(); return self.runtime
        self.bridge=WorkerBridge(self.io,function,factory or make)
        self.until(lambda:self.bridge.initialized)

    def until(self, predicate, timeout=2):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            self.io.poll(.005); self.bridge.step()
            if predicate(): return
        self.fail('worker condition timed out')

    def ready(self, name='a'):
        self.io.submit(name)
        self.until(lambda:self.io.session.page(name).state=='ready')
        return self.io.session.page(name)

    def run_page(self, page):
        call=page.call; meta,_=self.io.rpc('gpu','model.run',call_ids=[call.meta['call_id']])
        self.assertEqual(meta['code'],'OK'); self.bridge.step(); return call

    def resume(self, page):
        self.until(lambda:page.state=='done')
        self.assertEqual(self.io.rpc('result','model.resume',call_id=page.call.meta['call_id'])[0]['code'],'OK')
        self.bridge.step()

    def tearDown(self):
        if self.runtime:
            self.runtime.gate.set(); self.runtime.fail_sync=False
        if self.bridge and not self.bridge.closed:
            if self.bridge.stats()['quarantined']: self.bridge.request_quiescence()
            self.assertTrue(self.bridge.close(3))
        self.io.selector.close()

    def test_two_waiting_cpu_workers_have_independent_gpu_owner(self):
        cpu_ids=set(); barrier=threading.Barrier(2)
        def cpu(payload, kind, api):
            cpu_ids.add(threading.get_ident()); barrier.wait(2)
            return algorithm(payload,kind,api)
        self.start(cpu); self.io.submit('a'); self.io.submit('b')
        self.until(lambda:all(p.state=='ready' for p in self.io.session.pages()))
        old=[]
        for name in ('b','a'):
            page=self.io.session.page(name); old.append(self.run_page(page)); self.resume(page)
            self.until(lambda:page.state=='complete')
        self.assertEqual(len(cpu_ids),2); self.assertNotIn(self.runtime.owner,cpu_ids)
        self.assertEqual(set(self.runtime.identities),{self.runtime.owner})
        self.assertEqual(self.runtime.calls,2)
        self.assertTrue(all(not c.inputs and not c.outputs for c in old))
        self.assertEqual([m['image_id'] for m in self.io.events if m['op']=='image.done'],['b','a'])

    def test_cancel_waiting_request_retires_cpu(self):
        self.start(); page=self.ready()
        self.io.rpc('result','image.cancel',image_id='a'); self.bridge.step()
        self.until(lambda:not page.active)
        self.assertEqual(page.error,'CANCELLED'); self.assertEqual(self.runtime.calls,0)
        self.assertEqual(self.bridge.stats()['jobs'],0)

    def test_explicit_det_pair_has_two_owned_results_and_one_execution(self):
        self.io.session.allow_det_batch=True; self.start()
        self.io.submit('a'); self.io.submit('b')
        self.until(lambda:all(p.state=='ready' for p in self.io.session.pages()))
        pages=self.io.session.pages(); calls=[p.call for p in pages]
        reply,_=self.io.rpc('gpu','model.run',call_ids=[c.meta['call_id'] for c in calls])
        self.assertEqual(reply['code'],'OK'); self.bridge.step()
        self.until(lambda:all(p.state=='done' for p in pages))
        for page in pages: self.resume(page)
        self.until(lambda:all(p.state=='complete' for p in pages))
        self.assertEqual(self.runtime.calls,1); self.assertEqual(self.bridge.gpu_calls,1)
        self.assertTrue(all(not c.inputs and not c.outputs for c in calls))

    def test_batched_failure_latches_both_pages_without_single_retry(self):
        self.io.session.allow_det_batch=True; self.start(); self.runtime.fail_run=True
        self.io.submit('a'); self.io.submit('b')
        self.until(lambda:all(p.state=='ready' for p in self.io.session.pages()))
        pages=self.io.session.pages()
        self.io.rpc('gpu','model.run',call_ids=[p.call.meta['call_id'] for p in pages]); self.bridge.step()
        self.until(lambda:all(not p.active for p in pages))
        self.assertTrue(self.io.session.gpu_failed); self.assertEqual(self.runtime.calls,1)
        self.assertEqual([p.error for p in pages],['GPU_FAILED','GPU_FAILED'])

    def test_batched_unsafe_sync_retains_both_inputs_until_explicit_confirmation(self):
        self.io.session.allow_det_batch=True; self.start(); self.runtime.fail_sync=True
        self.io.submit('a'); self.io.submit('b')
        self.until(lambda:all(p.state=='ready' for p in self.io.session.pages()))
        pages=self.io.session.pages(); calls=[p.call for p in pages]
        self.io.rpc('gpu','model.run',call_ids=[c.meta['call_id'] for c in calls]); self.bridge.step()
        self.until(lambda:self.bridge.stats()['quarantined'])
        self.assertFalse(self.bridge.close(.03))
        self.assertTrue(all(c.inputs and c.page.active for c in calls))
        self.runtime.fail_sync=False; self.bridge.request_quiescence()
        self.assertTrue(self.bridge.close(2)); self.assertEqual(self.runtime.calls,1)
        self.assertTrue(all(not c.inputs and not c.page.active for c in calls))

    def test_cancel_one_running_batch_slot_does_not_corrupt_the_other(self):
        self.io.session.allow_det_batch=True; self.start(); self.runtime.gate.clear()
        self.io.submit('a'); self.io.submit('b')
        self.until(lambda:all(p.state=='ready' for p in self.io.session.pages()))
        a,b=self.io.session.pages()
        self.io.rpc('gpu','model.run',call_ids=[a.call.meta['call_id'],b.call.meta['call_id']]); self.bridge.step()
        self.assertTrue(self.runtime.enter.wait(1))
        self.io.rpc('result','image.cancel',image_id='a'); self.bridge.step(); self.runtime.gate.set()
        self.resume(b); self.until(lambda:b.state=='complete' and not a.active)
        self.assertEqual(a.error,'CANCELLED'); self.assertFalse(b.error); self.assertEqual(self.runtime.calls,1)

    def test_running_disconnect_keeps_input_until_gpu_safe(self):
        self.start(); self.runtime.gate.clear(); page=self.ready(); call=self.run_page(page)
        self.assertTrue(self.runtime.enter.wait(1))
        self.assertFalse(self.bridge.close(.03))
        self.assertTrue(call.inputs); self.assertTrue(page.active)
        self.runtime.gate.set(); self.assertTrue(self.bridge.close(2))
        self.assertFalse(call.inputs); self.assertFalse(page.active)
        self.assertEqual(self.io.session.stats()['images'],0)

    def test_execution_error_latches_without_retry_or_fallback(self):
        self.start(); self.runtime.fail_run=True; page=self.ready(); self.run_page(page)
        self.until(lambda:not page.active)
        self.assertTrue(self.io.session.gpu_failed); self.assertEqual(page.error,'GPU_FAILED')
        self.assertEqual(self.runtime.calls,1); self.assertIsNone(self.io.session.running)
        self.assertEqual(self.io.rpc('input','image.submit',b'b',image_id='b',image_type=3,format='encoded')[0]['code'],'GPU_FAILED')

    def test_failed_sync_quarantines_until_explicit_safe_confirmation(self):
        self.start(); self.runtime.fail_sync=True; page=self.ready(); call=self.run_page(page)
        self.until(lambda:self.bridge.stats()['quarantined'])
        self.assertFalse(self.bridge.close(.03)); self.assertTrue(call.inputs)
        self.assertFalse(self.runtime.closed); self.assertTrue(page.active)
        self.runtime.fail_sync=False; self.bridge.request_quiescence()
        self.assertTrue(self.bridge.close(2)); self.assertFalse(call.inputs)
        self.assertTrue(self.io.session.gpu_failed); self.assertEqual(self.runtime.calls,1)

    def test_event_backpressure_before_ready_does_not_register(self):
        self.start(); self.io.blocked=True; self.io.submit('a')
        self.until(lambda:self.io.session.page('a').active)
        for _ in range(5): self.io.poll(.001); self.bridge.step()
        self.assertIsNone(self.io.session.page('a').call)
        self.io.blocked=False; self.until(lambda:self.io.session.page('a').state=='ready')

    def test_completion_backpressure_does_not_rerun_gpu(self):
        self.start(); page=self.ready(); self.io.blocked=True; self.run_page(page)
        self.until(lambda:self.bridge.gpu_result is not None)
        for _ in range(5): self.bridge.step()
        self.assertEqual(self.runtime.calls,1); self.assertEqual(page.state,'running')
        self.io.blocked=False; self.resume(page); self.until(lambda:page.state=='complete')

    def test_finished_result_waits_for_event_room(self):
        gate=threading.Event()
        def cpu(payload,kind,api):
            assert gate.wait(2); return b'{}'
        self.start(cpu); self.io.submit('a'); self.until(lambda:self.bridge.jobs)
        self.io.blocked=True; gate.set(); self.until(lambda:self.bridge.jobs['a'].finished)
        self.assertTrue(self.io.session.page('a').active)
        self.io.blocked=False; self.until(lambda:self.io.session.page('a').state=='complete')

    def test_cpu_failure_invalid_result_and_invalid_model_shape(self):
        for function in (lambda *_:b'[]',lambda *_:b'{"x":NaN}',
                         lambda p,k,a:next(a.infer('det',(Tensor('x',(1,),bytes(4)),)).gen)):
            with self.subTest(function=function):
                self.start(function); self.io.submit('a')
                self.until(lambda:self.io.session.page('a').state=='terminal')
                self.assertEqual(self.io.session.page('a').error,'STATE')
                self.assertTrue(self.bridge.close()); self.bridge=None
                self.io.selector.close(); self.io=OwnerIO()

    def test_multiple_model_stages_release_before_next_ready(self):
        def cpu(p,k,api):
            for _ in range(2):
                with api.infer('det',(input_tensor(),)) as out: self.assertEqual(len(out),1)
            return b'{}'
        self.start(cpu); page=self.ready(); old=[]
        for i in range(2):
            self.until(lambda:page.state=='ready'); old.append(self.run_page(page)); self.resume(page)
            if i==0: self.until(lambda:page.state=='ready')
        self.until(lambda:page.state=='complete')
        self.assertTrue(all(not c.inputs and not c.outputs for c in old))

    def test_factory_failure_stops_without_admission(self):
        def factory(): raise RuntimeError('synthetic initialization failure')
        self.start(factory=factory)
        self.assertTrue(self.io.stopped); self.assertEqual(self.bridge.stats()['jobs'],0)

    def test_owner_guard_and_tensor_integrity(self):
        self.start(); failures=[]
        def wrong():
            try: self.bridge.step()
            except StateError as error: failures.append(error.code)
        t=threading.Thread(target=wrong); t.start(); t.join(1)
        self.assertEqual(failures,['STATE'])
        for shape,data in (((0,),b''),((True,),bytes(4)),((2**30,),b''),((1,),bytearray(4))):
            with self.assertRaises(StateError): Tensor('x',shape,data)

    def test_inflight_admission_and_page_deadline_wake_waiters(self):
        self.start()
        for name in ('a','b','c','d'): self.io.submit(name)
        self.until(lambda:len(self.bridge.jobs)==2 and all(j.request for j in self.bridge.jobs.values()))
        meta,_=self.io.rpc('input','image.submit',b'e',image_id='e',image_type=3,format='encoded')
        self.assertEqual(meta['code'],'BUSY'); self.assertEqual(self.io.session.stats()['active'],2)
        page=self.io.session.page('a'); page.deadline=time.monotonic()-1
        self.until(lambda:not page.active)
        self.assertEqual(page.error,'TIMEOUT'); self.assertLessEqual(self.bridge.high_jobs,2)

    def test_close_failure_reported_but_no_live_threads(self):
        self.start(); self.runtime.fail_close=True
        self.assertFalse(self.bridge.close()); self.assertEqual(self.bridge.stats()['threads_alive'],0)
        self.assertTrue(self.bridge.close_error); self.assertTrue(self.bridge.closed)
        self.assertFalse(self.bridge.close())

    def test_partial_thread_start_failure_joins_started_cpu(self):
        original=threading.Thread.start; started=[]
        def start(thread):
            if thread.name=='ocr-gpu': raise RuntimeError('synthetic thread limit')
            original(thread); started.append(thread)
        with patch.object(threading.Thread,'start',start):
            with self.assertRaises(RuntimeError): WorkerBridge(self.io,algorithm,Runtime)
        self.assertEqual(len(started),2); self.assertFalse(any(t.is_alive() for t in started))
        self.assertEqual(len(self.io.selector.get_map()),0)

    def test_int32_input_is_rejected_before_gpu_submission(self):
        def cpu(payload,kind,api):
            with api.infer('det',(Tensor('x',(1,),bytes(4),'int32'),)): pass
            return b'{}'
        self.start(cpu); self.io.submit('a')
        self.until(lambda:self.io.session.page('a').state=='terminal')
        self.assertEqual(self.io.session.page('a').error,'STATE'); self.assertEqual(self.runtime.calls,0)


@unittest.skipUnless(sys.platform=='linux' and os.environ.get('AUTOOCR_STAGE_PROBE'),'Linux compiled C++ client')
class WorkerChannelTests(unittest.TestCase):
    def test_cpp_four_channels_two_cpu_one_gpu_workers(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path=Path(folder)/'workers.sock'
            with wire.private_listener(path) as listener:
                io=ChannelServer(listener); owners=[]
                def factory():
                    runtime=Runtime(); owners.append(runtime); return runtime
                def cpu(payload,kind,api): return algorithm(api.job.page.image_id.encode(),kind,api)
                bridge=WorkerBridge(io,cpu,factory)
                process=subprocess.Popen([os.environ['AUTOOCR_STAGE_PROBE'],'--workers',str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                try:
                    end=time.monotonic()+12
                    while process.poll() is None and time.monotonic()<end:
                        io.poll(.01); bridge.step()
                    out,err=process.communicate(timeout=1)
                    self.assertEqual(process.returncode,0,err)
                    self.assertEqual(json.loads(out)['status'],'cpp_python_workers_passed_not_ocr')
                    self.assertEqual(bridge.high_jobs,2); self.assertEqual(bridge.gpu_calls,2)
                finally:
                    if process.poll() is None: process.kill(); process.communicate()
                    self.assertTrue(bridge.close()); io.close()
                self.assertTrue(owners[0].closed)
                self.assertEqual(io.session.stats()['images'],0)


if __name__=='__main__': unittest.main()
