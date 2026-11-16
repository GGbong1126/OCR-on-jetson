from concurrent.futures import Future
import json
import hashlib
import io
import os
from pathlib import Path
import queue
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'ipc'))
import ipc_wire as wire
from channel_io import ChannelServer
from stage_contract import validate_result
from stage_session import encoded, StateError
from worker_bridge import Tensor, WorkerBridge
import batch_contract as batch


def request(path, rows=None, *, raw=None, body=None, ms=300000):
    sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); sock.connect(str(path))
    until=time.monotonic()+10
    if raw is None:
        meta=dict(schema='ocr.batch/1',op='batch.health',remaining_ms=ms); payload=b''
        if rows is not None:
            meta.update(op='batch.submit',images=[dict(image_id=name,image_type=kind,encoded_bytes=len(data)) for name,kind,data in rows])
            payload=b''.join(data for _,_,data in rows)
        raw=encoded(meta); body=payload
    try:
        wire.send_frame(sock,wire.Frame(1,raw,body),until,validator=lambda *_:None)
        reply=wire.receive_frame(sock,until,validator=batch.validate)
        assert reply.request_id==1
        return json.loads(reply.metadata),batch.result(reply.payload)
    finally: sock.close()


class CodecTests(unittest.TestCase):
    def test_batch_limits_and_strict_object_policy(self):
        valid=dict(schema='ocr.batch/1',op='batch.submit',remaining_ms=600000,
                   images=[dict(image_id='a',image_type=3,encoded_bytes=3)])
        self.assertEqual(batch.validate(encoded(valid),3),valid)
        bad=[dict(valid,extra=1),dict(valid,remaining_ms=True),dict(valid,remaining_ms=600001),
             dict(valid,images=valid['images']*2),dict(valid,images=[]),dict(valid,path='/tmp/x')]
        for value in bad:
            with self.assertRaises(wire.ProtocolError): batch.validate(encoded(value),3)
        for key,value in (('image_id','../x'),('encoded_bytes',0),('encoded_bytes',16*1024**2+1),('image_type',True)):
            item=dict(valid['images'][0],**{key:value})
            with self.assertRaises(wire.ProtocolError): batch.validate(encoded(dict(valid,images=[item])),3)
        for raw in (b'{"x":NaN}',b'{"x":{"a":1,"a":2}}',b'[]',b'{"x":"\\ud800"}',b'{"x":1e999}'):
            with self.assertRaises(wire.ProtocolError): batch.result(raw)

    def test_batch_size_does_not_change_page_default(self):
        value=b'{"x":"'+b'a'*(8*1024**2)+b'"}'
        with self.assertRaises(wire.ProtocolError): validate_result(value)
        self.assertEqual(len(batch.result(value)['x']),8*1024**2)
        with self.assertRaises(wire.ProtocolError): validate_result(b'{}',max_bytes=True)


class SyntheticRuntime:
    def __init__(self, gate): self.gate=gate; self.owner=threading.get_ident(); self.calls=0; self.fail=False
    def run(self, model, tensors):
        assert threading.get_ident()==self.owner; self.calls+=1
        assert self.gate.wait(4)
        if self.fail: raise RuntimeError('synthetic GPU fault')
        return (Tensor('output',(1,),tensors[0].data[:4]),)
    def run_batch(self, requests):
        assert threading.get_ident()==self.owner; self.calls+=1
        assert len(requests)==2 and all(model=='det' for model,_ in requests)
        return tuple((Tensor('output',(1,),tensors[0].data[:4]),) for _,tensors in requests)
    def synchronize(self): assert threading.get_ident()==self.owner
    def close(self): assert threading.get_ident()==self.owner


class Backend:
    def __init__(self, folder,det_batch=False,predecessor=False,prepare_delay=0):
        self.path=Path(folder)/'backend.sock'; self.ready=threading.Event(); self.quit=threading.Event()
        self.commands=queue.Queue(4); self.gate=threading.Event(); self.gate.set(); self.errors=[]; self.runtime=None
        self.thread=threading.Thread(target=self.run)
        self.det_batch=det_batch; self.barrier=threading.Barrier(2)
        self.predecessor=predecessor; self.prepare_delay=prepare_delay
    def __enter__(self):
        self.thread.start(); assert self.ready.wait(3)
        if self.errors: raise self.errors[0]
        return self
    def call(self, fn):
        done=Future(); self.commands.put((fn,done),timeout=1); return done.result(3)
    def run(self):
        try:
            with wire.private_listener(self.path) as listener:
                channel=ChannelServer(listener,allow_det_batch=self.det_batch)
                def factory(): self.runtime=SyntheticRuntime(self.gate); return self.runtime
                def algorithm(payload,kind,api):
                    if payload==b'bad': raise ValueError('synthetic bad image')
                    if self.det_batch and payload[0]<=2: self.barrier.wait(2)
                    value=float(payload[0]); observed=[]
                    if self.predecessor and payload[0]==2:
                        with api.infer('enhance',(Tensor('images',(1,3,512,512),struct.pack('f',value)*(3*512*512)),)): pass
                    if payload[0]==2: time.sleep(self.prepare_delay)
                    for model,shape in (('det',(1,3,32,32)),('rec',(1,3,48,32))):
                        data=struct.pack('f',value)*(shape[0]*shape[1]*shape[2]*shape[3])
                        with api.infer(model,(Tensor('x',shape,data),)) as output:
                            observed.append(struct.unpack('f',output[0].data)[0])
                    assert observed==[value,value]
                    return encoded(dict(synthetic=True,value=int(value),image_type=kind,format=api.job.page.format,
                        shape=list(api.job.page.shape),payload_sha256=hashlib.sha256(payload).hexdigest()))
                bridge=WorkerBridge(channel,algorithm,factory); self.ready.set()
                try:
                    while not self.quit.is_set():
                        channel.poll(.005); bridge.step()
                        try: fn,future=self.commands.get_nowait()
                        except queue.Empty: continue
                        try: future.set_result(fn(channel,bridge))
                        except Exception as error: future.set_exception(error)
                finally:
                    self.gate.set(); assert bridge.close(5); channel.close()
        except BaseException as error: self.errors.append(error); self.ready.set()
    def __exit__(self,*_):
        self.gate.set(); self.quit.set(); self.thread.join(6)
        assert not self.thread.is_alive()
        if self.errors: raise self.errors[0]


@unittest.skipUnless(sys.platform=='linux' and os.environ.get('AUTOOCR_BATCH_APP'),'Compiled Linux C++ batch application')
class AppTests(unittest.TestCase):
    def fixture(self, check,*,det_batch=False,jpeg=False,predecessor=False,prepare_delay=0):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            with Backend(folder,det_batch=det_batch,predecessor=predecessor,prepare_delay=prepare_delay) as backend:
                front=Path(folder)/'front.sock'
                process=subprocess.Popen([os.environ['AUTOOCR_BATCH_APP'],str(backend.path),str(front),
                    *(['--det-batch'] if det_batch else []),*(['--jpeg'] if jpeg else [])],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                try:
                    end=time.monotonic()+5
                    while not front.exists() and process.poll() is None and time.monotonic()<end: time.sleep(.005)
                    self.assertTrue(front.exists())
                    check(backend,front,process)
                finally:
                    if process.poll() is None: process.terminate()
                    out,err=process.communicate(timeout=8)
                    self.assertEqual(out,''); self.assertIn('cpp_batch_app',err)
                self.assertFalse(front.exists())
    def test_det_pair_and_odd_tail_keep_slot_identity(self):
        def check(h,path,p):
            _,data=request(path,[('p'+str(i),3,bytes([i+1])) for i in range(3)])
            self.assertEqual([r['data']['value'] for r in data['results']],[1,2,3])
            _,state=request(path)
            self.assertEqual(state['execution']['det_pairs'],1)
            self.assertEqual(state['execution']['model_runs'],5)
            self.assertEqual(h.runtime.calls,5)
        self.fixture(check,det_batch=True)
    def test_det_wait_advances_other_pages_predecessor_without_extra_gpu_thread(self):
        def check(h,path,p):
            _,data=request(path,[('a',3,b'\x01'),('b',3,b'\x02')])
            self.assertEqual([r['data']['value'] for r in data['results']],[1,2])
            self.assertEqual(data['execution']['det_pairs'],1)
            # The predecessor may legally finish before det arrives; pure policy checks cover the defer branch.
            self.assertIn(data['execution']['det_predecessors'],(0,1))
            self.assertEqual(h.runtime.calls,4)
        self.fixture(check,det_batch=True,predecessor=True)
    def test_native_policy_self_check(self):
        process=subprocess.run([os.environ['AUTOOCR_BATCH_APP'],'--self-test'],capture_output=True,text=True,timeout=3)
        self.assertEqual(process.returncode,0); self.assertIn('batch_policy_self_test_passed',process.stdout)
    def test_det_collect_allows_ipc_resume_and_cpu_preparation(self):
        def check(h,path,p):
            _,data=request(path,[('a',3,b'\x01'),('b',3,b'\x02')])
            self.assertEqual([r['data']['value'] for r in data['results']],[1,2])
            self.assertEqual(data['execution']['det_pairs'],1)
        self.fixture(check,det_batch=True,predecessor=True,prepare_delay=.04)
    @unittest.skipUnless(os.environ.get('AUTOOCR_TEST_JPEG')=='1','Matching compiled Jetson hardware path required')
    def test_hardware_partial_mcu_pixels_and_explicit_software_routes(self):
        from PIL import Image, ImageDraw
        image=Image.new('RGB',(512,258),(40,90,180)); ImageDraw.Draw(image).line((0,257,511,257),fill=(230,40,70))
        stream=io.BytesIO(); image.save(stream,format='JPEG',quality=95,subsampling=2); good=stream.getvalue()
        with Image.open(io.BytesIO(good)) as decoded: expected=hashlib.sha256(decoded.convert('RGB').tobytes('raw','BGR')).hexdigest()
        def encode(size,format):
            out=io.BytesIO(); Image.new('RGB',size,(30,60,90)).save(out,format=format); return out.getvalue()
        odd=encode((513,257),'JPEG'); png=encode((512,258),'PNG')
        def check(h,path,p):
            _,data=request(path,[('good',3,good),('odd',3,odd),('png',3,png),('bad',3,b'\xff\xd8broken')])
            rows=data['results']; self.assertEqual([r['status'] for r in rows],['ok','ok','ok','error'])
            self.assertEqual(rows[0]['data']['format'],'bgr8'); self.assertEqual(rows[0]['data']['shape'],[258,512,3])
            self.assertEqual(rows[0]['data']['payload_sha256'],expected)
            for row,payload in zip(rows[1:3],(odd,png)):
                self.assertEqual(row['data']['format'],'encoded')
                self.assertEqual(row['data']['payload_sha256'],hashlib.sha256(payload).hexdigest())
            _,again=request(path,[('again',3,good)]); self.assertEqual(again['results'][0]['status'],'ok')
            stats=again['execution']; self.assertEqual([stats[k] for k in ('jpeg_used','jpeg_software','jpeg_failed')],[2,2,1])
            self.assertEqual(h.runtime.calls,8)
        self.fixture(check,jpeg=True)
    def test_four_pages_two_waves_persistent_batches_and_order(self):
        def check(h,path,p):
            rows=[('p'+str(i),1 if i%2 else 3,bytes([i+1])) for i in range(4)]
            meta,data=request(path,rows)
            self.assertEqual(meta['status'],'ok'); self.assertEqual(meta['count'],4)
            self.assertEqual([v['image_id'] for v in data['results']],[r[0] for r in rows])
            self.assertEqual([v['data']['value'] for v in data['results']],[1,2,3,4])
            self.assertEqual(data['execution']['model_runs'],8)
            meta,data=request(path,[('p0',3,b'\x09')]); self.assertEqual(data['results'][0]['data']['value'],9)
            _,health=request(path); stats=health['execution']
            self.assertEqual(stats['active_high'],512*1024**2)
            self.assertEqual(stats['active_reserved'],0); self.assertEqual(stats['result_reserved'],0)
            self.assertEqual(stats['transport_reserved'],0); self.assertLessEqual(stats['transport_high'],64*1024**2)
            self.assertEqual(stats['input_workers_observed'],2); self.assertEqual(stats['result_workers_observed'],2)
            self.assertEqual(stats['batches'],2); self.assertEqual(stats['model_runs'],10)
            self.assertEqual(h.call(lambda c,b:c.session.stats()['images']),0)
        self.fixture(check)
    def test_bad_image_is_per_item_and_next_batch_can_run(self):
        def check(h,path,p):
            _,data=request(path,[('bad',3,b'bad'),('good',3,b'\x02')])
            self.assertEqual(data['results'][0],dict(image_id='bad',status='error',code='STATE'))
            self.assertEqual(data['results'][1]['status'],'ok')
            _,data=request(path,[('again',3,b'\x03')]); self.assertEqual(data['results'][0]['data']['value'],3)
        self.fixture(check)

    def test_shared_front_client_and_odd_batch(self):
        def check(h,path,p):
            from batch_front_client import exchange
            data=exchange(path,[('a',3,b'\x01'),('b',1,b'\x02'),('c',2,b'\x03')])
            self.assertEqual([r['data']['value'] for r in data['results']],[1,2,3])
            self.assertEqual(exchange(path)['execution']['result_reserved'],0)
        self.fixture(check)
    def test_invalid_front_metadata_does_not_execute_or_destroy_session(self):
        def check(h,path,p):
            base=dict(schema='ocr.batch/1',op='batch.submit',remaining_ms=300000,
                      images=[dict(image_id='a',image_type=3,encoded_bytes=1)])
            for m in (dict(base,path='/tmp/x'),dict(base,images=[]),dict(base,remaining_ms=True),
                      dict(base,images=base['images']*2),dict(base,op='Extract')):
                reply,_=request(path,raw=encoded(m),body=b'x'); self.assertEqual(reply['status'],'error')
            self.assertEqual(h.runtime.calls,0)
            _,data=request(path,[('a',3,b'\x01')]); self.assertEqual(data['results'][0]['status'],'ok')
        self.fixture(check)
    def test_front_disconnect_preserves_running_backend_until_safe_completion(self):
        def check(h,path,p):
            h.gate.clear(); sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); sock.connect(str(path))
            m=dict(schema='ocr.batch/1',op='batch.submit',remaining_ms=300000,images=[dict(image_id='a',image_type=3,encoded_bytes=1)])
            wire.send_frame(sock,wire.Frame(1,encoded(m),b'\x01'),time.monotonic()+2,validator=batch.validate)
            end=time.monotonic()+2
            while h.runtime.calls==0 and time.monotonic()<end: time.sleep(.005)
            self.assertEqual(h.runtime.calls,1); sock.close(); p.wait(timeout=4)
            self.assertEqual(p.returncode,1)
            self.assertTrue(h.call(lambda c,b:c.session.running is not None))
            self.assertEqual(h.call(lambda c,b:c.session.stats()['active']),1)
            h.gate.set()
            end=time.monotonic()+2
            while h.call(lambda c,b:c.session.stats()['images']) and time.monotonic()<end: time.sleep(.005)
            self.assertEqual(h.call(lambda c,b:c.session.stats()['images']),0)
        self.fixture(check)
    def test_backend_disconnect_exits_without_retry(self):
        def check(h,path,p):
            h.call(lambda c,b:c.stop('TEST_DISCONNECT')); p.wait(timeout=3)
            self.assertEqual(p.returncode,1); self.assertEqual(h.runtime.calls,0)
        self.fixture(check)
    def test_gpu_failure_is_reported_and_never_retried(self):
        def check(h,path,p):
            h.runtime.fail=True
            meta,data=request(path,[('a',3,b'\x01')])
            self.assertEqual(meta['status'],'error'); self.assertEqual(data['code'],'GPU_FAILED')
            p.wait(3); self.assertEqual(p.returncode,1); self.assertEqual(h.runtime.calls,1)
            self.assertTrue(h.call(lambda c,b:c.session.gpu_failed))
        self.fixture(check)
    def test_partial_front_frame_can_disconnect_then_health_progresses(self):
        def check(h,path,p):
            sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); sock.connect(str(path)); sock.sendall(b'AOCR\0'); sock.close()
            meta,_=request(path); self.assertEqual(meta['status'],'ok'); self.assertEqual(h.runtime.calls,0)
        self.fixture(check)


if __name__=='__main__': unittest.main()
