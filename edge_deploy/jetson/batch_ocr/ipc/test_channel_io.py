from concurrent.futures import Future
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock

import ipc_wire as wire
import stage_contract as contract
from channel_io import ChannelServer, Reader, Writer
from stage_session import Session, StateError, encoded
from test_stage_contract import base


class CodecTests(unittest.TestCase):
    def test_partitioned_transport_budget_rejects_without_overcommit(self):
        server=ChannelServer.__new__(ChannelServer); server.transport_limit=64*1024**2; server.used=server.high=0
        server.reserve(63*1024**2)
        with self.assertRaises(StateError): server.reserve(2*1024**2)
        self.assertEqual(server.used,63*1024**2)
        server.release(63*1024**2); self.assertEqual(server.used,0)

    def test_bind_schema_and_reserved_request_id(self):
        for op in ('channel.bind','channel.bound'):
            for role in ('input','gpu','result','events'):
                contract.validate_stage(encoded(base(op,channel=role)),0)
            for role in ('Extract',True,[],None):
                with self.assertRaises(wire.ProtocolError): contract.validate_stage(encoded(base(op,channel=role)),0)
        s=Session(); s.bind_channel('input')
        m=s.meta('image.submit',image_id='a',image_type=3,format='encoded')
        reply=s.handle(wire.Frame(1,encoded(m),b'x'),'input')
        self.assertEqual(json.loads(reply.frame.metadata)['code'],'STATE')
        with self.assertRaises(StateError): s.bind_channel('input')

    def test_incremental_reader_preserves_coalesced_frame_and_credits(self):
        metadata=encoded(base('image.submit',image_id='a',image_type=3,format='encoded'))
        raw=wire.header(2,len(metadata),3)+metadata+b'abc'
        pending=bytearray(raw+raw); sock=Mock(); credits=[0]
        def read(view):
            n=min(len(view),3,len(pending)); view[:n]=pending[:n]; del pending[:n]; return n
        sock.recv_into.side_effect=read
        reader=Reader(contract.validate_stage,lambda n:credits.__setitem__(0,credits[0]+n),
                      lambda n:credits.__setitem__(0,credits[0]-n),lambda:0,5)
        for _ in range(2):
            frame=None
            for _ in range(1000):
                frame=reader.read(sock,1)
                if frame: break
            self.assertEqual(frame.payload,b'abc'); self.assertEqual(credits[0],6)
            reader.reset(); self.assertEqual(credits[0],0)
        self.assertFalse(pending)

    def test_reader_rejects_metadata_before_body_reservation(self):
        meta=encoded(base('model.run',call_ids=[1])); pending=bytearray(wire.header(2,len(meta),wire.MAX_PAYLOAD)+meta)
        sock=Mock(); reserved=[]
        def read(view):
            n=min(len(view),len(pending)); view[:n]=pending[:n]; del pending[:n]; return n
        sock.recv_into.side_effect=read
        reader=Reader(contract.validate_stage,reserved.append,lambda n:None,lambda:0,5)
        with self.assertRaises(wire.ProtocolError): reader.read(sock)
        self.assertEqual(reserved,[]); self.assertEqual(reader.phase,'metadata')
        now=[0.0]; pending.extend(wire.header(2,len(meta),0)+meta)
        def expired(raw,n): contract.validate_stage(raw,n); now[0]=1
        reader=Reader(expired,reserved.append,lambda n:None,lambda:now[0],1)
        with self.assertRaises(TimeoutError): reader.read(sock)
        self.assertEqual(reserved,[])

    def test_reader_partial_progress_does_not_extend_deadline(self):
        now=[0.0]; sock=Mock(); sock.recv_into.side_effect=lambda view:(view.__setitem__(0,65) or 1)
        reader=Reader(contract.validate_stage,lambda n:None,lambda n:None,lambda:now[0],1)
        reader.read(sock,1); self.assertEqual(reader.deadline,1)
        now[0]=.9; reader.read(sock,1); self.assertEqual(reader.deadline,1)
        now[0]=1
        with self.assertRaises(TimeoutError): reader.read(sock)

    def test_writer_short_sends_and_would_block(self):
        m=encoded(base('image.submit',image_id='a',image_type=3,format='encoded'))
        frame=wire.Frame(2,m,b'abc'); writer=Writer(frame,1); output=bytearray(); calls=[0]; sock=Mock()
        def send(view):
            calls[0]+=1
            if calls[0]%4==0: raise BlockingIOError()
            n=min(3,len(view)); output.extend(view[:n]); return n
        sock.send.side_effect=send
        for _ in range(1000):
            if writer.write(sock,7): break
        self.assertEqual(bytes(output),wire.header(2,len(m),3)+m+b'abc')


class Harness:
    """Test-only owner-thread mailbox and synthetic driver, not an OCR worker bridge."""
    def __init__(self, folder, *, seconds=2, auto=False):
        self.path=Path(folder)/'channels.sock'; self.seconds=seconds; self.auto=auto
        self.started=threading.Event(); self.quit=threading.Event(); self.commands=queue.Queue(4); self.errors=[]
        self.thread=threading.Thread(target=self.run)

    def __enter__(self):
        self.thread.start()
        if not self.started.wait(3): raise AssertionError('I/O fixture did not start')
        if self.errors: raise self.errors[0]
        return self

    def call(self, function):
        future=Future(); self.commands.put((function,future),timeout=2)
        return future.result(timeout=3)

    def run(self):
        io=None
        try:
            with wire.private_listener(self.path) as listener:
                io=ChannelServer(listener,seconds=self.seconds); self.started.set()
                try:
                    while not self.quit.is_set():
                        notices=io.poll(.01)
                        if self.auto:
                            for _,meta,ok in notices:
                                if not ok: continue
                                if meta['op']=='image.submit' and io.session.stats()['images']==2:
                                    for image in ('b','a'):
                                        p=io.session.page(image); io.session.activate(p)
                                        io.model_ready(p,'det',[('x',[1,3,32,32],bytes(12288))])
                                elif meta['op']=='model.run': io.model_complete(io.session.running,[(bytes(4),)])
                                elif meta['op']=='model.resume':
                                    p=next(p for p in io.session._pages.values() if p.call and p.call.meta['call_id']==meta['call_id'])
                                    io.session.release_call(p); io.session.finish(p,encoded(dict(synthetic=True,image_id=p.image_id)))
                        try: function,future=self.commands.get_nowait()
                        except queue.Empty: continue
                        try: future.set_result(function(io))
                        except BaseException as error: future.set_exception(error)
                finally:
                    io.stop()
                    # All fixture execution is simulated; no CUDA work exists to wait for.
                    if io.session.running:
                        batch=io.session.running; io.session.complete(batch,[() for _ in batch.calls])
                    for page in list(io.session._pages.values()):
                        if page.active: io.session.retire(page)
                    io.close()
        except BaseException as error: self.errors.append(error); self.started.set()

    def __exit__(self,*_):
        self.quit.set(); self.thread.join(3)
        if self.thread.is_alive(): raise AssertionError('I/O thread did not exit')
        if self.errors: raise self.errors[0]


class Clients:
    def __init__(self, path): self.path=path; self.sockets={}; self.ids={}; self.session=None

    def connect(self, role, *, session=None):
        sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); sock.settimeout(2); sock.connect(str(self.path))
        hello=wire.receive_frame(sock,time.monotonic()+2,validator=contract.validate_stage)
        nonce=json.loads(hello.metadata)['session']
        if self.session is None: self.session=nonce
        if session is None: session=nonce
        meta=dict(schema='ocr.stage/1',op='channel.bind',session=session,remaining_ms=1000,channel=role)
        wire.send_frame(sock,wire.Frame(1,encoded(meta),b''),time.monotonic()+2,validator=contract.validate_stage)
        try:
            response=wire.receive_frame(sock,time.monotonic()+2,validator=contract.validate_stage)
            assert json.loads(response.metadata)['channel']==role and response.request_id==1
        except BaseException: sock.close(); raise
        self.sockets[role]=sock; self.ids[role]=1; return sock

    def all(self):
        for role in ('input','gpu','result','events'): self.connect(role)
        return self

    def frame(self, role, op, payload=b'', **fields):
        self.ids[role]+=1
        meta=dict(schema='ocr.stage/1',op=op,session=self.session,remaining_ms=1000,**fields)
        return wire.Frame(self.ids[role],encoded(meta),payload)

    def send(self, role, op, payload=b'', **fields):
        frame=self.frame(role,op,payload,**fields)
        wire.send_frame(self.sockets[role],frame,time.monotonic()+2,validator=contract.validate_stage)
        return frame

    def rpc(self, role, op, payload=b'', **fields):
        request=self.send(role,op,payload,**fields)
        frame=wire.receive_frame(self.sockets[role],time.monotonic()+2,validator=contract.validate_stage)
        assert frame.request_id==request.request_id
        return json.loads(frame.metadata),frame.payload

    def close(self):
        for sock in self.sockets.values(): sock.close()


@unittest.skipUnless(sys.platform=='linux','AF_UNIX peer PID and selector integration')
class ChannelTests(unittest.TestCase):
    def run_fixture(self, function, **options):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            with Harness(folder,**options) as h:
                c=Clients(h.path)
                try: function(h,c)
                finally: c.close()
            self.assertFalse(h.path.exists())

    def test_bind_roles_pid_watermarks_and_idle_connections(self):
        def check(h,c):
            c.all(); time.sleep(.08)
            self.assertTrue(h.call(lambda io:io.ready))
            self.assertEqual(h.call(lambda io:io.pid),os.getpid())
            result,_=c.rpc('input','image.submit',b'x',image_id='a',image_type=3,format='encoded')
            self.assertEqual(result['code'],'OK')
            result,_=c.rpc('gpu','model.run',call_ids=[99]); self.assertEqual(result['code'],'STATE')
            result,_=c.rpc('result','image.cancel',image_id='a'); self.assertEqual(result['code'],'OK')
            result,_=c.rpc('result','image.collect',image_id='a'); self.assertEqual(result['code'],'CANCELLED')
            self.assertEqual(h.call(lambda io:io.session.stats()['images']),0)
        self.run_fixture(check)

    def test_duplicate_and_stale_binding_do_not_replace_valid_role(self):
        def check(h,c):
            c.connect('input'); other=Clients(h.path)
            with self.assertRaises(wire.PeerClosed): other.connect('input')
            with self.assertRaises(wire.PeerClosed): other.connect('gpu',session='f'*32)
            self.assertEqual(h.call(lambda io:tuple(io.roles)),('input',))
            for role in ('gpu','result','events'): c.connect(role)
            self.assertTrue(h.call(lambda io:io.ready))
        self.run_fixture(check)

    def test_different_process_rejected_before_hello(self):
        def check(h,c):
            c.connect('input')
            code="import socket,sys; s=socket.socket(socket.AF_UNIX); s.settimeout(2); s.connect(sys.argv[1]); assert s.recv(24)==b''; s.close()"
            child=subprocess.run([sys.executable,'-B','-c',code,str(h.path)],capture_output=True,text=True,timeout=3)
            self.assertEqual(child.returncode,0,child.stderr)
            self.assertEqual(h.call(lambda io:len(io.roles)),1)
        self.run_fixture(check)

    def test_partial_bind_times_out_without_running_work(self):
        def check(h,c):
            c.connect('input'); time.sleep(.13)
            stats=h.call(lambda io:io.stats())
            self.assertTrue(stats['stopped']); self.assertEqual(stats['reason'],'BIND_TIMEOUT')
            self.assertEqual(stats['transport_bytes'],0)
        self.run_fixture(check,seconds=.08)

    def test_slow_half_frame_does_not_block_gpu_channel(self):
        def check(h,c):
            c.all()
            frame=c.frame('input','image.submit',b'abc',image_id='a',image_type=3,format='encoded')
            raw=wire.header(frame.request_id,len(frame.metadata),3)+frame.metadata+b'abc'
            c.sockets['input'].sendall(raw[:7])
            result,_=c.rpc('gpu','model.run',call_ids=[99]); self.assertEqual(result['code'],'STATE')
            c.sockets['input'].sendall(raw[7:])
            result=wire.receive_frame(c.sockets['input'],time.monotonic()+2,validator=contract.validate_stage)
            self.assertEqual(json.loads(result.metadata)['code'],'OK')
        self.run_fixture(check)

    def test_event_queue_backpressure_does_not_consume_ready_state(self):
        def check(h,c):
            c.all(); c.rpc('input','image.submit',b'x',image_id='a',image_type=3,format='encoded')
            def fill(io):
                p=io.session.page('a'); io.session.activate(p)
                for _ in range(4): io.emit(io.session.meta('model.done',call_ids=[99],output_bytes=[0]))
                before=io.event_id
                with self.assertRaisesRegex(StateError,'BUSY'): io.model_ready(p,'det',[('x',[1,3,32,32],bytes(12288))])
                self.assertIsNone(p.call); self.assertEqual(io.event_id,before)
                return len(io.roles['events'].out)
            self.assertEqual(h.call(fill),4)
        self.run_fixture(check)

    def test_stalled_result_send_preserves_credit_and_other_channel_progress(self):
        def check(h,c):
            c.all(); c.rpc('input','image.submit',b'x',image_id='a',image_type=3,format='encoded')
            def finish(io):
                p=io.session.page('a'); io.session.activate(p)
                io.session.finish(p,b'{}'+b' '*(1024*1024))
                io.roles['result'].sock.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,4096)
            h.call(finish); c.send('result','image.collect',image_id='a')
            result,_=c.rpc('gpu','model.run',call_ids=[99]); self.assertEqual(result['code'],'STATE')
            self.assertGreater(h.call(lambda io:io.session.stats()['result_bytes']),0)
            time.sleep(.25)
            stats=h.call(lambda io:io.stats())
            self.assertTrue(stats['stopped']); self.assertEqual(stats['transport_bytes'],0)
            self.assertEqual(h.call(lambda io:io.session.stats()['result_bytes']),0)
        self.run_fixture(check,seconds=.18)

    def test_disconnect_retains_running_page_until_safe_complete_and_retire(self):
        def check(h,c):
            c.all(); c.rpc('input','image.submit',b'x',image_id='a',image_type=3,format='encoded')
            def ready(io):
                p=io.session.page('a'); io.session.activate(p)
                return io.model_ready(p,'det',[('x',[1,3,32,32],bytes(12288))])
            meta=h.call(ready)
            wire.receive_frame(c.sockets['events'],time.monotonic()+2,validator=contract.validate_stage)
            result,_=c.rpc('gpu','model.run',call_ids=[meta['call_id']]); self.assertEqual(result['code'],'OK')
            c.sockets['events'].close(); time.sleep(.04)
            def inspect(io):
                self.assertTrue(io.stopped); self.assertGreater(io.session.stats()['active_reserved'],0)
                with self.assertRaises(StateError): io.close()
                p=io.session.page('a'); io.model_complete(io.session.running,[()]); io.session.retire(p)
                return io.session.stats()['images']
            self.assertEqual(h.call(inspect),0)
        self.run_fixture(check)

    @unittest.skipUnless(os.environ.get('AUTOOCR_STAGE_PROBE'),'Compiled C++ stage client required')
    def test_real_four_connection_cpp_roundtrip(self):
        def check(h,c):
            child=subprocess.run([os.environ['AUTOOCR_STAGE_PROBE'],'--channels',str(h.path)],capture_output=True,text=True,timeout=5)
            self.assertEqual(child.returncode,0,child.stderr)
            result=json.loads(child.stdout)
            self.assertEqual(result['status'],'cpp_python_channels_passed_not_ocr')
            self.assertEqual(result['channels'],4); self.assertEqual(result['requests'],9)
            self.assertEqual(h.call(lambda io:io.session.stats()['images']),0)
        self.run_fixture(check,auto=True)


if __name__=='__main__': unittest.main()
