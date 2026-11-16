import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import ipc_wire as wire
import stage_contract as contract
from stage_session import Session, StateError, encoded


class Clock:
    value=10.0
    def __call__(self): return self.value


class SessionTests(unittest.TestCase):
    def test_encoded_and_raw_format_preserved_for_algorithm_boundary(self):
        s=Session()
        for i,extra,payload in ((1,dict(image_id='a',image_type=3,format='encoded'),b'x'),
                                (2,dict(image_id='b',image_type=3,format='bgr8',shape=[1,1,3],stride=3),b'abc')):
            frame=wire.Frame(i,encoded(s.meta('image.submit',**extra)),payload)
            self.assertEqual(json.loads(s.handle(frame,'input').frame.metadata)['code'],'OK')
            self.assertEqual(s.page(extra['image_id']).format,extra['format'])

    def setUp(self):
        self.clock=Clock(); self.s=Session(clock=self.clock)
        self.ids={channel:0 for channel in ('input','gpu','result')}

    def send(self, op, payload=b'', **fields):
        channel='input' if op in ('image.submit','image.cancel') else ('gpu' if op=='model.run' else 'result')
        self.ids[channel]+=1
        meta=self.s.meta(op,**fields)
        reply=self.s.handle(wire.Frame(self.ids[channel],encoded(meta),payload),channel)
        contract.validate_stage(reply.frame.metadata,len(reply.frame.payload))
        return reply

    def code(self, reply): return json.loads(reply.frame.metadata)['code']

    def page(self, name='a', remaining=300000):
        meta=self.s.meta('image.submit',image_id=name,image_type=3,format='encoded')
        meta['remaining_ms']=remaining; self.ids['input']+=1
        reply=self.s.handle(wire.Frame(self.ids['input'],encoded(meta),b'synthetic'),'input')
        self.assertEqual(self.code(reply),'OK')
        return self.s.page(name)

    def ready(self, page):
        self.s.activate(page)
        return self.s.ready(page,'det',[('x',[1,3,32,32],bytes(12288))])

    def start_batch(self, *ready):
        self.assertEqual(self.code(self.send('model.run',call_ids=[m['call_id'] for m in ready])),'OK')
        return self.s.running

    def resume(self, page):
        call_id=page.call.meta['call_id']
        self.assertEqual(self.code(self.send('model.resume',call_id=call_id)),'OK')
        self.s.release_call(page)

    def test_hello_and_full_call_result_delivery_lifetime(self):
        contract.validate_stage(encoded(self.s.hello()),0)
        p=self.page(); meta=self.ready(p); batch=self.start_batch(meta)
        done=self.s.complete(batch,[(b'out',)]); contract.validate_stage(encoded(done),0)
        self.assertGreater(self.s.stats()['active_reserved'],0)
        self.resume(p)
        # A second call on the same page gets new call and buffer IDs.
        another=self.s.ready(p,'det',[('x',[1,3,32,32],bytes(12288))])
        self.assertGreater(another['call_id'],meta['call_id'])
        self.assertGreater(another['inputs'][0]['handle'],meta['inputs'][0]['handle'])
        self.s.complete(self.start_batch(another),[(b'next',)]); self.resume(p)
        self.s.finish(p,b'{"synthetic":true}')
        self.assertEqual(self.s.stats()['active_reserved'],0)
        reply=self.send('image.collect',image_id='a')
        self.assertEqual(reply.frame.payload,b'{"synthetic":true}')
        self.assertGreater(self.s.stats()['result_bytes'],0)
        self.assertEqual(self.code(self.send('image.collect',image_id='a')),'STATE')
        self.s.delivered(reply)
        self.assertEqual(self.s.stats()['registered_bytes'],0)
        self.assertEqual(self.s.stats()['images'],0)
        with self.assertRaises(StateError): self.s.delivered(reply)

    def test_replay_per_channel_and_stale_session(self):
        p=self.page()
        meta=self.s.meta('image.cancel',image_id=p.image_id)
        duplicate=wire.Frame(self.ids['input'],encoded(meta),b'')
        self.assertEqual(self.code(self.s.handle(duplicate,'input')),'STATE')
        meta['session']='b'*32
        self.assertEqual(self.code(self.s.handle(wire.Frame(999,encoded(meta),b''),'input')),'SESSION')
        self.assertEqual(self.code(self.send('image.cancel',image_id='a')),'OK')
        # Result IDs start from 1 independently of the input channel.
        reply=self.send('image.collect',image_id='a'); self.s.delivered(reply)

    def test_inflight_active_and_ingress_backpressure(self):
        pages=[self.page(str(i)) for i in range(4)]
        self.assertEqual(self.code(self.send('image.submit',b'x',image_id='extra',image_type=3,format='encoded')),'BUSY')
        self.s.activate(pages[0]); self.s.activate(pages[1])
        with self.assertRaisesRegex(StateError,'BUSY'): self.s.activate(pages[2])
        self.assertEqual(self.s.stats()['active_reserved'],2*contract.LIMITS['page_reservation_bytes'])
        self.s.disconnect()
        for p in pages[:2]: self.s.retire(p)
        self.assertEqual(self.s.stats()['images'],0)
        with patch.dict(contract.LIMITS,{'input_queue_bytes':3}):
            self.s=Session(clock=self.clock)
            self.assertEqual(self.code(self.send('image.submit',b'four',image_id='a',image_type=3,format='encoded')),'BUSY')
            self.assertEqual(self.s.stats()['images'],0)

    def test_registry_page_and_ready_byte_budgets(self):
        p=self.page(); self.s.activate(p)
        tensors=[('x',[1,3,32,32],bytes(12288))]
        for limit in ({'page_reservation_bytes':100},{'gpu_queue_bytes':100}):
            with patch.dict(contract.LIMITS,limit), self.assertRaises(StateError): self.s.ready(p,'det',tensors)
            self.assertIsNone(p.call)
        meta=self.s.ready(p,'det',tensors); meta['inputs'][0]['shape'][2]=960
        self.assertEqual(p.call.meta['inputs'][0]['shape'],[1,3,32,32])
        with self.assertRaises(StateError): self.s.ready(p,'det',tensors)

    def test_cancel_queued_and_active_need_cpu_retirement(self):
        queued=self.page('queued')
        self.assertEqual(self.code(self.send('image.cancel',image_id='queued')),'OK')
        self.assertEqual(self.s.stats()['ingress_bytes'],0)
        self.s.delivered(self.send('image.collect',image_id='queued'))
        active=self.page('active'); self.ready(active)
        self.send('image.cancel',image_id='active')
        self.assertGreater(self.s.stats()['active_reserved'],0)
        self.assertEqual(self.code(self.send('image.collect',image_id='active')),'STATE')
        self.s.retire(active)
        reply=self.send('image.collect',image_id='active')
        self.assertEqual(self.code(reply),'CANCELLED'); self.s.delivered(reply)

    def test_cancel_running_keeps_inputs_until_safe_completion(self):
        p=self.page(); batch=self.start_batch(self.ready(p))
        before=self.s.stats()['registered_bytes']
        self.send('image.cancel',image_id='a')
        self.assertEqual(self.s.stats()['registered_bytes'],before)
        with self.assertRaises(StateError): self.s.retire(p)
        self.assertIsNone(self.s.complete(batch,[(b'late output',)]))
        self.assertIsNone(p.call); self.assertGreater(self.s.stats()['active_reserved'],0)
        self.s.retire(p); self.s.delivered(self.send('image.collect',image_id='a'))
        self.assertEqual(self.s.stats()['registered_bytes'],0)

    def test_model_timeout_does_not_release_or_extend_page_deadline(self):
        p=self.page(); original_deadline=p.deadline
        batch=self.start_batch(self.ready(p)); self.assertLess(batch.deadline,p.deadline)
        self.clock.value=batch.deadline; self.s.expire()
        self.assertEqual(p.error,'TIMEOUT'); self.assertIs(self.s.running,batch)
        self.assertEqual(p.deadline,original_deadline)
        with self.assertRaises(StateError): self.s.retire(p)
        self.s.complete(batch,[()]); self.s.retire(p)
        self.assertEqual(self.code(self.send('image.collect',image_id='a')),'TIMEOUT')

    def test_page_deadline_queued_and_short_run_deadline(self):
        p=self.page(remaining=1000); self.clock.value=p.deadline
        with self.assertRaisesRegex(StateError,'TIMEOUT'): self.s.activate(p)
        self.s.delivered(self.send('image.collect',image_id='a'))
        p=self.page('b'); meta=self.ready(p)
        raw=self.s.meta('model.run',call_ids=[meta['call_id']]); raw['remaining_ms']=10
        self.ids['gpu']+=1
        self.s.handle(wire.Frame(self.ids['gpu'],encoded(raw),b''),'gpu')
        self.assertAlmostEqual(self.s.running.deadline,self.clock.value+.01)

    def test_gpu_failure_latches_without_unsafe_release_or_retry(self):
        p=self.page(); batch=self.start_batch(self.ready(p)); q=self.page('q')
        event=self.s.gpu_fault(batch); contract.validate_stage(encoded(event),0)
        self.assertTrue(self.s.gpu_failed); self.assertIs(self.s.running,batch)
        with self.assertRaises(StateError): self.s.retire(p)
        self.assertEqual(self.code(self.send('image.submit',b'x',image_id='new',image_type=3,format='encoded')),'GPU_FAILED')
        self.assertEqual(self.code(self.send('model.run',call_ids=[1])),'GPU_FAILED')
        self.assertEqual(q.error,'GPU_FAILED')
        self.s.complete(batch,[()]); self.s.retire(p)
        self.assertTrue(self.s.gpu_failed)

    def test_disconnect_quarantines_running_and_then_drains(self):
        p=self.page(); batch=self.start_batch(self.ready(p)); self.page('q')
        self.s.disconnect()
        self.assertEqual(self.s.stats()['images'],1)
        self.assertGreater(self.s.stats()['registered_bytes'],0)
        self.assertEqual(self.code(self.send('image.collect',image_id='a')),'DISCONNECTED')
        self.s.complete(batch,[()]); self.s.retire(p)
        self.assertEqual(self.s.stats()['images'],0)
        self.assertNotEqual(Session().session,self.s.session)

    def test_no_forged_done_wrong_direction_or_duplicate_resume(self):
        p=self.page(); meta=self.ready(p)
        self.assertEqual(self.code(self.send('model.resume',call_id=meta['call_id'])),'STATE')
        for op,fields in (('model.done',dict(call_ids=[1],output_bytes=[1])),
                          ('model.failed',dict(call_ids=[1],code='GPU_FAILED'))):
            with self.assertRaises(wire.ProtocolError):
                self.s.handle(wire.Frame(1,encoded(self.s.meta(op,**fields)),b''),'result')
        self.s.complete(self.start_batch(meta),[(b'x',)])
        self.assertEqual(self.code(self.send('model.resume',call_id=meta['call_id'])),'OK')
        self.assertEqual(self.code(self.send('model.resume',call_id=meta['call_id'])),'STATE')
        with self.assertRaises(StateError): self.s.finish(p,b'{}')
        self.s.release_call(p)

    def test_batch_requires_opt_in_and_can_cancel_one_member(self):
        a,b=self.page('a'),self.page('b'); ma,mb=self.ready(a),self.ready(b)
        with self.assertRaises(wire.ProtocolError): self.send('model.run',call_ids=[ma['call_id'],mb['call_id']])
        self.assertIsNone(self.s.running)
        self.s.allow_det_batch=True; batch=self.start_batch(ma,mb)
        self.send('image.cancel',image_id='a')
        done=self.s.complete(batch,[(b'discard',),(b'keep',)])
        self.assertEqual(done['call_ids'],[mb['call_id']])
        self.s.retire(a); self.resume(b); self.s.finish(b,b'{}')

    def test_batch_identity_duplicate_completion_and_id_reuse_safety(self):
        p=self.page(); batch=self.start_batch(self.ready(p))
        other=Session(clock=self.clock)
        with self.assertRaises(StateError): other.complete(batch,[(b'x',)])
        self.s.complete(batch,[(b'x',)])
        with self.assertRaises(StateError): self.s.complete(batch,[(b'x',)])
        self.resume(p); self.s.finish(p,b'{}')
        self.s.delivered(self.send('image.collect',image_id='a'))
        new=self.page('a')
        with self.assertRaises(StateError): self.s.activate(p)
        self.assertIs(self.s.page('a'),new)

    def test_result_validation_quota_and_disconnect_during_delivery(self):
        p=self.page(); self.s.activate(p)
        for bad in (b'[]',b'{',b'{"x":NaN}',b'{"x":1e999}',b'{"x":1,"x":2}',b'\xff'):
            with self.assertRaises(StateError): self.s.finish(p,bad)
            self.assertTrue(p.active)
        with patch.dict(contract.LIMITS,{'result_queue_bytes':1}), self.assertRaisesRegex(StateError,'BUSY'):
            self.s.finish(p,b'{}')
        self.s.finish(p,b'{}'); reply=self.send('image.collect',image_id='a')
        self.assertEqual(self.s.stats()['result_bytes'],2)
        self.s.disconnect(); self.assertEqual(self.s.stats()['result_bytes'],2)
        with self.assertRaises(StateError): self.s.delivered(reply)
        self.s.abandon_delivery(reply); self.assertEqual(self.s.stats()['result_bytes'],0)
        with self.assertRaises(StateError): self.s.abandon_delivery(reply)

    def test_oversize_completed_output_retires_without_stuck_running(self):
        p=self.page(); batch=self.start_batch(self.ready(p))
        with patch.dict(contract.LIMITS,{'max_tensor_bytes':1}):
            self.assertIsNone(self.s.complete(batch,[(b'too large',)]))
        self.assertIsNone(self.s.running); self.assertEqual(p.error,'LIMIT')
        self.s.retire(p); self.assertEqual(self.s.stats()['registered_bytes'],0)

    def test_old_batch_token_does_not_keep_released_registry_bytes(self):
        p=self.page(); batch=self.start_batch(self.ready(p))
        self.s.complete(batch,[(b'output',)])
        self.resume(p)
        self.assertEqual(sum(map(len,batch.calls[0].inputs)),0)
        self.assertEqual(sum(map(len,batch.calls[0].outputs)),0)
        self.s.finish(p,b'{}')

    def test_old_delivery_token_does_not_keep_page_result(self):
        p=self.page(); self.s.activate(p); self.s.finish(p,b'{"synthetic":true}')
        reply=self.send('image.collect',image_id='a'); self.s.delivered(reply)
        self.assertEqual(len(p.result),0)

    def test_state_has_one_owner(self):
        errors=[]
        def foreign():
            try: self.s.stats()
            except StateError as error: errors.append(error.code)
        thread=threading.Thread(target=foreign); thread.start(); thread.join(2)
        self.assertFalse(thread.is_alive()); self.assertEqual(errors,['STATE'])

    def test_error_envelopes_are_strict_and_do_not_contain_exception_text(self):
        valid=self.s.meta('rpc.reply',for_op='image.collect',status='error',code='STATE')
        contract.validate_stage(encoded(valid),0)
        for changes in ({'code':'private traceback'},{'code':[]},{'message':'private text'},
                        {'for_op':'Extract'},{'status':'ok'}):
            with self.assertRaises(wire.ProtocolError): contract.validate_stage(encoded(dict(valid,**changes)),0)
        for meta in (self.s.meta('image.result',image_id='a'),
                     self.s.meta('image.result',image_id='a',status='error'),
                     self.s.meta('image.result',image_id='a',status='cancelled',code='STATE')):
            with self.assertRaises(wire.ProtocolError): contract.validate_stage(encoded(meta),0)

    @unittest.skipUnless(sys.platform=='linux','Private AF_UNIX listener and SO_PEERCRED')
    def test_real_private_unix_rpc_lifecycle_without_models(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path=Path(folder)/'stage.sock'
            with wire.private_listener(path) as listener:
                client=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); client.settimeout(2)
                server=None
                try:
                    client.connect(str(path)); listener.settimeout(2); server,_=listener.accept()
                    wire.check_peer(client); wire.check_peer(server)
                    def event(meta, event_id):
                        wire.send_frame(server,wire.Frame(event_id,encoded(meta),b''),time.monotonic()+2,validator=contract.validate_stage)
                        response=wire.receive_frame(client,time.monotonic()+2,validator=contract.validate_stage)
                        self.assertEqual(json.loads(response.metadata),meta)
                    def rpc(channel, request_id, op, payload=b'', **fields):
                        frame=wire.Frame(request_id,encoded(self.s.meta(op,**fields)),payload)
                        wire.send_frame(client,frame,time.monotonic()+2,validator=contract.validate_stage)
                        incoming=wire.receive_frame(server,time.monotonic()+2,validator=contract.validate_stage)
                        reply=self.s.handle(incoming,channel)
                        wire.send_frame(server,reply.frame,time.monotonic()+2,validator=contract.validate_stage)
                        response=wire.receive_frame(client,time.monotonic()+2,validator=contract.validate_stage)
                        self.assertEqual(response.request_id,request_id)
                        if reply.delivery:
                            self.assertEqual(json.loads(response.metadata)['image_id'],fields['image_id'])
                            self.assertEqual(response.payload,b'{"synthetic":true}')
                            self.s.delivered(reply)
                        else: self.assertEqual(json.loads(response.metadata)['code'],'OK')
                    event(self.s.hello(),1)
                    for request_id in (1,2):
                        rpc('input',request_id,'image.submit',b'synthetic',image_id=str(request_id),image_type=3,format='encoded')
                    for request_id in (2,1):
                        page=self.s.page(str(request_id)); meta=self.ready(page)
                        event(meta,request_id+1)
                        rpc('gpu',3-request_id,'model.run',call_ids=[meta['call_id']])
                        event(self.s.complete(self.s.running,[(b'synthetic output',)]),request_id+3)
                        rpc('result',3-request_id,'model.resume',call_id=meta['call_id'])
                        self.s.release_call(page); self.s.finish(page,b'{"synthetic":true}')
                    for request_id in (2,1):
                        rpc('result',5-request_id,'image.collect',image_id=str(request_id))
                    self.assertEqual(self.s.stats()['images'],0)
                    self.s.disconnect(); self.assertEqual(self.s.stats()['registered_bytes'],0)
                finally:
                    client.close()
                    if server: server.close()
            self.assertFalse(path.exists())


if __name__=='__main__': unittest.main()
