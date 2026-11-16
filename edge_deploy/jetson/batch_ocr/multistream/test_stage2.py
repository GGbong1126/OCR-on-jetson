import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'ipc'))
import ipc_wire as wire
from stage_session import encoded, StateError
from worker_bridge import Tensor
from stage2 import MultiSession, to_wire, validate


class SessionTests(unittest.TestCase):
    def setUp(self): self.s=MultiSession(4); self.ids={}
    def send(self,channel,op,body=b'',**fields):
        self.ids[channel]=self.ids.get(channel,0)+1
        m=to_wire(self.s.meta(op,**fields))
        return self.s.handle(wire.Frame(self.ids[channel],encoded(m),body),channel)
    def code(self,r): return wire.decode_metadata(r.frame.metadata)['code']
    def page(self,name='a'):
        self.assertEqual(self.code(self.send('input','image.submit',b'x',image_id=name,image_type=3,format='encoded')),'OK')
        p=self.s.page(name); self.s.activate(p); return p
    def ready(self,page,index=0,width=32):
        return self.s.ready_rec(page,(Tensor('x',(1,3,48,width),bytes(576*width)),),1,index,index)
    def start(self,m,slot): return self.send('gpu','model.run',call_ids=[m['call_id']],slot=slot)
    def test_explicit_version_and_capabilities(self):
        m=self.s.hello(); self.assertEqual(validate(encoded(m),0)['rec_slots'],4)
        with self.assertRaises(wire.ProtocolError): validate(encoded(dict(m,schema='ocr.stage/1')),0)
        for key,value in (('rec_slots',3),('rec_slots',True),('rec_queue_items',9),('rec_queue_bytes',0)):
            with self.assertRaises(wire.ProtocolError): validate(encoded(dict(m,**{key:value})),0)
    def test_multiple_crop_calls_same_page_reverse_completion(self):
        p=self.page(); a=self.ready(p); b=self.ready(p,1,160)
        self.assertEqual(self.code(self.start(a,1)),'OK'); self.assertEqual(self.code(self.start(b,2)),'OK')
        for m,slot in ((b,2),(a,1)):
            done=self.s.complete_rec(slot,(b'output',)); validate(encoded(to_wire(done)),0)
            self.assertEqual(self.code(self.send('result','model.resume',call_id=m['call_id'])),'OK')
            event=self.s.release_rec(m['call_id']); validate(encoded(to_wire(event)),0)
        self.s.finish(p,b'{}'); reply=self.send('result','image.collect',image_id=p.image_id); self.s.delivered(reply)
        self.assertEqual(self.s.stats()['registered_bytes'],0); self.assertFalse(self.s.rec_calls)
    def test_slot_credit_until_cpu_consumed(self):
        p=self.page(); a=self.ready(p); b=self.ready(p,1)
        self.start(a,1); self.s.complete_rec(1,(b'ok',))
        self.assertEqual(self.code(self.start(b,1)),'BUSY')
        with self.assertRaises(StateError): self.s.release_rec(a['call_id'])
        self.send('result','model.resume',call_id=a['call_id']); self.s.release_rec(a['call_id'])
        self.assertEqual(self.code(self.start(b,1)),'OK')
    def test_queue_count_bytes_and_unchanged_absolute_deadline(self):
        p=self.page()
        for i in range(8): self.ready(p,i)
        with self.assertRaises(StateError): self.ready(p,8)
        self.assertEqual(p.deadline,self.s.page('a').deadline)
    def test_duplicate_crop_old_id_and_wrong_route(self):
        p=self.page(); a=self.ready(p)
        with self.assertRaises(StateError): self.ready(p)
        self.assertEqual(self.code(self.start(a,5)),'STATE')
        bad=to_wire(dict(a,route='cpu'))
        with self.assertRaises(wire.ProtocolError): validate(encoded(bad),0)
        self.assertEqual(self.code(self.send('gpu','model.run',call_ids=[999],slot=1)),'STATE')
    def test_cancel_does_not_release_running_crop(self):
        p=self.page(); a=self.ready(p); self.start(a,1)
        self.send('input','image.cancel',image_id='a')
        with self.assertRaises(StateError): self.s.release_rec(a['call_id'])
        with self.assertRaises(StateError): self.s.retire(p)
        self.assertIsNone(self.s.complete_rec(1,()))
        self.s.release_rec(a['call_id']); self.s.retire(p)
        r=self.send('result','image.collect',image_id='a'); self.s.delivered(r)
        self.assertEqual(self.s.stats()['registered_bytes'],0)
    def test_legacy_call_and_rec_overlap_distinct_lifetimes(self):
        p=self.page(); q=self.page('b'); r=self.ready(p); self.start(r,1)
        m=self.s.ready(q,'det',[('x',[1,3,32,32],bytes(12288))])
        self.assertEqual(self.code(self.send('gpu','model.run',call_ids=[m['call_id']])),'OK')
        self.s.complete(self.s.running,[(b'det',)])
        self.assertTrue(self.s.rec_running); self.assertIsNone(self.s.running)
    def test_cpu_range_preserved(self):
        p=self.page(); a=self.ready(p,width=2305)
        self.assertEqual(a['route'],'cpu')
        self.assertEqual(self.code(self.start(a,1)),'STATE')
        self.assertEqual(self.code(self.start(a,5)),'OK')

    def test_deadline_retains_submitted_buffer_until_completion(self):
        clock=[100.]; self.s=MultiSession(4,clock=lambda:clock[0])
        p=self.page(); a=self.ready(p); self.start(a,1)
        deadline=self.s.rec_running[1].deadline; clock[0]=deadline+.001; self.s.expire()
        self.assertEqual(p.error,'TIMEOUT')
        with self.assertRaises(StateError): self.s.release_rec(a['call_id'])
        self.s.complete_rec(1,()); self.s.release_rec(a['call_id']); self.s.retire(p)
        self.assertEqual(self.s.stats()['registered_bytes'],0)


if __name__=='__main__': unittest.main()
