"""Explicit stage/2 candidate; original stage/1 state machine remains the baseline."""
import copy
import time

import ipc_wire as wire
import stage_contract as old
from channel_io import ChannelServer
from stage_session import Session, Call, Batch, Reply, encoded, require

SCHEMA='ocr.stage/2'
REC_EXTRA={'group_id','crop_id','box_index','route'}
QUEUE_ITEMS=8
QUEUE_BYTES=16*1024**2
PAGE_BYTES=128*1024**2


def route(width): return 'narrow' if width<=1536 else ('wide' if width<=2304 else 'cpu')


def _validate(raw,n):
    m=wire.decode_metadata(raw); old.need(m.get('schema')==SCHEMA)
    base=dict(m,schema='ocr.stage/1'); op=m.get('op')
    if op=='rec.consumed':
        old.need(set(m)==old.COMMON|{'call_id','slot'} and n==0)
        old.need(old.integer(m['slot'],1,5)); base.pop('slot'); base['op']='model.resume'
    elif op=='session.ready':
        old.need(set(m)==old.COMMON|{'rec_slots','rec_queue_items','rec_queue_bytes'})
        old.need(type(m['rec_slots']) is int and m['rec_slots'] in (1,2,4))
        old.need(type(m['rec_queue_items']) is int and m['rec_queue_items']==QUEUE_ITEMS)
        old.need(type(m['rec_queue_bytes']) is int and m['rec_queue_bytes']==QUEUE_BYTES)
        for k in ('rec_slots','rec_queue_items','rec_queue_bytes'): base.pop(k)
    elif op=='model.ready' and m.get('model')=='rec':
        old.need(set(m)==old.COMMON|{'image_id','image_type','call_id','model','inputs'}|REC_EXTRA)
        old.need(old.integer(m['group_id'],1,wire.MAX_ID) and old.integer(m['crop_id'],0,wire.MAX_ID))
        old.need(old.integer(m['box_index'],0,wire.MAX_ID))
        old.need(m['route']==route(m['inputs'][0]['shape'][3]))
        for k in REC_EXTRA: base.pop(k)
    elif op=='model.run' and 'slot' in m:
        old.need(old.integer(m['slot'],1,5) and len(m.get('call_ids',[]))==1); base.pop('slot')
    old.validate_stage(encoded(base),n)
    return m


def validate(raw,n):
    try: return _validate(raw,n)
    except (KeyError,IndexError,TypeError) as error:
        raise wire.ProtocolError('Invalid stage/2 fields') from error


def to_wire(m): return dict(m,schema=SCHEMA)


class MultiSession(Session):
    def __init__(self,rec_slots,**kwargs):
        require(type(rec_slots) is int and rec_slots in (1,2,4))
        super().__init__(**kwargs); self.rec_slots=rec_slots; self.rec_calls={}; self.rec_running={}; self.rec_occupied={}
        self.rec_seen={}

    def hello(self):
        self._owned(); require(self.connected,'DISCONNECTED')
        return to_wire(self.meta('session.ready',rec_slots=self.rec_slots,rec_queue_items=QUEUE_ITEMS,rec_queue_bytes=QUEUE_BYTES))

    def stats(self):
        d=super().stats()
        d.update(rec_tasks=len(self.rec_calls),rec_running=len(self.rec_running),rec_slots=self.rec_slots)
        d['active_reserved']=d['active']*PAGE_BYTES
        d['static_io_reservation']=240*1024**2
        d['ready_bytes']+=sum(sum(map(len,c.inputs)) for c in self.rec_calls.values() if c.state=='ready')
        # super.stats invokes our _page_bytes, so rec bytes are not counted twice.
        return d

    def _page_bytes(self,page):
        return super()._page_bytes(page)+sum(sum(map(len,c.inputs+c.outputs)) for c in self.rec_calls.values() if c.page is page)

    def _stop(self,page,code):
        super()._stop(page,code)
        for c in list(self.rec_calls.values()):
            if c.page is page and c.state!='running': c.inputs=c.outputs=()

    def ready(self,page,model,tensors):
        require(self._page_bytes(page)+sum(len(data) for _,_,data in tensors)<=PAGE_BYTES,'LIMIT')
        return super().ready(page,model,tensors)

    def complete(self,batch,outputs):
        for c,items in zip(batch.calls,outputs):
            if self._page_bytes(c.page)+sum(map(len,items))>PAGE_BYTES: self._stop(c.page,'LIMIT')
        return super().complete(batch,outputs)

    def expire(self):
        super().expire()
        for batch in list(self.rec_running.values()):
            if self._clock()>=batch.deadline:
                for c in batch.calls: self._stop(c.page,'TIMEOUT')

    def ready_rec(self,page,tensors,group_id,crop_id,box_index):
        self._owned(); self.expire(); self._page(page)
        require(page.active and page.state=='active' and page.call is None and not page.error)
        waiting=[c for c in self.rec_calls.values() if c.state=='ready']
        require(len(waiting)<QUEUE_ITEMS,'BUSY')
        size=sum(len(t.data) for t in tensors)
        require(sum(sum(map(len,c.inputs)) for c in waiting)+size<=QUEUE_BYTES,'BUSY')
        require(self._page_bytes(page)+size<=PAGE_BYTES,'LIMIT')
        require(len(tensors)==1 and tensors[0].name=='x' and tensors[0].dtype=='float32')
        require(self._next_call<=wire.MAX_ID and self._next_handle<=wire.MAX_ID,'LIMIT')
        seen=self.rec_seen.setdefault(page.image_id,set())
        require((group_id,crop_id) not in seen); require(len(seen)<65536,'LIMIT')
        t=tensors[0]
        m=self.meta('model.ready',image_id=page.image_id,image_type=page.image_type,call_id=self._next_call,
                    model='rec',group_id=group_id,crop_id=crop_id,box_index=box_index,route=route(t.shape[3]),
                    inputs=[dict(name='x',handle=self._next_handle,dtype='float32',shape=list(t.shape),bytes=len(t.data))])
        validate(encoded(to_wire(m)),0)
        self.rec_calls[self._next_call]=Call(page,m,(t.data,))
        seen.add((group_id,crop_id))
        self._next_call+=1; self._next_handle+=1
        return copy.deepcopy(m)

    def handle(self,frame,channel):
        self._owned(); m=validate(frame.metadata,len(frame.payload))
        require(m['session']==self.session,'SESSION')
        op=m['op']
        rec=(op=='model.run' and 'slot' in m) or (op=='model.resume' and m['call_id'] in self.rec_calls)
        if not rec:
            require(not (op=='model.run' and any(i in self.rec_calls for i in m['call_ids'])))
            reply=super().handle(wire.Frame(frame.request_id,encoded(dict(m,schema='ocr.stage/1')),frame.payload),channel)
            return Reply(wire.Frame(reply.frame.request_id,encoded(to_wire(wire.decode_metadata(reply.frame.metadata))),reply.frame.payload),reply.delivery)
        if channel!=('gpu' if op=='model.run' else 'result'): raise wire.ProtocolError('Wrong rec channel')
        try:
            require(self.connected,'DISCONNECTED'); require(frame.request_id>self._last[channel])
            self._last[channel]=frame.request_id; self.expire()
            if op=='model.run':
                require(not self.gpu_failed,'GPU_FAILED'); slot=m['slot']
                require(slot not in self.rec_occupied,'BUSY')
                c=self.rec_calls.get(m['call_ids'][0]); require(c is not None and c.state=='ready' and not c.page.error)
                require((slot==5) if c.meta['route']=='cpu' else 1<=slot<=self.rec_slots)
                deadline=min(c.page.deadline,self._clock()+min(m['remaining_ms'],old.LIMITS['model_timeout_ms'])/1000)
                self.rec_running[slot]=Batch((c,),deadline); self.rec_occupied[slot]=c; c.slot=slot; c.state='running'
            else:
                c=self.rec_calls[m['call_id']]; require(c.state=='done' and not c.page.error)
                c.state='resumed'
            answer=self.meta('rpc.reply',for_op=op,status='ok',code='OK')
        except Exception as error:
            from stage_session import StateError
            if not isinstance(error,StateError): raise
            answer=self.meta('rpc.reply',for_op=op,status='error',code=error.code)
        return Reply(wire.Frame(frame.request_id,encoded(to_wire(answer)),b''))

    def complete_rec(self,slot,outputs):
        self._owned(); self.expire(); batch=self.rec_running[slot]; c=batch.calls[0]
        require(type(outputs) is tuple and all(type(x) is bytes for x in outputs))
        size=sum(map(len,outputs))
        require(size<=old.LIMITS['max_tensor_bytes'],'LIMIT')
        if self._page_bytes(c.page)+size>PAGE_BYTES: self._stop(c.page,'LIMIT')
        del self.rec_running[slot]
        if c.page.error: c.inputs=c.outputs=(); c.state='cancelled'; return None
        c.outputs=outputs; c.state='done'
        return self.meta('model.done',call_ids=[c.meta['call_id']],output_bytes=[size])

    def release_rec(self,call_id):
        self._owned(); c=self.rec_calls[call_id]
        require(c.state in ('resumed','cancelled') or (c.page.error and c.state!='running'))
        c.inputs=c.outputs=(); del self.rec_calls[call_id]
        slot=getattr(c,'slot',None)
        if slot is not None:
            require(self.rec_occupied.pop(slot) is c)
            return self.meta('rec.consumed',call_id=call_id,slot=slot)

    def finish(self,page,result):
        require(not any(c.page is page for c in self.rec_calls.values()))
        value=super().finish(page,result); self.rec_seen.pop(page.image_id,None); return value

    def retire(self,page):
        require(not any(c.page is page for c in self.rec_calls.values()))
        value=super().retire(page); self.rec_seen.pop(page.image_id,None); return value


class MultiChannel(ChannelServer):
    def __init__(self,listener,rec_slots,**kwargs):
        super().__init__(listener,session=MultiSession(rec_slots,allow_det_batch=kwargs.get('allow_det_batch',False)),validator=validate,**kwargs)

    def queue(self,peer,frame,delivery=None):
        m=wire.decode_metadata(frame.metadata)
        super().queue(peer,wire.Frame(frame.request_id,encoded(to_wire(m)),frame.payload),delivery)

    def emit(self,meta):
        if meta['op']!='rec.consumed': return super().emit(meta)
        self.session._owned(); self.event_room()
        self.queue(self.roles['events'],wire.Frame(self.event_id+1,encoded(to_wire(meta)),b'')); self.event_id+=1
