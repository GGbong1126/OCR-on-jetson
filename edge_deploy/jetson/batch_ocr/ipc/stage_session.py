"""Bounded lifecycle core, owned by one I/O thread; no model or HTTP execution."""
import copy
from dataclasses import dataclass
import json
import secrets
import threading
import time

import ipc_wire as wire
from stage_contract import LIMITS, validate_result, validate_run, validate_stage

CHANNELS = {'input':{'image.submit','image.cancel'}, 'gpu':{'model.run'},
            'result':{'model.resume','image.collect','image.cancel'}}


class StateError(RuntimeError):
    def __init__(self, code):
        super().__init__(code)
        self.code=code


def require(condition, code='STATE'):
    if not condition: raise StateError(code)


def encoded(meta): return json.dumps(meta,separators=(',',':'),allow_nan=False).encode()


@dataclass(eq=False)
class Page:
    image_id: str
    image_type: int
    deadline: float
    payload: bytes
    state: str='queued'
    active: bool=False
    call: object=None
    result: bytes=b''
    error: str=''
    format: str='encoded'
    shape: tuple=()
    stride: int=0


@dataclass(eq=False)
class Call:
    page: Page
    meta: dict
    inputs: tuple
    outputs: tuple=()
    state: str='ready'


@dataclass(eq=False)
class Batch:
    calls: tuple
    deadline: float


@dataclass(frozen=True)
class Reply:
    frame: wire.Frame
    delivery: object=None


class Session:
    def __init__(self, *, clock=time.monotonic, allow_det_batch=False):
        self.session=secrets.token_hex(16)
        self._clock=clock
        self._owner=threading.get_ident()
        self._pages={}
        self._last={channel:0 for channel in CHANNELS}
        self._next_call=self._next_handle=1
        self.running=None
        self.connected=True
        self.gpu_failed=False
        self.allow_det_batch=allow_det_batch is True

    def _owned(self):
        require(threading.get_ident()==self._owner)

    def _page(self, page):
        require(self._pages.get(page.image_id) is page)

    def meta(self, op, **fields):
        return dict(schema='ocr.stage/1',op=op,session=self.session,
                    remaining_ms=LIMITS['image_timeout_ms'],**fields)

    def hello(self):
        self._owned(); require(self.connected,'DISCONNECTED')
        return self.meta('session.ready')

    def bind_channel(self, channel):
        self._owned(); require(self.connected,'DISCONNECTED')
        require(channel in self._last and self._last[channel]==0)
        self._last[channel]=1

    def stats(self):
        self._owned()
        pages=list(self._pages.values())
        return dict(images=len(pages),active=sum(p.active for p in pages),
                    active_reserved=sum(p.active for p in pages)*LIMITS['page_reservation_bytes'],
                    ingress_bytes=sum(len(p.payload) for p in pages if not p.active),
                    result_bytes=sum(len(p.result) for p in pages),
                    ready_bytes=sum(sum(map(len,p.call.inputs)) for p in pages if p.call and p.call.state=='ready'),
                    registered_bytes=sum(self._page_bytes(p) for p in pages))

    @staticmethod
    def _page_bytes(page):
        return len(page.payload)+(sum(map(len,page.call.inputs+page.call.outputs)) if page.call else 0)

    def _drop_call(self, page):
        if page.call: page.call.inputs=page.call.outputs=()
        page.call=None

    def _stop(self, page, code):
        if page.state=='delivering': return
        if not page.error: page.error=code
        page.result=b''
        if page.call and page.call.state!='running': self._drop_call(page)
        if not page.active:
            page.payload=b''
            page.state='terminal'
        else:
            page.state='stopping'
        if not self.connected and not page.active: self._pages.pop(page.image_id)

    def expire(self):
        self._owned()
        now=self._clock()
        if self.running and now>=self.running.deadline:
            for call in self.running.calls: self._stop(call.page,'TIMEOUT')
        for page in list(self._pages.values()):
            if now>=page.deadline and page.state!='delivering': self._stop(page,'TIMEOUT')

    def handle(self, frame, channel):
        self._owned()
        wire.header(frame.request_id,len(frame.metadata),len(frame.payload))
        require(isinstance(frame.payload,bytes))
        meta=validate_stage(frame.metadata,len(frame.payload))
        if channel not in CHANNELS or meta['op'] not in CHANNELS[channel]:
            raise wire.ProtocolError('Wrong OCR control direction/channel')
        try:
            require(self.connected,'DISCONNECTED')
            require(meta['session']==self.session,'SESSION')
            require(frame.request_id>self._last[channel])
            self._last[channel]=frame.request_id
            self.expire()
            op=meta['op']
            if op=='image.submit':
                require(not self.gpu_failed,'GPU_FAILED')
                require(meta['image_id'] not in self._pages)
                stats=self.stats()
                require(stats['images']<LIMITS['max_inflight_images'],'BUSY')
                require(stats['ingress_bytes']+len(frame.payload)<=LIMITS['input_queue_bytes'],'BUSY')
                self._pages[meta['image_id']]=Page(meta['image_id'],meta['image_type'],
                    self._clock()+meta['remaining_ms']/1000,frame.payload,format=meta['format'],
                    shape=tuple(meta.get('shape',())),stride=meta.get('stride',0))
            elif op=='model.run':
                self._run(meta,frame.metadata)
            else:
                page=(next((p for p in self._pages.values() if p.call and p.call.meta['call_id']==meta['call_id']),None)
                      if op=='model.resume' else self._pages.get(meta['image_id']))
                require(page is not None)
                if op=='model.resume':
                    require(not page.error,page.error or 'STATE')
                    require(page.call.state=='done')
                    page.call.state='resumed'; page.state='active'
                elif op=='image.cancel':
                    require(page.state!='delivering')
                    self._stop(page,'CANCELLED')
                elif op=='image.collect':
                    require(not page.active and page.state in ('complete','terminal'))
                    status='cancelled' if page.error=='CANCELLED' else ('error' if page.error else 'ok')
                    fields=dict(image_id=page.image_id,status=status)
                    if page.error: fields['code']=page.error
                    body=page.result
                    response=self.meta('image.result',**fields)
                    validate_stage(encoded(response),len(body))
                    page.state='delivering'
                    return Reply(wire.Frame(frame.request_id,encoded(response),body),page)
            response=self.meta('rpc.reply',for_op=op,status='ok',code='OK')
        except StateError as error:
            response=self.meta('rpc.reply',for_op=meta['op'],status='error',code=error.code)
        validate_stage(encoded(response),0)
        return Reply(wire.Frame(frame.request_id,encoded(response),b''))

    def page(self, image_id):
        self._owned()
        return self._pages[image_id]

    def pages(self):
        self._owned()
        return tuple(self._pages.values())

    def fail_page(self, page, code):
        self._owned(); self._page(page)
        require(code in ('STATE','LIMIT','TIMEOUT','CANCELLED','GPU_FAILED','DISCONNECTED'))
        self._stop(page,code)

    def activate(self, page):
        self._owned(); self.expire(); self._page(page)
        require(self.connected,'DISCONNECTED'); require(not self.gpu_failed,'GPU_FAILED')
        require(not page.error,page.error or 'STATE')
        require(page.state=='queued')
        require(self.stats()['active']<LIMITS['max_active_images'],'BUSY')
        page.active=True; page.state='active'

    def ready(self, page, model, tensors):
        self._owned(); self.expire(); self._page(page)
        require(not page.error,page.error or 'STATE')
        require(page.active and page.state=='active' and page.call is None)
        # Immutable bytes are the current registry boundary; a NumPy adapter needs separate ownership tests.
        require(type(tensors) is list and 1<=len(tensors)<=3,'LIMIT')
        require(all(type(data) is bytes for _,_,data in tensors))
        require(self._next_call<=wire.MAX_ID and self._next_handle+len(tensors)-1<=wire.MAX_ID,'LIMIT')
        specs=[dict(name=name,handle=self._next_handle+i,dtype='float32',shape=list(shape),bytes=len(data))
               for i,(name,shape,data) in enumerate(tensors)]
        meta=self.meta('model.ready',image_id=page.image_id,image_type=page.image_type,
                       call_id=self._next_call,model=model,inputs=specs)
        validate_stage(encoded(meta),0)
        inputs=tuple(data for _,_,data in tensors); size=sum(map(len,inputs))
        require(self._page_bytes(page)+size<=LIMITS['page_reservation_bytes'],'LIMIT')
        require(self.stats()['ready_bytes']+size<=LIMITS['gpu_queue_bytes'],'BUSY')
        page.call=Call(page,meta,inputs); page.state='ready'
        self._next_call+=1; self._next_handle+=len(tensors)
        return copy.deepcopy(meta)

    def _run(self, meta, raw):
        require(not self.gpu_failed,'GPU_FAILED')
        require(self.running is None,'BUSY')
        calls={p.call.meta['call_id']:p.call for p in self._pages.values() if p.call}
        require(all(i in calls and calls[i].state=='ready' and not calls[i].page.error for i in meta['call_ids']))
        validate_run(raw,{i:c.meta for i,c in calls.items()},allow_det_batch=self.allow_det_batch)
        selected=tuple(calls[i] for i in meta['call_ids'])
        deadline=min(self._clock()+min(meta['remaining_ms'],LIMITS['model_timeout_ms'])/1000,
                     *(c.page.deadline for c in selected))
        self.running=Batch(selected,deadline)
        for call in selected: call.state='running'; call.page.state='running'

    def complete(self, batch, outputs):
        """Only an internal event after confirmed safe execution completion may call this."""
        self._owned(); self.expire()
        require(batch is self.running and batch is not None)
        require(type(outputs) is list and len(outputs)==len(batch.calls))
        require(all(type(items) is tuple and len(items)<=16 and all(type(v) is bytes for v in items) for items in outputs))
        for call,items in zip(batch.calls,outputs):
            size=sum(map(len,items))
            if size>LIMITS['max_tensor_bytes'] or self._page_bytes(call.page)+size>LIMITS['page_reservation_bytes']:
                self._stop(call.page,'LIMIT')
        self.running=None
        ids=[]; sizes=[]
        for call,items in zip(batch.calls,outputs):
            if call.page.error:
                self._drop_call(call.page)
            else:
                call.outputs=items; call.state='done'; call.page.state='done'
                ids.append(call.meta['call_id']); sizes.append(sum(map(len,items)))
        return self.meta('model.done',call_ids=ids,output_bytes=sizes) if ids else None

    def gpu_fault(self, batch):
        """Latch failure, but do not claim that failing CUDA synchronization freed buffers."""
        self._owned(); require(batch is self.running and batch is not None)
        self.gpu_failed=True
        for page in list(self._pages.values()): self._stop(page,'GPU_FAILED')
        return self.meta('model.failed',call_ids=[c.meta['call_id'] for c in batch.calls],code='GPU_FAILED')

    def release_call(self, page):
        self._owned(); self._page(page)
        require(page.call is not None and page.call.state=='resumed')
        self._drop_call(page)

    def finish(self, page, result):
        self._owned(); self.expire(); self._page(page)
        require(page.active and page.state=='active' and page.call is None and not page.error)
        require(type(result) is bytes and 1<=len(result)<=LIMITS['max_result_bytes'],'LIMIT')
        try:
            validate_result(result)
        except wire.ProtocolError as error:
            raise StateError('STATE') from error
        require(self.stats()['result_bytes']+len(result)<=LIMITS['result_queue_bytes'],'BUSY')
        page.payload=b''; page.result=result; page.active=False; page.state='complete'

    def retire(self, page):
        """CPU finalizer confirms it no longer uses this stopped page or its buffers."""
        self._owned(); self._page(page)
        require(page.active and bool(page.error))
        require(page.call is None or page.call.state!='running')
        self._drop_call(page); page.payload=b''; page.active=False; page.state='terminal'
        if not self.connected: self._pages.pop(page.image_id)

    def delivered(self, reply):
        self._owned(); require(self.connected); self._release_delivery(reply)

    def abandon_delivery(self, reply):
        """I/O confirms the closed connection's outstanding send is discarded."""
        self._owned(); require(not self.connected); self._release_delivery(reply)

    def _release_delivery(self, reply):
        page=reply.delivery
        require(page is not None); self._page(page)
        require(page.state=='delivering' and not page.active)
        page.result=b''
        self._pages.pop(page.image_id)

    def disconnect(self):
        self._owned(); self.connected=False
        for page in list(self._pages.values()): self._stop(page,'DISCONNECTED')
