"""Single-owner nonblocking Unix connection group; workers use registered wakeups."""
from collections import deque
import selectors
import socket
import time

import ipc_wire as wire
from stage_contract import LIMITS, validate_stage
from stage_session import CHANNELS, Session, StateError, encoded, require

ROLES=tuple(CHANNELS)+('events',)


class Reader:
    def __init__(self, validate, reserve, release, clock, seconds):
        self.validate=validate; self.reserve=reserve; self.release=release
        self.clock=clock; self.seconds=seconds; self.credit=0
        self.reset()

    def reset(self):
        if self.credit: self.release(self.credit)
        self.credit=0; self.phase='header'; self.buffer=bytearray(wire.HEADER.size)
        self.offset=0; self.deadline=None; self.metadata=b''; self.sizes=None

    def read(self, sock, budget=65536):
        while budget>0:
            if self.deadline is not None and self.clock()>=self.deadline: raise TimeoutError('frame deadline')
            if self.offset<len(self.buffer):
                try: n=sock.recv_into(memoryview(self.buffer)[self.offset:self.offset+budget])
                except (BlockingIOError,InterruptedError): return None
                if not n: raise wire.PeerClosed('connection closed')
                if self.deadline is None: self.deadline=self.clock()+self.seconds
                self.offset+=n; budget-=n
                if self.offset<len(self.buffer): continue
            if self.phase=='header':
                self.sizes=wire.parse_header(bytes(self.buffer))
                self.phase='metadata'; self.buffer=bytearray(self.sizes[1]); self.offset=0
            elif self.phase=='metadata':
                self.metadata=bytes(self.buffer); n=self.sizes[2]
                self.validate(self.metadata,n)
                if self.clock()>=self.deadline: raise TimeoutError('frame deadline')
                self.reserve(2*n); self.credit=2*n
                self.phase='payload'; self.buffer=bytearray(n); self.offset=0
                if not n:
                    if self.clock()>=self.deadline: raise TimeoutError('frame deadline')
                    return wire.Frame(self.sizes[0],self.metadata,b'')
            else:
                if self.clock()>=self.deadline: raise TimeoutError('frame deadline')
                return wire.Frame(self.sizes[0],self.metadata,bytes(self.buffer))
        return None


class Writer:
    def __init__(self, frame, deadline, delivery=None):
        self.parts=(memoryview(wire.header(frame.request_id,len(frame.metadata),len(frame.payload))),
                    memoryview(frame.metadata),memoryview(frame.payload))
        self.size=sum(map(len,self.parts)); self.index=self.offset=0
        self.deadline=deadline; self.delivery=delivery

    def write(self, sock, budget=65536):
        while self.index<len(self.parts):
            part=self.parts[self.index]
            if self.offset==len(part): self.index+=1; self.offset=0; continue
            if budget<=0: return False
            try: n=sock.send(part[self.offset:self.offset+budget])
            except (BlockingIOError,InterruptedError): return False
            if not n: raise wire.PeerClosed('send closed')
            self.offset+=n; budget-=n
        return True


class Peer:
    def __init__(self, sock, pid, server):
        self.sock=sock; self.pid=pid; self.role=None; self.mask=0; self.out=deque()
        self.bind_deadline=server.clock()+server.seconds
        self.reader=Reader(lambda raw,n:server.validate_peer(self,raw,n),
                           server.reserve,server.release,server.clock,server.seconds)


class ChannelServer:
    def __init__(self, listener, *, clock=time.monotonic, seconds=None, transport_bytes=LIMITS['transport_budget_bytes'],allow_det_batch=False,session=None,validator=validate_stage):
        self.clock=clock; self.seconds=LIMITS['control_timeout_ms']/1000 if seconds is None else seconds
        require(self.seconds>0)
        require(type(transport_bytes) is int and 8*wire.MAX_META<transport_bytes<=LIMITS['transport_budget_bytes'])
        self.transport_limit=transport_bytes
        require(type(allow_det_batch) is bool)
        self.validate=validator
        self.session=session if session is not None else Session(clock=clock,allow_det_batch=allow_det_batch)
        self.session._owned(); self.selector=selectors.DefaultSelector()
        self.listener=listener; self.original_timeout=listener.gettimeout(); listener.setblocking(False)
        self.selector.register(listener,selectors.EVENT_READ,None)
        self.peers={}; self.roles={}; self.pid=None; self.group_deadline=None
        self.used=self.high=0; self.received=self.sent=0; self.event_id=1
        self.stopped=self.closed=False; self.reason=''

    @property
    def ready(self): return len(self.roles)==len(ROLES) and not self.stopped

    def reserve(self, n):
        # Reserve small fixed header/metadata space separately from body/output credits.
        require(self.used+n<=self.transport_limit-4*(2*wire.MAX_META+wire.HEADER.size),'BUSY')
        self.used+=n; self.high=max(self.high,self.used)

    def release(self, n):
        self.used-=n; require(self.used>=0)

    def stats(self):
        self.session._owned()
        return dict(peers=len(self.peers),roles=len(self.roles),transport_bytes=self.used,
                    high_transport_bytes=self.high,received=self.received,sent=self.sent,
                    stopped=self.stopped,reason=self.reason,selector=type(self.selector).__name__,transport_limit=self.transport_limit)

    def validate_peer(self, peer, raw, n):
        meta=self.validate(raw,n)
        require(meta['session']==self.session.session,'SESSION')
        if peer.role is None: require(meta['op']=='channel.bind')
        else: require(peer.role in CHANNELS and meta['op'] in CHANNELS[peer.role])
        return meta

    def sync(self, peer):
        if peer.sock not in self.peers: return
        mask=selectors.EVENT_WRITE if peer.out else 0
        if peer.role is None or (self.ready and (peer.role=='events' or not peer.out)):
            mask|=selectors.EVENT_READ
        if mask and not peer.mask: self.selector.register(peer.sock,mask,peer)
        elif mask and mask!=peer.mask: self.selector.modify(peer.sock,mask,peer)
        elif not mask and peer.mask: self.selector.unregister(peer.sock)
        peer.mask=mask

    def queue(self, peer, frame, delivery=None):
        self.validate(frame.metadata,len(frame.payload))
        require(len(peer.out)<LIMITS['queue_items'],'BUSY')
        output=Writer(frame,self.clock()+self.seconds,delivery)
        self.reserve(output.size); peer.out.append(output); self.sync(peer)

    def accept(self):
        for _ in range(4):
            try: sock,_=self.listener.accept()
            except (BlockingIOError,InterruptedError): break
            try:
                pid,_,_=wire.check_peer(sock)
                require(len(self.peers)<4 and (self.pid is None or pid==self.pid))
                sock.setblocking(False); peer=Peer(sock,pid,self); self.peers[sock]=peer
                if self.group_deadline is None: self.group_deadline=self.clock()+self.seconds
                self.queue(peer,wire.Frame(1,encoded(self.session.hello()),b''))
            except (StateError,PermissionError,OSError):
                if sock in self.peers: self.drop(self.peers[sock])
                else: sock.close()
                if not self.peers: self.group_deadline=None

    def bind(self, peer, frame):
        meta=self.validate_peer(peer,frame.metadata,len(frame.payload)); role=meta['channel']
        require(frame.request_id==1 and role not in self.roles)
        require(self.pid is None or peer.pid==self.pid)
        if role in CHANNELS: self.session.bind_channel(role)
        self.pid=peer.pid; peer.role=role; self.roles[role]=peer
        self.queue(peer,wire.Frame(1,encoded(self.session.meta('channel.bound',channel=role)),b''))
        for other in self.peers.values(): self.sync(other)

    def emit(self, meta):
        self.session._owned(); require(self.ready,'DISCONNECTED')
        require(meta['session']==self.session.session and meta['op'] in ('model.ready','model.done','model.failed','image.done'))
        peer=self.roles['events']
        # Keep the event ID unchanged when backpressure rejects a publication.
        frame=wire.Frame(self.event_id+1,encoded(meta),b'')
        self.queue(peer,frame); self.event_id+=1

    def event_room(self):
        require(self.ready,'DISCONNECTED')
        require(len(self.roles['events'].out)<LIMITS['queue_items'],'BUSY')
        require(self.used+wire.MAX_META+wire.HEADER.size<=self.transport_limit-4*(2*wire.MAX_META+wire.HEADER.size),'BUSY')

    def model_ready(self, page, model, tensors):
        self.session._owned(); self.event_room()
        meta=self.session.ready(page,model,tensors); self.emit(meta); return meta

    def model_complete(self, batch, outputs):
        self.session._owned()
        if not self.stopped: self.event_room()
        meta=self.session.complete(batch,outputs)
        if meta is not None: self.emit(meta)
        return meta

    def drop(self, peer):
        if peer.mask: self.selector.unregister(peer.sock); peer.mask=0
        self.peers.pop(peer.sock,None); self.roles.pop(peer.role,None)
        peer.sock.close(); peer.reader.reset()
        deliveries=[]
        while peer.out:
            output=peer.out.popleft(); self.release(output.size)
            if output.delivery: deliveries.append(output.delivery)
            output.parts=()
        return deliveries

    def stop(self, reason='STOPPED'):
        self.session._owned()
        if self.stopped: return
        self.stopped=True; self.reason=reason; self.selector.unregister(self.listener)
        deliveries=[]
        for peer in list(self.peers.values()): deliveries.extend(self.drop(peer))
        self.session.disconnect()
        for reply in deliveries: self.session.abandon_delivery(reply)

    def fail(self, peer, reason):
        if peer.role is not None: self.stop(reason)
        else:
            self.drop(peer)
            if not self.peers: self.group_deadline=None

    def scan(self):
        self.session.expire(); now=self.clock()
        if self.stopped: return
        if self.roles and not self.ready and now>=self.group_deadline:
            self.stop('BIND_TIMEOUT'); return
        for peer in list(self.peers.values()):
            if peer.role is None and now>=peer.bind_deadline: self.fail(peer,'BIND_TIMEOUT')
            elif peer.reader.deadline is not None and now>=peer.reader.deadline: self.fail(peer,'READ_TIMEOUT')
            elif peer.out and now>=peer.out[0].deadline: self.fail(peer,'WRITE_TIMEOUT')
            if self.stopped: break

    def poll(self, timeout=.05):
        self.session._owned(); require(not self.closed)
        self.scan(); notices=[]
        for key,mask in self.selector.select(max(0,min(timeout,.05))):
            peer=key.data
            if peer is None:
                if not self.stopped: self.accept()
                continue
            if callable(peer):
                peer()
                continue
            if peer.sock not in self.peers: continue
            try:
                if mask & selectors.EVENT_WRITE and peer.out:
                    output=peer.out[0]
                    if self.clock()>=output.deadline: raise TimeoutError('send deadline')
                    complete=output.write(peer.sock)
                    if self.clock()>=output.deadline: raise TimeoutError('send deadline')
                    if complete:
                        peer.out.popleft(); self.release(output.size); self.sent+=1
                        if output.delivery: self.session.delivered(output.delivery)
                        output.parts=()
                if mask & selectors.EVENT_READ:
                    frame=peer.reader.read(peer.sock)
                    if frame is not None:
                        if peer.role is None: self.bind(peer,frame)
                        else:
                            reply=self.session.handle(frame,peer.role)
                            try: self.queue(peer,reply.frame,reply if reply.delivery else None)
                            except (StateError,wire.ProtocolError,MemoryError):
                                self.stop('OUTPUT_LIMIT')
                                if reply.delivery: self.session.abandon_delivery(reply)
                                raise
                            meta=self.validate(frame.metadata,len(frame.payload))
                            response=self.validate(reply.frame.metadata,len(reply.frame.payload))
                            notices.append((peer.role,meta,response['op']=='image.result' or response['status']=='ok'))
                        self.received+=1; peer.reader.reset()
                self.sync(peer)
            except (wire.ProtocolError,wire.PeerClosed,TimeoutError,StateError,OSError,MemoryError):
                self.fail(peer,'CONNECTION_FAILED')
        self.scan(); return notices

    def close(self):
        self.session._owned()
        if self.closed: return
        self.stop()
        require(self.session.running is None and self.session.stats()['images']==0)
        require(self.used==0)
        self.selector.close(); self.listener.settimeout(self.original_timeout); self.closed=True
