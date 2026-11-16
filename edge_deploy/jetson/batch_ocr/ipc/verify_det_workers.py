"""Independent two-image software decode + original det through real worker IPC.

Sequential reference and candidate reuse ONE N1 engine on ONE GPU thread.
Technical hashes/statistics only; not full OCR, hardware JPEG, GT or benchmark.
"""
import argparse
from collections import deque
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np

import ipc_wire as wire
from channel_io import ChannelServer
from stage_contract import validate_stage
from stage_session import encoded, require
from worker_bridge import Tensor, WorkerBridge

BASE=Path('/home/xchq/work/autoocr')
RELEASE=BASE/'releases/jetson-r2'
SOURCE=BASE/'staging/det_batch_e153_r1'


def digest(data): return hashlib.sha256(data).hexdigest()


def client(path, inputs, results, errors, image_types=None):
    sockets={}; ids={}; nonce=None; deadline=time.monotonic()+240
    try:
        for role in ('input','gpu','result','events'):
            sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); sockets[role]=sock
            sock.settimeout(5); sock.connect(str(path))
            hello=wire.receive_frame(sock,deadline,validator=validate_stage)
            meta=json.loads(hello.metadata); require(meta['op']=='session.ready')
            if nonce is None: nonce=meta['session']
            require(meta['session']==nonce)
            bind=dict(schema='ocr.stage/1',op='channel.bind',session=nonce,remaining_ms=300000,channel=role)
            wire.send_frame(sock,wire.Frame(1,encoded(bind),b''),deadline,validator=validate_stage)
            ack=wire.receive_frame(sock,deadline,validator=validate_stage)
            require(ack.request_id==1 and json.loads(ack.metadata)['channel']==role); ids[role]=1
        def rpc(role,op,body=b'',**fields):
            ids[role]+=1
            m=dict(schema='ocr.stage/1',op=op,session=nonce,remaining_ms=300000,**fields)
            wire.send_frame(sockets[role],wire.Frame(ids[role],encoded(m),body),deadline,validator=validate_stage)
            reply=wire.receive_frame(sockets[role],deadline,validator=validate_stage)
            r=json.loads(reply.metadata)
            require(reply.request_id==ids[role] and r['session']==nonce)
            require(r.get('code')=='OK' if op!='image.collect' else r.get('status')=='ok')
            return reply
        for index,data in enumerate(inputs):
            rpc('input','image.submit',data,image_id='image'+str(index),
                image_type=3 if image_types is None else image_types[index],format='encoded')
        done=set(); watermark=1; pending=deque(); running=None
        while len(done)<len(inputs):
            event=wire.receive_frame(sockets['events'],deadline,validator=validate_stage)
            meta=json.loads(event.metadata); require(meta['session']==nonce and event.request_id>watermark)
            watermark=event.request_id
            if meta['op']=='model.ready': pending.append(meta['call_id'])
            elif meta['op']=='model.done':
                require(meta['call_ids']==[running]); running=None
                for call_id in meta['call_ids']: rpc('result','model.resume',call_id=call_id)
            elif meta['op']=='image.done':
                require(meta['status']=='ok' and meta['image_id'] not in done)
                done.add(meta['image_id'])
            else: raise RuntimeError('Unexpected worker event')
            if running is None and pending:
                running=pending.popleft(); rpc('gpu','model.run',call_ids=[running])
        for index in reversed(range(len(inputs))):
            r=rpc('result','image.collect',image_id='image'+str(index))
            results.append(json.loads(r.payload))
    except Exception as error: errors.append(type(error).__name__)
    finally:
        for sock in sockets.values(): sock.close()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    args.output=args.output.resolve()
    require(not args.output.exists())
    mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    require(int(mem['MemAvailable'].split()[0])>=8*1024**2,'LIMIT')
    sys.path.insert(0,str(SOURCE)); sys.path.insert(0,str(RELEASE/'edge_deploy/jetson'))
    import det_batch2 as d
    from det_worker_adapter import DetRuntime
    cv2.setNumThreads(1)
    plan=json.loads((SOURCE/'plan.json').read_text())
    rows=plan['rows'][:2]; require(len(rows)==2 and len({r['id'] for r in rows})==2)
    method=d.legacy_method(); prepared=d.inputs(dict(plan,rows=rows),method)
    payloads=[(Path(plan['images'])/r['file']).read_bytes() for r in rows]
    require(all(digest(data)==r['id'] for data,r in zip(payloads,rows)))
    cfg=json.loads((RELEASE/'edge_deploy/jetson/deployment.json').read_text())['engines']['det']
    references={}; runtime=[]; identities=set(); cpu_ids=set(); results=[]; errors=[]
    def factory():
        gpu=DetRuntime(RELEASE/cfg['path'],cfg['sha256']); runtime.append(gpu)
        identities.add(threading.get_ident())
        for row,(_,x) in zip(rows,prepared):
            references[row['id']]=gpu.run('det',(Tensor('x',tuple(x.shape),x.tobytes()),))[0]
        gpu.synchronize(); return gpu
    def algorithm(payload, kind, api):
        cpu_ids.add(threading.get_ident()); require(kind==3)
        row=next(r for r in rows if r['id']==digest(payload)); metrics={}
        require(row['image_hw'][0]*row['image_hw'][1]<=16_000_000)
        image=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_COLOR); require(image is not None)
        require(list(image.shape[:2])==row['image_hw'])
        with ExitStack() as stack:
            class Proxy:
                def get_inputs(self): return [SimpleNamespace(name='x')]
                def run(self, names, feeds):
                    x=feeds['x']; require(d.tensor_sha(x)==row['tensor_sha256'])
                    out=stack.enter_context(api.infer('det',(Tensor('x',tuple(x.shape),x.tobytes()),)))[0]
                    ref=references[row['id']]
                    require(out.shape==ref.shape and out.data==ref.data)
                    metrics.update(input_shape=list(x.shape),input_sha256=digest(x.tobytes()),
                                   output_shape=list(out.shape),output_sha256=digest(out.data),probability_exact=True)
                    return [np.frombuffer(out.data,np.float32).reshape(out.shape)]
            boxes=method(SimpleNamespace(det_session=Proxy()),image)
            ref=references[row['id']]
            _,original_boxes=d.legacy(method,image,np.frombuffer(ref.data,np.float32).reshape(ref.shape))
            require(np.array_equal(np.asarray(boxes),np.asarray(original_boxes)))
            metrics.update(image_id=api.job.page.image_id,image_sha256=row['id'],box_count=len(boxes),boxes_exact=True,
                           boxes_sha256=digest(np.asarray(boxes,np.float32).tobytes()))
        return encoded(metrics)
    bridge=None; io=None; closed=False
    try:
        with tempfile.TemporaryDirectory(prefix='det_workers_',dir=args.output.parent) as folder, \
                wire.private_listener(Path(folder)/'det_workers.sock') as listener:
            path=Path(folder)/'det_workers.sock'
            io=ChannelServer(listener); bridge=WorkerBridge(io,algorithm,factory)
            thread=threading.Thread(target=client,args=(path,payloads,results,errors),name='validation-client')
            thread.start(); end=time.monotonic()+250
            try:
                while thread.is_alive() and time.monotonic()<end:
                    io.poll(.01); bridge.step()
                thread.join(1); require(not thread.is_alive()); require(not errors)
                require(len(results)==2 and {r['image_id'] for r in results}=={'image0','image1'})
            finally:
                closed=bridge.close(5)
                if closed: io.close()
        require(closed and bridge.gpu_calls==2 and len(runtime)==1 and bridge.high_jobs==2)
        require(runtime[0].session.closed and runtime[0].session.errors==0 and runtime[0].session.calls==4)
        require(len(cpu_ids)==2 and runtime[0].owner not in cpu_ids)
        report=dict(status='two_real_det_pages_exact_not_full_ocr',images=results,workers=bridge.stats(),
                    io=io.stats(),session=io.session.stats(),cpu_workers_observed=len(cpu_ids),gpu_owner_count=len(identities),
                    engine_sha256=cfg['sha256'],reference_gpu_calls=2,candidate_gpu_calls=2,
                    backend_calls=runtime[0].session.calls,backend_errors=runtime[0].session.errors,
                    note='First two frozen E153 inputs, software decode; original det pre/post via AST. '
                         'One engine, sequential reference then two worker pages; extra reference tensors diagnostic only. '
                         'No N2, layout/rec/CTC, business thumbnail/enhancement, HTTP, Extract, JPEG hardware, GT or speedup.')
    except Exception as error:
        report=dict(status='failed',error_type=type(error).__name__,errors=errors,closed=closed,
                    workers=bridge.stats() if bridge else None)
    with args.output.open('x',encoding='utf-8') as file: json.dump(report,file,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('status','closed','workers') if k in report}),flush=True)
    require(report['status']=='two_real_det_pages_exact_not_full_ocr')


if __name__=='__main__': main()
