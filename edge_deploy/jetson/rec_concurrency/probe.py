"""Isolated original-engine REC kernel concurrency experiment; never a service."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

B=Path('/home/xchq/work/autoocr')
SOURCE=B/'overlays/batch-ocr-plan2-e164-r1'
NARROW=B/'releases/jetson-r2/pth/tensorrt/ocr_rec_fp32_w32_1536_r1.engine'
WIDE=B/'overlays/rec-wide-e124-r1/rec_wide_fp32.engine'
HASHES=('da2ff03a315a410ccd1d795af129c4220905d45ea2732da0d4441c8ea914c6ff',
        '8e3f9167d1496011c8bf528df0c5eff006bc8a0d4f254e2d42a9a47009e7c95b')


def memory():
    rows=dict(l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines())
    return {k:int(rows[k].split()[0]) for k in ('MemAvailable','NvMapMemUsed','SwapFree')}


def tokens(a):
    ids=a[0].argmax(axis=1); previous=-1; result=[]
    for value in ids:
        if value and value!=previous: result.append(int(value))
        previous=value
    return result


def mapping(slots):
    driver=ct.CDLL('libcuda.so.1'); cupti=ct.CDLL(os.environ['AUTOOCR_CUPTI_LIBRARY'])
    driver.cuCtxGetCurrent.argtypes=[ct.POINTER(ct.c_void_p)]; driver.cuCtxGetCurrent.restype=ct.c_int
    cupti.cuptiGetContextId.argtypes=[ct.c_void_p,ct.POINTER(ct.c_uint32)]; cupti.cuptiGetContextId.restype=ct.c_int
    cupti.cuptiGetStreamIdEx.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_uint8,ct.POINTER(ct.c_uint32)]
    cupti.cuptiGetStreamIdEx.restype=ct.c_int
    context=ct.c_void_p(); identifier=ct.c_uint32()
    assert driver.cuCtxGetCurrent(ct.byref(context))==0
    assert cupti.cuptiGetContextId(context,ct.byref(identifier))==0
    ids={}
    for i,slot in enumerate(slots):
        value=ct.c_uint32()
        assert cupti.cuptiGetStreamIdEx(context,slot.stream,0,ct.byref(value))==0
        ids['rec'+str(i)]=value.value
    return dict(context_id=identifier.value,stream_ids=ids)


def graph_functions(cuda):
    signatures={
        'cudaStreamBeginCapture':[ct.c_void_p,ct.c_int],
        'cudaStreamEndCapture':[ct.c_void_p,ct.POINTER(ct.c_void_p)],
        'cudaGraphInstantiate':[ct.POINTER(ct.c_void_p),ct.c_void_p,ct.POINTER(ct.c_void_p),ct.c_char_p,ct.c_size_t],
        'cudaGraphLaunch':[ct.c_void_p,ct.c_void_p],
        'cudaGraphExecDestroy':[ct.c_void_p], 'cudaGraphDestroy':[ct.c_void_p]}
    for name,args in signatures.items():
        f=getattr(cuda.lib,name); f.argtypes=args; f.restype=ct.c_int


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('slot','resident','graph'),required=True)
    p.add_argument('--width',type=int,choices=(32,160,1287,2081),required=True)
    p.add_argument('--slots',type=int,choices=(1,2,4,8,16),default=16)
    p.add_argument('--trace',action='store_true'); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); out=args.output.resolve()
    assert out.is_relative_to(B/'staging/rec_concurrency_e166_r1') and out.parent.is_dir() and not out.exists()
    with socket.socket() as sock:
        sock.settimeout(2); assert sock.connect_ex(('127.0.0.1',5052))!=0,'Stable service must be normally stopped'
    assert memory()['MemAvailable']>=8*1024**2,'Do not duplicate a loaded model set'
    for name in ('CUDA_LAUNCH_BLOCKING','CUDA_DEVICE_MAX_CONNECTIONS'):
        assert not os.environ.get(name),name+' must not override the experiment'
    sys.path[:0]=[str(SOURCE/'multistream'),str(B/'releases/jetson-r2/edge_deploy/jetson')]
    from trt82_backend import Session
    from trt_async import Slot
    report=dict(status='running',mode=args.mode,width=args.width,requested_slots=args.slots,
                batch=1,synthetic=True,precision='original_fp32_no_tf32',passes=[],closed=False,
                memory_before=memory(),trace_requested=args.trace,uid=os.geteuid(),
                note='Fixed-shape synthetic REC only, no GT or HTTP throughput; profiler timing is diagnostic only.')
    slots=[]; sessions=[]; refs=[]; feeds=[]; graphs=[]; executables=[]; capturing=None; trace_on=False
    def save(): out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def sync(slot): slot.cuda.call('cudaStreamSynchronize',slot.stream)
    def pointers(slot): return [slot.device[i].value for i in range(slot.session.engine.num_bindings)]
    def enqueue(slot):
        if not slot.session.context.execute_async_v2(pointers(slot),slot.stream.value): raise RuntimeError('Enqueue failed')
    def outputs(slot):
        values=[]
        for item in slot.session.outputs:
            a=slot.arrays[item.index]
            slot.cuda.call('cudaMemcpyAsync',slot.host[item.index],slot.device[item.index],a.nbytes,2,slot.stream)
        sync(slot)
        for item in slot.session.outputs: values.append(slot.arrays[item.index].copy())
        return values
    save()
    try:
        wide=args.width>1536; path=WIDE if wide else NARROW; expected=HASHES[int(wide)]
        for i in range(args.slots):
            if memory()['MemAvailable']<3*1024**2:
                report['capacity_stop']='MemAvailable below 3GiB before next slot'; break
            s=Session(path,expected,'isolated_rec_'+str(i)); sessions.append(s)
            slot=Slot(s); slots.append(slot)
            feed={'x':np.random.RandomState(166+i).uniform(-1,1,(1,3,48,args.width)).astype(np.float32)}
            feeds.append(feed)
            reference=s.run(None,feed)[0]; assert np.isfinite(reference).all(); refs.append(reference)
            for _ in range(3):
                slot.submit(('warm',i),feed); sync(slot); slot.collect()
            # Recreate NumPy views after collect clears arrays; then keep resident buffers immutable.
            slot.submit(('resident',i),feed); sync(slot)
            report['memory_loaded']=memory(); save()
        assert slots
        cuda=slots[0].session.cuda
        cuda.lib.cudaDeviceGetAttribute.argtypes=[ct.POINTER(ct.c_int),ct.c_int,ct.c_int]
        cuda.lib.cudaDeviceGetAttribute.restype=ct.c_int
        props={}
        for name,attribute in (('multiprocessor_count',16),('concurrent_kernels',31),('cc_major',75),('cc_minor',76)):
            v=ct.c_int(); cuda.call('cudaDeviceGetAttribute',ct.byref(v),attribute,0); props[name]=v.value
        report.update(device=props,allocated_slots=len(slots),engine_sha256=expected,
                      fixed_io_bytes=sum(s.host_bytes+s.device_bytes for s in slots),
                      declared_context_bytes=sum(s.session.engine.device_memory_size for s in slots))
        if args.mode=='graph':
            graph_functions(cuda)
            for slot in slots:
                enqueue(slot); sync(slot)
                graph=ct.c_void_p(); executable=ct.c_void_p()
                cuda.call('cudaStreamBeginCapture',slot.stream,1); capturing=slot
                enqueue(slot)
                code=cuda.lib.cudaStreamEndCapture(slot.stream,ct.byref(graph)); capturing=None
                if code: cuda.call('cudaGetLastError'); raise RuntimeError('Graph capture failed: '+str(code))
                graphs.append(graph)
                cuda.call('cudaGraphInstantiate',ct.byref(executable),graph,None,None,0); executables.append(executable)
                cuda.call('cudaGraphLaunch',executable,slot.stream); sync(slot)
        if args.mode=='slot':
            for slot in slots: slot.collect()
        if args.trace:
            report['trace']=mapping(slots)
            cuda.call('cudaProfilerStart'); trace_on=True
        counts=([len(slots)] if args.trace else [n for n in (1,2,4,8,16) if n<=len(slots)])
        for direction in (('trace',) if args.trace else ('forward','reverse')):
            for count in (list(reversed(counts)) if direction=='reverse' else counts):
                rounds=5 if args.trace else 9; rows=[]
                for repeat in range(rounds):
                    starts=[]; calls=[]; begin=time.monotonic()
                    for i,slot in enumerate(slots[:count]):
                        a=time.monotonic(); starts.append((a-begin)*1000)
                        if args.mode=='slot': slot.submit((direction,count,repeat,i),feeds[i])
                        elif args.mode=='resident': enqueue(slot)
                        else: slot.cuda.call('cudaGraphLaunch',executables[i],slot.stream)
                        calls.append((time.monotonic()-a)*1000)
                    submitted=time.monotonic()
                    for slot in slots[:count]: sync(slot)
                    elapsed=time.monotonic()-begin
                    if args.mode=='slot':
                        for slot in slots[:count]: slot.collect()
                    if args.trace or repeat>=3:
                        rows.append(dict(wall_ms=elapsed*1000,submit_ms=(submitted-begin)*1000,
                                         starts_ms=starts,call_ms=calls))
                report['passes'].append(dict(direction=direction,slots=count,rounds=rows,
                    inferences_per_second=count*len(rows)/sum(r['wall_ms']/1000 for r in rows)))
                save(); print('Done',args.mode,args.width,direction,count,flush=True)
        if trace_on:
            cuda.call('cudaProfilerStop'); trace_on=False; report['trace']['stopped']=True
        checks=[]
        for i,slot in enumerate(slots):
            if args.mode=='slot':
                slot.submit(('verify',i),feeds[i]); sync(slot)
            value=outputs(slot)[0]; ref=refs[i]
            check=dict(slot=i,all_finite=bool(np.isfinite(value).all()),
                       allclose=bool(np.allclose(value,ref,rtol=1e-3,atol=1e-4)),
                       ctc_indices_equal=tokens(value)==tokens(ref),max_abs_error=float(np.abs(value-ref).max()))
            checks.append(check)
        report['checks']=checks
        assert all(r['all_finite'] and r['allclose'] and r['ctc_indices_equal'] for r in checks)
        report['status']='rec_outputs_consistent'
    except Exception as error:
        report.update(status='failed',error=type(error).__name__,detail=str(error))
        raise
    finally:
        safe=False
        try:
            if capturing is not None:
                graph=ct.c_void_p(); capturing.cuda.cuda.lib.cudaStreamEndCapture(capturing.stream,ct.byref(graph))
                if graph.value: graphs.append(graph)
            if trace_on:
                slots[0].cuda.call('cudaProfilerStop'); report['trace']['stopped']=True
            for slot in slots: slot.quiesce()
            safe=True
            if slots:
                for exe in executables: slots[0].cuda.call('cudaGraphExecDestroy',exe)
                for graph in graphs: slots[0].cuda.call('cudaGraphDestroy',graph)
            for slot in reversed(slots): slot.close()
            for session in reversed(sessions): session.close()
            report['closed']=True
        finally:
            report['safe_quiescence_confirmed']=safe; report['memory_end']=memory(); save()
            if not safe:
                print('QUARANTINED: CUDA safety unknown; retain resources, no reset/kill',flush=True)
                while True: time.sleep(1)
    print(json.dumps(dict(status=report['status'],slots=len(slots),closed=report['closed'])),flush=True)


if __name__=='__main__': main()
