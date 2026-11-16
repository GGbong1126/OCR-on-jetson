"""Low-resource PTX positive control: NOT REC or a model speed benchmark."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

from probe import B,memory,mapping

# CUDA10.2 PTX ISA6.5, bounded ~millisecond clock loop; no NVCC/toolkit install.
PTX=r'''
.version 6.5
.target sm_72
.address_size 64
.visible .entry concurrent_control(.param .u64 ticks, .param .u64 output)
{
 .reg .pred %p;
 .reg .u64 %r<6>;
 ld.param.u64 %r0, [ticks];
 ld.param.u64 %r1, [output];
 mov.u64 %r2, %clock64;
 again:
 mov.u64 %r3, %clock64;
 sub.u64 %r4, %r3, %r2;
 setp.lt.u64 %p, %r4, %r0;
 @%p bra again;
 st.global.u64 [%r1], %r4;
 ret;
}
'''


def main():
    p=argparse.ArgumentParser(); p.add_argument('--trace',action='store_true'); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); out=args.output.resolve()
    assert out.is_relative_to(B/'staging/rec_concurrency_e166_r1') and not out.exists() and out.parent.is_dir()
    with socket.socket() as sock: assert sock.connect_ex(('127.0.0.1',5052))!=0
    assert memory()['MemAvailable']>=3*1024**2
    sys.path.insert(0,str(B/'releases/jetson-r2/edge_deploy/jetson'))
    sys.path.insert(0,str(B/'overlays/batch-ocr-plan2-e164-r1/multistream'))
    from trt82_backend import Cuda
    from trt_async import AsyncCuda
    cuda=Cuda(); ac=AsyncCuda(cuda); driver=ct.CDLL('libcuda.so.1')
    signatures={'cuModuleLoadData':[ct.POINTER(ct.c_void_p),ct.c_void_p],
                'cuModuleGetFunction':[ct.POINTER(ct.c_void_p),ct.c_void_p,ct.c_char_p],
                'cuModuleUnload':[ct.c_void_p],
                'cuLaunchKernel':[ct.c_void_p]+[ct.c_uint]*7+[ct.c_void_p,ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_void_p)]}
    for name,types in signatures.items(): getattr(driver,name).argtypes=types; getattr(driver,name).restype=ct.c_int
    def call(name,*values):
        code=getattr(driver,name)(*values)
        if code: raise RuntimeError(name+' '+str(code))
    module=ct.c_void_p(); function=ct.c_void_p(); slots=[]; buffers=[]; trace_on=False
    report=dict(status='running',kind='positive_control_NOT_REC',closed=False,rounds=[],ptx=PTX)
    def save(): out.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        call('cuModuleLoadData',ct.byref(module),ct.cast(ct.c_char_p(PTX.encode()),ct.c_void_p))
        call('cuModuleGetFunction',ct.byref(function),module,b'concurrent_control')
        def launch(i):
            ticks=ct.c_uint64(2000000); pointer=ct.c_uint64(buffers[i].value)
            params=(ct.c_void_p*2)(ct.cast(ct.byref(ticks),ct.c_void_p),ct.cast(ct.byref(pointer),ct.c_void_p))
            call('cuLaunchKernel',function,1,1,1,1,1,1,0,slots[i].stream,params,None)
        for i in range(16):
            stream=ct.c_void_p(); ptr=ct.c_void_p()
            ac.call('cudaStreamCreateWithFlags',ct.byref(stream),1); slots.append(SimpleNamespace(stream=stream,cuda=ac))
            cuda.call('cudaMalloc',ct.byref(ptr),8); buffers.append(ptr)
            launch(i); ac.call('cudaStreamSynchronize',stream)
        if args.trace:
            report['trace']=mapping(slots); cuda.call('cudaProfilerStart'); trace_on=True
        for _ in range(5):
            begin=time.monotonic()
            for i in range(16): launch(i)
            for slot in slots: ac.call('cudaStreamSynchronize',slot.stream)
            report['rounds'].append(dict(seconds=time.monotonic()-begin))
        if trace_on: cuda.call('cudaProfilerStop'); trace_on=False; report['trace']['stopped']=True
        values=[]
        for ptr in buffers:
            value=ct.c_uint64(); cuda.call('cudaMemcpy',ct.byref(value),ptr,8,2); values.append(value.value)
        assert all(v>=2000000 for v in values)
        report.update(status='positive_control_completed_NOT_REC',ticks=values,slots=16)
    finally:
        safe=False
        try:
            if trace_on: cuda.call('cudaProfilerStop')
            for slot in slots: ac.call('cudaStreamSynchronize',slot.stream)
            safe=True
            for ptr in buffers: cuda.call('cudaFree',ptr)
            for slot in slots: ac.call('cudaStreamDestroy',slot.stream)
            if module.value: call('cuModuleUnload',module)
            report['closed']=True
        finally:
            report['safe_quiescence_confirmed']=safe; save()
            if not safe:
                while True: time.sleep(1)


if __name__=='__main__': main()
