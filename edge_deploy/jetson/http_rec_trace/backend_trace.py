"""Run the exact E167 backend; add startup stream IDs and idle profiler gates only."""
import ctypes as ct
import importlib.util
import json
import os
from pathlib import Path
import threading
import traceback

from settings import S,SOURCE,TOOL
import sys
sys.path[:0]=[str(SOURCE/'app'),str(SOURCE/'ipc'),str(SOURCE/'multistream')]
import threaded_rec

F=S/'trace'; identities={}; lock=threading.Lock(); phase='off'
Original=threaded_rec.Resource


def atomic(name,value):
    p=F/name; q=p.with_suffix('.new'); q.write_text(json.dumps(value,indent=2)+'\n'); os.replace(q,p)


class ObservedResource(Original):
    def __init__(self,index):
        super().__init__(index)
        try:
            driver=ct.CDLL('libcuda.so.1'); cupti=ct.CDLL(str(TOOL/'targets/aarch64-linux/lib/libcupti.so.10.2.175'))
            driver.cuCtxGetCurrent.argtypes=[ct.POINTER(ct.c_void_p)]; driver.cuCtxGetCurrent.restype=ct.c_int
            cupti.cuptiGetContextId.argtypes=[ct.c_void_p,ct.POINTER(ct.c_uint32)]; cupti.cuptiGetContextId.restype=ct.c_int
            cupti.cuptiGetStreamIdEx.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_uint8,ct.POINTER(ct.c_uint32)]
            cupti.cuptiGetStreamIdEx.restype=ct.c_int
            context=ct.c_void_p(); identifier=ct.c_uint32()
            assert driver.cuCtxGetCurrent(ct.byref(context))==0
            assert cupti.cuptiGetContextId(context,ct.byref(identifier))==0
            streams={}
            for name,slot in self.routes.items():
                value=ct.c_uint32()
                assert cupti.cuptiGetStreamIdEx(context,slot.stream,0,ct.byref(value))==0
                streams[name]=value.value
            with lock:
                identities[index]=dict(context_id=identifier.value,stream_ids=streams,native_thread_id=threading.get_native_id())
                atomic('mapping.json',dict(owners=identities))
        except Exception:
            self.close(); raise


threaded_rec.Resource=ObservedResource
spec=importlib.util.spec_from_file_location('production_backend',SOURCE/'app/serve_backend.py')
backend=importlib.util.module_from_spec(spec); spec.loader.exec_module(backend)
original_status=backend.status
import bridge
original_init=bridge.MultiBridge.__init__


def observed_init(self,io,algorithm,runtime_factory,*args,**kwargs):
    def factory():
        try: return runtime_factory()
        except Exception:
            atomic('factory_error.json',dict(traceback=traceback.format_exc())); raise
    original_init(self,io,algorithm,factory,*args,**kwargs)


bridge.MultiBridge.__init__=observed_init


def observed_status(path,data):
    global phase
    original_status(path,data)
    if data.get('status')!='ready': return
    cmd=F/'command.json'
    if not cmd.exists(): return
    command=json.loads(cmd.read_text())['command']
    if (phase,command) not in (('off','start'),('on','stop')): return
    session=data['session']
    assert session['images']==session['registered_bytes']==session['active_reserved']==0
    assert len(identities)==4 and len({v['context_id'] for v in identities.values()})==1
    cuda=ct.CDLL('libcudart.so.10.2'); cuda.cudaSetDevice.argtypes=[ct.c_int]; cuda.cudaSetDevice.restype=ct.c_int
    assert cuda.cudaSetDevice(0)==0
    function=cuda.cudaProfilerStart if command=='start' else cuda.cudaProfilerStop
    function.argtypes=[]; function.restype=ct.c_int; assert function()==0
    phase='on' if command=='start' else 'stopped'
    atomic('gate.json',dict(phase=phase,backend_pid=os.getpid(),idle_confirmed=True))


backend.status=observed_status
if __name__=='__main__':
    try: backend.main()
    finally:
        marker=F/'runtime/status.json'
        state=json.loads(marker.read_text()) if marker.exists() else {}
        atomic('backend_exit.json',dict(phase=phase,status=state.get('status'),
               close_error=state.get('worker',{}).get('close_error'),backend_pid=os.getpid()))
