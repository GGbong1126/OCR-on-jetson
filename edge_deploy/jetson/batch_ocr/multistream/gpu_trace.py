"""Optional own-process CUPTI stream IDs; not used by production or benchmarks."""
import ctypes as ct
import os
from pathlib import Path


def start(runtime):
    path=Path(os.environ['AUTOOCR_CUPTI_LIBRARY']).resolve()
    if not path.is_relative_to(Path('/home/xchq/work/autoocr/staging')): raise ValueError('Project-local pinned CUPTI required')
    cupti=ct.CDLL(str(path)); driver=ct.CDLL('libcuda.so.1')
    driver.cuCtxGetCurrent.argtypes=[ct.POINTER(ct.c_void_p)]; driver.cuCtxGetCurrent.restype=ct.c_int
    cupti.cuptiGetContextId.argtypes=[ct.c_void_p,ct.POINTER(ct.c_uint32)]; cupti.cuptiGetContextId.restype=ct.c_int
    cupti.cuptiGetStreamIdEx.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_uint8,ct.POINTER(ct.c_uint32)]
    cupti.cuptiGetStreamIdEx.restype=ct.c_int
    context=ct.c_void_p(); context_id=ct.c_uint32()
    if driver.cuCtxGetCurrent(ct.byref(context)) or cupti.cuptiGetContextId(context,ct.byref(context_id)):
        raise RuntimeError('Cannot identify current CUDA context')
    pairs=[('det_n1',runtime.det1)]
    if runtime.det2 is not None: pairs.append(('det_n2',runtime.det2))
    for i,routes in enumerate(runtime.rec): pairs.extend(('rec%d_%s'%(i,name),slot) for name,slot in routes.items())
    ids={}
    for label,slot in pairs:
        value=ct.c_uint32()
        if cupti.cuptiGetStreamIdEx(None,slot.stream,0,ct.byref(value)): raise RuntimeError('Cannot identify CUDA stream')
        ids[label]=value.value
    runtime.cuda.call('cudaProfilerStart')
    return dict(context_id=context_id.value,stream_ids=ids,scope='candidate_after_reference_before_safe_close')
