"""Original OCR algorithms; candidate async det/rec resources owned by one thread."""
import ast
import inspect
from pathlib import Path
import textwrap
import time
from types import MethodType

import numpy as np

from ocr_worker_adapter import OCRRuntime
from stage_session import require
from worker_bridge import Tensor
from trt_async import Slot

B=Path('/home/xchq/work/autoocr'); R=B/'releases/jetson-r2'


def rec_functions(engine):
    """Reuse the loaded original function AST, not a second CTC/preprocess implementation."""
    original=engine._rec_run.__func__
    tree=ast.parse(textwrap.dedent(inspect.getsource(original))); f=tree.body[0]
    cut=[i for i,s in enumerate(f.body) if isinstance(s,ast.Assign) and
         any(isinstance(t,ast.Name) and t.id=='output' for t in s.targets)]
    require(len(cut)==1); i=cut[0]
    assignment=f.body[i]
    require(isinstance(assignment.value,ast.Subscript) and isinstance(assignment.value.value,ast.Call))
    call=assignment.value.value
    require(isinstance(call.func,ast.Attribute) and call.func.attr=='run')
    prepare=ast.FunctionDef(name='prepare',args=f.args,body=f.body[:i]+[ast.Return(ast.Name('input_tensor',ast.Load()))],
                           decorator_list=[],returns=None,type_comment=None)
    decode=ast.FunctionDef(name='decode',args=ast.arguments(posonlyargs=[],args=[ast.arg('self'),ast.arg('output')],
                           vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),
                           body=f.body[i+1:],decorator_list=[],returns=None,type_comment=None)
    module=ast.fix_missing_locations(ast.Module(body=[prepare,decode],type_ignores=[])); namespace=dict(original.__globals__)
    exec(compile(module,'<original_rec_split>','exec'),namespace)
    return MethodType(namespace['prepare'],engine),MethodType(namespace['decode'],engine)


class AsyncRuntime(OCRRuntime):
    def __init__(self,*args,rec_slots=1,**kwargs):
        self.slots=[]; self.rec_slots_count=rec_slots; self.async_high=0; self.async_timings={}
        super().__init__(*args,**kwargs)
        from trt82_backend import Session
        from rec_wide_adapter import ENGINE_SHA, ORIGINAL_PROFILE, WIDE_PROFILE
        det=self.models.models['det'].gpu
        rec=self.models.models['rec'].gpu
        require(rec.base.profiles=={'x':ORIGINAL_PROFILE} and rec.wide.profiles=={'x':WIDE_PROFILE})
        # Startup-only removal of old synchronous scratch buffers; contexts/weights remain original.
        for s in (det,rec.base,rec.wide,self.det_batch):
            if s is None: continue
            for ptr,_ in s.buffers.values(): s.cuda.call('cudaFree',ptr)
            s.buffers.clear()
        self.det2=self.add(self.det_batch) if self.det_batch is not None else None
        self.det1=self.add(det,storage=self.det2)
        self.rec=[]
        for i in range(rec_slots):
            wide=rec.wide if i==0 else Session(B/'overlays/rec-wide-e124-r1/rec_wide_fp32.engine',ENGINE_SHA,'async_rec_wide_'+str(i))
            narrow=rec.base if i==0 else Session(R/'pth/tensorrt/ocr_rec_fp32_w32_1536_r1.engine',
                'da2ff03a315a410ccd1d795af129c4220905d45ea2732da0d4441c8ea914c6ff','async_rec_'+str(i))
            w=self.add(wide,owned_session=i>0); n=self.add(narrow,storage=w,owned_session=i>0)
            self.rec.append(dict(narrow=n,wide=w))
        self.slot_allocation=[s.stats() for s in self.slots]
        require(sum(s.host_bytes+s.device_bytes for s in self.slots)<=240*1024**2,'LIMIT')
        from jetson_models import CheckedCPU, cpu_session
        model=self.models.models['rec']
        if model.cpu is None: model.cpu=CheckedCPU(cpu_session(model.onnx),'rec','out_of_profile_original_tensor_no_resize')
        self.cpu=model.cpu

    def add(self,session,**kwargs):
        slot=Slot(session,**kwargs); self.slots.append(slot); return slot

    def submit_det(self,ticket,requests):
        self._owned(); require(1<=len(requests)<=2)
        arrays=[np.frombuffer(r.tensors[0].data,np.float32).reshape(r.tensors[0].shape) for r in requests]
        require(all(r.model=='det' and len(r.tensors)==1 for r in requests))
        slot=self.det1 if len(requests)==1 else self.det2; require(slot is not None)
        slot.submit(ticket,{'x':arrays[0] if len(arrays)==1 else np.concatenate(arrays)})
        return slot

    def submit_rec(self,slot_id,ticket,request):
        self._owned(); t=request.tensors[0]; width=t.shape[3]
        require(32<=width<=2304 and 1<=slot_id<=self.rec_slots_count)
        slot=self.rec[slot_id-1]['narrow' if width<=1536 else 'wide']
        slot.submit(ticket,{'x':np.frombuffer(t.data,np.float32).reshape(t.shape)})
        return slot

    def collect(self,slot,role):
        _,values,timing=slot.collect(as_bytes=True)
        require(len(values)==len(self.names[role]) and sum(len(v[1]) for v in values)<=64*1024**2,'LIMIT')
        self.output_dtypes[role].update(v[2] for v in values)
        return tuple(Tensor(n,shape,data,dtype) for n,(shape,data,dtype) in zip(self.names[role],values)),timing

    def run_cpu(self,request):
        t=request.tensors[0]; require(t.shape[3]>2304)
        values=self.cpu.run(None,{'x':np.frombuffer(t.data,np.float32).reshape(t.shape)})
        return tuple(Tensor(n,tuple(v.shape),v.tobytes(),v.dtype.name) for n,v in zip(self.names['rec'],values))

    def stats(self):
        value=super().stats()
        if not hasattr(self,'rec'): return value
        rows=[{name:slot.session.stats() for name,slot in routes.items()} for routes in self.rec]
        value['rec'].update(calls=sum(s['calls'] for r in rows for s in r.values()),
                            errors=sum(s['errors'] for r in rows for s in r.values()),slot_execution=rows)
        return value

    def synchronize(self):
        self._owned()
        for s in self.slots: s.quiesce()
        super().synchronize()

    def close(self):
        self._owned()
        for slot in reversed(self.slots): slot.close()
        super().close()

    def time_method(self,target,name,label,**kwargs):
        split=rec_functions(target) if name=='_rec_run' and getattr(self.local,'api',None) is not None else None
        super().time_method(target,name,label,**kwargs)
        if split is None: return
        prepare,decode=split
        def predict(engine,image):
            boxes=engine._det_run(image)
            if not boxes: return []
            api=self.local.api
            def make(box):
                crop=engine._crop_text_region(image,box)
                if crop is None: return None
                feeds=prepare(crop); require(type(feeds) is dict and set(feeds)=={'x'})
                x=np.ascontiguousarray(feeds['x']); self.local.calls['rec']+=1; self.local.widths.append(x.shape[3])
                return (Tensor('x',tuple(x.shape),x.tobytes()),)
            try: pairs=api.recognize(image,boxes,make,decode)
            except Exception: self.local.failed=True; raise
            result=[]
            for box,pair in zip(boxes,pairs):
                if pair is not None and pair[0]:
                    result.append([[[int(box[i][0]),int(box[i][1])] for i in range(4)],[pair[0],pair[1]]])
            return result
        target.predict=MethodType(predict,target)
