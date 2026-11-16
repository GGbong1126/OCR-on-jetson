"""Original six-model OCR business through WorkerBridge; no HTTP or Extract.

Keep original profiles, explicit OCR CPU routing, and business pre/postprocessing.
Only this isolated process is patched. Never install into a running service.
"""
from collections import Counter
import copy
import io
from pathlib import Path
import tempfile
import threading
import time
from types import FunctionType, MethodType, SimpleNamespace

import numpy as np

from stage_contract import LIMITS, token
from stage_session import encoded, require
from worker_bridge import Tensor

ROLES=('layout','cell','yolo','enhance','det','rec')


def tensors_to_arrays(outputs):
    # Legacy torch.from_numpy/NMS consumers need writable, separately owned storage.
    return [np.frombuffer(t.data,dtype=t.dtype).reshape(t.shape).copy() for t in outputs]


def bind_image_open(pipeline, opener):
    from PIL import Image
    original=pipeline._preprocess.__func__
    image=SimpleNamespace(**vars(Image))
    image.open=opener
    globals=dict(original.__globals__,Image=image)
    method=FunctionType(original.__code__,globals,original.__name__,original.__defaults__,original.__closure__)
    method.__kwdefaults__=original.__kwdefaults__
    pipeline._preprocess=MethodType(method,pipeline)


def raw_preprocess(pipeline, payload, shape, stride, source, timings=None):
    from PIL import Image
    require(type(shape) is tuple and len(shape)==3 and shape[2]==3)
    h,w,_=shape
    require(type(h) is int and type(w) is int and 0<h<=LIMITS['max_dimension'] and
            0<w<=LIMITS['max_dimension'] and h*w<=LIMITS['max_pixels'],'LIMIT')
    require(stride==w*3 and len(payload)==h*stride,'LIMIT')
    def open_image(path,*args,**kwargs):
        if str(path)==str(source):
            require(not args and not kwargs)
            start=time.monotonic(); image=Image.frombytes('RGB',(w,h),payload,'raw','BGR',stride)
            if timings is not None: timings['bgr_to_pil_handoff_ms']+=(time.monotonic()-start)*1000
            return image
        return Image.open(path,*args,**kwargs)
    bind_image_open(pipeline,open_image)
    # Only this page's existing preprocessor sees raw pixels; no global PIL monkeypatch or re-encoding.


def timed_decode(pipeline, source, timings):
    from PIL import Image
    def open_image(path,*args,**kwargs):
        image=Image.open(path,*args,**kwargs)
        if str(path)==str(source):
            original=image.load
            def load(*a,**k):
                pending=bool(getattr(image,'tile',()))
                start=time.monotonic()
                try: return original(*a,**k)
                finally:
                    if pending: timings['software_source_decode_ms']+=(time.monotonic()-start)*1000
            image.load=load
        return image
    bind_image_open(pipeline,open_image)


class LockedYOLO:
    def __init__(self, target): self.target=target; self.lock=threading.Lock()

    def predict(self, *args, **kwargs):
        require(not kwargs.get('stream',False))
        with self.lock: return copy.deepcopy(self.target.predict(*args,**kwargs))


class OCRRuntime:
    def __init__(self, load_business, close_models, work_root, *, metrics_limit=4, det_batch_factory=None,public_response=False):
        self.owner=threading.get_ident(); self.local=threading.local()
        require(type(metrics_limit) is int and 1<=metrics_limit<=32)
        self.metrics_limit=metrics_limit
        self.work_root=Path(work_root); require(self.work_root.is_dir())
        self.server,self.pipeline,self.models,self.config=load_business()
        require(type(public_response) is bool)
        from ocr_response import renderer
        self.render=renderer(self.server) if public_response else None
        self.cleanup=close_models; self.closed=False
        self.original={role:self.models.models[role].run for role in ROLES}
        self.names={role:tuple(v.name for v in self.models.models[role].get_outputs()) for role in ROLES}
        self.output_dtypes={role:set() for role in ROLES}
        self.cuda=getattr(self.models.models['det'],'gpu',self.models.models['det']).cuda
        self.page_stats={}; self.stats_lock=threading.Lock()
        self.det_batch=None if det_batch_factory is None else det_batch_factory()
        self.batch_calls=0
        for role in ROLES: self.models.models[role].run=self.proxy(role)
        self.pipeline._yolo_model=LockedYOLO(self.pipeline._yolo_model)

    def _owned(self): require(threading.get_ident()==self.owner)

    def proxy(self, role):
        def run(names, feeds):
            require(names in (None,[])); require(type(feeds) is dict)
            start=time.monotonic(); marker=getattr(self.local,'marker',None)
            if marker is not None and marker[0] is None: marker[0]=start
            try:
                if threading.get_ident()==self.owner:
                    out=self.original[role](names,feeds)
                else:
                    api=self.local.api; api.checkpoint()
                    require(all(isinstance(v,np.ndarray) and v.dtype==np.float32 for v in feeds.values()))
                    require(sum(v.nbytes for v in feeds.values())<=LIMITS['max_tensor_bytes'],'LIMIT')
                    tensors=tuple(Tensor(n,tuple(v.shape),np.ascontiguousarray(v).tobytes()) for n,v in feeds.items())
                    with api.infer(role,tensors) as outputs: out=tensors_to_arrays(outputs)
                self.local.calls[role]+=1
                if role=='rec': self.local.widths.append(feeds['x'].shape[3])
                return out
            except Exception:
                self.local.failed=True
                raise
            finally:
                end=time.monotonic()
                if marker is not None: marker[1]=end
                if getattr(self.local,'times',None) is not None: self.local.times['model_proxy_wall_ms']+=(end-start)*1000
        return run

    def time_method(self, target, name, label, *, split=False):
        if not hasattr(target,name): return
        original=getattr(target,name)
        def timed(_,*args,**kwargs):
            start=time.monotonic(); previous=self.local.marker; marker=[None,None]
            before=self.local.times['model_proxy_wall_ms']
            if split: self.local.marker=marker
            try: return original(*args,**kwargs)
            finally:
                end=time.monotonic(); self.local.marker=previous
                self.local.times[label+'_wall_ms']+=(end-start)*1000
                self.local.times[label+'_non_model_wall_ms']+=(end-start)*1000-(self.local.times['model_proxy_wall_ms']-before)
                if split and marker[0] is not None:
                    self.local.times[label+'_pre_ms']+=(marker[0]-start)*1000
                    self.local.times[label+'_post_ms']+=(end-marker[1])*1000
        setattr(target,name,MethodType(timed,target))

    def metrics(self):
        with self.stats_lock: return copy.deepcopy(self.page_stats)

    def run(self, model, tensors):
        self._owned(); require(model in ROLES and not self.closed)
        require(all(t.dtype=='float32' for t in tensors))
        feeds={t.name:np.frombuffer(t.data,np.float32).reshape(t.shape) for t in tensors}
        require(len(feeds)==len(tensors))
        values=self.original[model](None,feeds)
        require(len(values)==len(self.names[model]))
        require(all(v.dtype in (np.dtype('float32'),np.dtype('int32')) and np.isfinite(v).all() for v in values))
        require(sum(v.nbytes for v in values)<=LIMITS['max_tensor_bytes'],'LIMIT')
        self.output_dtypes[model].update(v.dtype.name for v in values)
        return tuple(Tensor(n,tuple(v.shape),v.tobytes(),v.dtype.name) for n,v in zip(self.names[model],values))

    def run_batch(self, requests):
        self._owned(); require(self.det_batch is not None and not self.closed and len(requests)==2)
        require(all(model=='det' and len(tensors)==1 and tensors[0].name=='x' and tensors[0].dtype=='float32'
                    for model,tensors in requests))
        a,b=(tensors[0] for _,tensors in requests)
        require(a.shape==b.shape and len(a.shape)==4 and a.shape[:2]==(1,3))
        x=np.concatenate([np.frombuffer(t.data,np.float32).reshape(t.shape) for t in (a,b)])
        values=self.det_batch.run(None,{'x':x})
        require(len(values)==1 and values[0].dtype==np.float32 and values[0].shape==(2,1,*x.shape[2:]))
        require(np.isfinite(values[0]).all()); self.batch_calls+=1
        return tuple((Tensor(self.names['det'][0],(1,1,*x.shape[2:]),values[0][i:i+1].tobytes()),) for i in range(2))

    def stats(self):
        value=self.models.stats()
        if self.det_batch is not None: value['det_batch']=self.det_batch.stats()
        return value

    def synchronize(self): self._owned(); self.cuda.call('cudaDeviceSynchronize')

    def close(self):
        self._owned()
        # Leave proxies installed: a cancelled CPU finalizer must not regain direct GPU access.
        if self.det_batch is not None: self.det_batch.close()
        self.cleanup(self.models); self.closed=True

    def page(self, payload, kind, image_id, api=None):
        from PIL import Image
        require(token(image_id) and type(payload) is bytes)
        require(type(kind) is int and kind in (1,2,3))
        if api is None: self._owned()
        else: api.checkpoint()
        raw=api is not None and api.job.page.format=='bgr8'
        if not raw:
            require(api is None or api.job.page.format=='encoded')
            require(len(payload)<=LIMITS['max_encoded_bytes'],'LIMIT')
            with Image.open(io.BytesIO(payload)) as header:
                w,h=header.size
                require(0<w<=LIMITS['max_dimension'] and 0<h<=LIMITS['max_dimension'] and w*h<=LIMITS['max_pixels'],'LIMIT')
                header.verify()
        pipeline=copy.copy(self.pipeline)
        pipeline._ocr_engine=copy.copy(self.pipeline._ocr_engine)
        pipeline._layout_fusion=copy.deepcopy(self.pipeline._layout_fusion)
        pipeline_state=pipeline.__dict__.copy(); engine_state=pipeline._ocr_engine.__dict__.copy()
        self.local.api=api; self.local.calls=Counter(); self.local.widths=[]; self.local.failed=False
        self.local.times=Counter(); self.local.marker=None
        try:
            with tempfile.TemporaryDirectory(prefix='page_',dir=self.work_root) as folder:
                path=Path(folder)/ (image_id+'.img')
                if raw:
                    raw_preprocess(pipeline,payload,getattr(api.job.page,'shape',()),getattr(api.job.page,'stride',0),path,self.local.times); path.touch()
                else:
                    path.write_bytes(payload)
                    if hasattr(pipeline,'_preprocess'): timed_decode(pipeline,path,self.local.times)
                self.time_method(pipeline,'_preprocess','resize_encode')
                self.time_method(pipeline,'_detect_layout','layout_cpu_and_io')
                self.time_method(pipeline,'_enhance_single','enhance_cpu_and_io')
                self.time_method(pipeline,'_run_ocr','ocr_business')
                self.time_method(pipeline._ocr_engine,'_det_run','det',split=True)
                self.time_method(pipeline._ocr_engine,'_rec_run','rec',split=True)
                result=pipeline.process(str(path),work_dir=str(Path(folder)/'work'),image_type=kind,run_ner=False)
            if api is not None: api.checkpoint()
            require(result['success'] and not self.local.failed)
            metrics=dict(calls=dict(self.local.calls),rec_widths=list(self.local.widths),cpu_thread=threading.get_ident(),
                         timings=dict(self.local.times),image_source='bgr8' if raw else 'encoded')
            if api is not None: metrics['timings'].update(getattr(api.job,'timings',{}))
            if raw: h,w,_=api.job.page.shape
            data=self.render(result,w,h) if self.render is not None else result['data']; encoded(data)
            return data,metrics
        finally:
            # Remove per-page bound-method cycles before credit release, not at a later cyclic GC.
            pipeline._ocr_engine.__dict__.clear(); pipeline._ocr_engine.__dict__.update(engine_state)
            pipeline.__dict__.clear(); pipeline.__dict__.update(pipeline_state)
            self.local.api=None; self.local.calls=None; self.local.widths=None; self.local.times=None; self.local.marker=None

    def algorithm(self, payload, kind, api):
        name=api.job.page.image_id
        data,metrics=self.page(payload,kind,name,api)
        with self.stats_lock:
            if name not in self.page_stats and len(self.page_stats)>=self.metrics_limit:
                self.page_stats.pop(next(iter(self.page_stats)))
            self.page_stats[name]=metrics
        return encoded(data)
