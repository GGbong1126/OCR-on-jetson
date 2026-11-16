"""N1 FP32 det only; construct/use/close on the WorkerBridge GPU owner.

shortcut: other models and non-FP32 outputs are not adapted; add their typed
contracts and ownership regressions before using the complete OCR pipeline.
"""
import threading

import numpy as np

from stage_session import require
from worker_bridge import Tensor


class DetRuntime:
    def __init__(self, engine, sha256):
        from trt82_backend import Session
        self.owner=threading.get_ident()
        self.session=Session(engine,sha256,'det')

    def _owned(self): require(threading.get_ident()==self.owner)

    def run(self, model, tensors):
        self._owned()
        require(model=='det' and len(tensors)==1 and tensors[0].name=='x')
        item=tensors[0]
        require(len(item.shape)==4 and item.shape[:2]==(1,3))
        require(all(32<=n<=960 and n%32==0 for n in item.shape[2:]))
        array=np.frombuffer(item.data,dtype=np.float32).reshape(item.shape)
        values=self.session.run(None,{'x':array})
        require(len(values)==1 and values[0].dtype==np.float32)
        require(values[0].shape==(1,1,*item.shape[2:]) and np.isfinite(values[0]).all())
        return (Tensor('probability',tuple(values[0].shape),values[0].tobytes()),)

    def synchronize(self):
        self._owned(); self.session.cuda.call('cudaDeviceSynchronize')

    def close(self):
        self._owned(); self.session.close()
