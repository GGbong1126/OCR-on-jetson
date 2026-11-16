"""Existing C++ stage/2 pipeline with real owner threads and CPU-only model mocks."""
import os
from pathlib import Path
import sys
import time
import unittest

root=Path(os.environ['AUTOOCR_THREAD_SOURCE'])
sys.path[:0]=[str(root/'multistream'),str(root/'ipc'),str(root/'app')]
import test_multibridge as base
from threaded_rec import Pool,Reply

Original=base.Runtime


class Runtime(Original):
    def __init__(self,fail=False):
        super().__init__(fail)
        parent=self
        class Resource:
            context=1
            def __init__(self,index): self.owner=__import__('threading').get_ident()
            def check(self): assert self.owner==__import__('threading').get_ident()
            def run(self,request):
                self.check(); time.sleep(.001)
                if parent.fail: raise RuntimeError('fake failure')
                return (base.Tensor('out',(1,1,1),request.tensors[0].data[:4]),),{}
            def synchronize(self): self.check()
            def close(self): self.check()
            def stats(self): return dict(fixed_io_bytes=1)
        self.pool=Pool(4,Resource)
    def submit_rec(self,slot,ticket,request):
        self.check(); self.calls+=1; return self.pool.submit(slot-1,request)
    def collect(self,event,role):
        if isinstance(event,Reply): return event.result()
        return super().collect(event,role)
    def synchronize(self): self.check(); self.pool.synchronize()
    def close(self): self.check(); self.pool.close(); super().close()


class Tests(unittest.TestCase):
    def setUp(self): base.Runtime=Runtime
    def tearDown(self): base.Runtime=Original
    def test_normal_zero_and_cpu(self):
        case=base.PipelineTests()
        for slots in (1,2,4): case.exercise(slots)
        case.exercise(4,zero=True); case.exercise(2,cpu=True)
    def test_decode_cancel(self): base.PipelineTests().exercise(2,fail_decode=True)
    def test_cpu_no_hol(self): base.PipelineTests().test_busy_cpu_route_does_not_head_of_line_block_gpu()


if __name__=='__main__': unittest.main()
