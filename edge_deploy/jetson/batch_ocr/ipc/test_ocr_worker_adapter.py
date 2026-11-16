from contextlib import contextmanager
import io
import gc
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
import weakref

try:
    import numpy as np
    from PIL import Image
except ImportError: np=None

from stage_session import StateError
from worker_bridge import Tensor


@unittest.skipIf(np is None,'Installed NumPy/Pillow required; no new dependency installed for tests')
class AdapterTests(unittest.TestCase):
    def test_mixed_output_dtypes_writable_and_no_alias(self):
        from ocr_worker_adapter import tensors_to_arrays
        tensors=(Tensor('boxes',(2,),bytes(8)),Tensor('count',(1,),(3).to_bytes(4,sys.byteorder,signed=True),'int32'))
        arrays=tensors_to_arrays(tensors)
        self.assertEqual(arrays[1].dtype,np.int32); self.assertEqual(int(arrays[1][0]),3)
        self.assertTrue(all(a.flags.writeable and a.flags.owndata for a in arrays))
        arrays[0][0]=123; arrays[1][0]=4
        self.assertEqual(tensors[0].data,bytes(8)); self.assertEqual(int(np.frombuffer(tensors[1].data,np.int32)[0]),3)

    def test_yolo_result_is_copied_and_lock_released_on_failure(self):
        from ocr_worker_adapter import LockedYOLO
        shared=[dict(boxes=np.ones(2,np.float32))]
        target=SimpleNamespace(predict=lambda *a,**k:shared)
        wrapper=LockedYOLO(target); result=wrapper.predict('x'); shared[0]['boxes'][0]=7
        self.assertEqual(result[0]['boxes'][0],1)
        with self.assertRaises(StateError): wrapper.predict('x',stream=True)
        def fail(*_,**__): raise ValueError('synthetic wrapper failure')
        target.predict=fail
        with self.assertRaises(ValueError): wrapper.predict('x')
        self.assertTrue(wrapper.lock.acquire(False)); wrapper.lock.release()

    def fixture(self, folder, *, swallowed=False):
        from ocr_worker_adapter import OCRRuntime, ROLES
        owner=threading.get_ident(); invoked=[]; closed=[]; copies=[]
        class Native:
            def get_outputs(self): return [SimpleNamespace(name='output')]
            def run(self,names,feeds):
                invoked.append(threading.get_ident()); return [np.array([1],np.int32)]
        models={role:Native() for role in (*ROLES,'ner')}
        models['det'].cuda=SimpleNamespace(call=lambda *args:invoked.append(threading.get_ident()))
        class Pipeline:
            _yolo_model=SimpleNamespace(predict=lambda *a,**k:[])
            _ocr_engine=SimpleNamespace(); _layout_fusion=SimpleNamespace(config={})
            def _preprocess(self,*args,**kwargs): return None
            def process(self,path,*,work_dir,image_type,run_ner):
                copies.append(weakref.ref(self))
                assert run_ner is False and Path(path).is_file()
                try:
                    a=models['layout'].run(None,{'image':np.zeros((1,3,800,800),np.float32),
                        'im_shape':np.ones((1,2),np.float32),'scale_factor':np.ones((1,2),np.float32)})
                except Exception:
                    if not swallowed: raise
                    a=[np.array([1])]
                return dict(success=True,data=dict(count=int(a[0][0])))
        def loader(): return None,Pipeline(),SimpleNamespace(models=models),{}
        runtime=OCRRuntime(loader,lambda m:closed.append(threading.get_ident()),folder)
        runtime.test_copies=copies
        stream=io.BytesIO(); Image.new('RGB',(2,2)).save(stream,format='PNG')
        return runtime,stream.getvalue(),invoked,closed,owner

    def test_reference_stays_on_owner_and_model_run_preserves_int32(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime,payload,called,closed,owner=self.fixture(folder)
            data,metrics=runtime.page(payload,3,'a')
            self.assertEqual(data,dict(count=1)); self.assertEqual(metrics['calls'],{'layout':1})
            result=runtime.run('layout',(Tensor('image',(1,),bytes(4)),))
            self.assertEqual(result[0].dtype,'int32')
            runtime.synchronize(); runtime.close()
            self.assertEqual(set(called+closed),{owner}); self.assertEqual(list(Path(folder).iterdir()),[])

    def test_worker_proxy_routes_through_api_and_detaches_numpy(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime,payload,called,_,owner=self.fixture(folder); records=[]; errors=[]; data=[]
            class API:
                job=SimpleNamespace(page=SimpleNamespace(image_id='a',format='encoded'))
                def checkpoint(self): pass
                @contextmanager
                def infer(self,role,tensors):
                    records.append((role,tensors)); yield (Tensor('output',(1,),np.array([2],np.int32).tobytes(),'int32'),)
            def worker():
                try: data.append(runtime.algorithm(payload,3,API()))
                except Exception as e: errors.append(type(e).__name__)
            thread=threading.Thread(target=worker); thread.start(); thread.join(2)
            self.assertFalse(thread.is_alive()); self.assertEqual(errors,[])
            self.assertEqual(data,[b'{"count":2}']); self.assertEqual(called,[])
            self.assertEqual(records[0][0],'layout'); runtime.close()

    def test_swallowed_business_error_cannot_return_partial_success(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime,payload,called,_,_=self.fixture(folder,swallowed=True); rejected=[]
            class API:
                job=SimpleNamespace(page=SimpleNamespace(image_id='a',format='encoded'))
                def checkpoint(self): pass
                @contextmanager
                def infer(self,*_): raise StateError('GPU_FAILED'); yield
            def worker():
                try: runtime.algorithm(payload,3,API())
                except StateError as error: rejected.append(error.code)
            thread=threading.Thread(target=worker); thread.start(); thread.join(2)
            self.assertEqual(rejected,['STATE']); self.assertEqual(called,[]); runtime.close()

    def test_bad_input_and_raw_bgr_are_not_misdecoded(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime,payload,called,_,_=self.fixture(folder)
            api=SimpleNamespace(job=SimpleNamespace(page=SimpleNamespace(format='bgr8')),checkpoint=lambda:None)
            with self.assertRaises(StateError): runtime.page(payload,3,'a',api)
            with self.assertRaises(Exception): runtime.page(b'broken',3,'a')
            with self.assertRaises(StateError): runtime.page(payload,3,'../a')
            with self.assertRaises(StateError): runtime.page(payload,True,'a')
            oversize=io.BytesIO(); Image.new('RGB',(8193,1)).save(oversize,format='PNG')
            with self.assertRaises(StateError): runtime.page(oversize.getvalue(),3,'a')
            self.assertEqual(called,[]); runtime.close()

    def test_model_owner_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime,_,called,_,_=self.fixture(folder); rejected=[]
            def worker():
                try: runtime.synchronize()
                except StateError as error: rejected.append(error.code)
            thread=threading.Thread(target=worker); thread.start(); thread.join(1)
            self.assertEqual(rejected,['STATE']); self.assertEqual(called,[]); runtime.close()

    def test_raw_bgr_uses_original_preprocessor_without_global_patch(self):
        from ocr_worker_adapter import raw_preprocess
        class Pipeline:
            def _preprocess(self,path,work_dir,skip_enhance=True):
                with Image.open(path) as im:
                    im.thumbnail((1500,1500),Image.Resampling.LANCZOS)
                    value=im.copy()
                return value
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'marker.img'; path.touch(); original=Image.open
            a=Pipeline(); b=Pipeline(); data=bytes([0,0,255,255,0,0])*2
            raw_preprocess(a,data,(2,2,3),6,path)
            with a._preprocess(path,folder) as image:
                self.assertEqual(image.size,(2,2)); self.assertEqual(image.getpixel((0,0)),(255,0,0))
                self.assertEqual(image.getpixel((1,0)),(0,0,255))
            self.assertIs(Image.open,original)
            with self.assertRaises(OSError): b._preprocess(path,folder)
            for shape,stride,payload in (((2,2,3),7,data),((2,3,3),9,data),((8193,2,3),6,data)):
                with self.assertRaises(StateError): raw_preprocess(Pipeline(),payload,shape,stride,path)

    def test_det_pair_preserves_slots_and_keeps_original_single_route(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime,_,_,_,_=self.fixture(folder)
            seen=[]
            class Batch:
                def run(self,names,feeds):
                    x=feeds['x']; seen.append(x.shape)
                    return [x[:,:1].copy()]
            runtime.det_batch=Batch()
            tensors=lambda value:(Tensor('x',(1,3,32,32),np.full((1,3,32,32),value,np.float32).tobytes()),)
            result=runtime.run_batch((('det',tensors(2)),('det',tensors(7))))
            self.assertEqual(seen,[(2,3,32,32)])
            self.assertEqual([float(np.frombuffer(r[0].data,np.float32)[0]) for r in result],[2,7])
            self.assertEqual(runtime.batch_calls,1)
            with self.assertRaises(StateError): runtime.run_batch((('rec',tensors(2)),('det',tensors(7))))
            runtime.det_batch=None; runtime.close()

    def test_decode_timing_preserves_lazy_jpeg_draft_and_encoded_output(self):
        from collections import Counter
        from ocr_worker_adapter import timed_decode
        class Pipeline:
            def _preprocess(self,path,work_dir,skip_enhance=True):
                stream=io.BytesIO()
                with Image.open(path) as image:
                    image.thumbnail((1500,1500),Image.Resampling.LANCZOS)
                    image.save(stream,format='JPEG',quality=90)
                return stream.getvalue()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'source.jpg'; Image.new('RGB',(4096,3072),(40,90,180)).save(path,quality=95)
            reference=Pipeline()._preprocess(path,folder); times=Counter(); candidate=Pipeline()
            timed_decode(candidate,path,times)
            self.assertEqual(candidate._preprocess(path,folder),reference)
            self.assertGreater(times['software_source_decode_ms'],0)

    def test_diagnostic_history_is_bounded_across_batches(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime,payload,_,_,_=self.fixture(folder)
            for i in range(9):
                api=SimpleNamespace(job=SimpleNamespace(page=SimpleNamespace(image_id=str(i),format='encoded')),checkpoint=lambda:None)
                runtime.algorithm(payload,3,api)
            self.assertEqual(list(runtime.page_stats),['5','6','7','8']); runtime.close()

    def test_per_page_method_cycles_do_not_retain_pixels_until_later_gc(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime,payload,_,_,_=self.fixture(folder); enabled=gc.isenabled(); gc.disable()
            try:
                for i in range(5): runtime.page(payload,3,str(i))
                self.assertTrue(all(ref() is None for ref in runtime.test_copies))
            finally:
                if enabled: gc.enable()
                runtime.close()


if __name__=='__main__': unittest.main()
