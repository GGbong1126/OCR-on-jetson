import base64
import io
import threading
import unittest
from unittest.mock import patch

try:
    import flask
    from flask.testing import FlaskClient
except ImportError: flask=None

if flask is not None:
    class ConsumingClient(FlaskClient):
        def open(self,*args,**kwargs):
            kwargs.setdefault('buffered',True)
            return super().open(*args,**kwargs)


class Fake:
    def __init__(self): self.calls=[]; self.available=True; self.block=None; self.enter=threading.Event(); self.fail=False
    def health(self): return dict(status='ready' if self.available else 'error',scope='batch_ocr_no_extract')
    def infer(self, images):
        self.calls.append(images); self.enter.set()
        if self.block: assert self.block.wait(3)
        if self.fail: raise RuntimeError('synthetic backend fault')
        return dict(results=[dict(image_id=name,status='error',code='STATE') if data==b'bad' else
                    dict(image_id=name,status='ok',data=dict(synthetic=True,image_type=kind)) for name,kind,data in images],execution=dict(synthetic=True))


@unittest.skipIf(flask is None,'Use existing Jetson Flask; no dependency installed for host tests')
class HttpTests(unittest.TestCase):
    def setUp(self):
        from serve_http import create_app
        self.service=Fake(); self.app=create_app(self.service); self.app.test_client_class=ConsumingClient
        self.client=self.app.test_client()
        self.good=dict(images=[dict(imageId='a',imageType=3,imageBase64=base64.b64encode(b'x').decode())])
    def test_success_order_and_no_legacy_routes(self):
        value=dict(images=self.good['images']+[dict(imageId='b',imageType=1,imageBase64='eQ==')])
        response=self.client.post('/OCRBatch',json=value)
        self.assertEqual(response.status_code,200); self.assertEqual(response.json['status'],'ok')
        self.assertEqual([r['image_id'] for r in response.json['results']],['a','b'])
        self.assertEqual(self.service.calls[0],[('a',3,b'x'),('b',1,b'y')])
        self.assertEqual(self.client.post('/Extract',json={}).status_code,404)
        self.assertEqual(self.client.post('/OCR',json={}).status_code,404)
    def test_invalid_requests_do_not_reach_models(self):
        bad=[None,{},dict(images=[]),dict(images=self.good['images']*2),dict(images=[dict(self.good['images'][0],imageType=True)]),
             dict(images=[dict(self.good['images'][0],imageBase64='bad\n')]),dict(images=[dict(self.good['images'][0],imageId='../x')]),
             dict(images=[dict(self.good['images'][0],path='/tmp/x')])]
        for value in bad:
            response=self.client.post('/OCRBatch',json=value)
            self.assertIn(response.status_code,(400,411,415))
        self.assertEqual(self.service.calls,[])
        self.assertEqual(self.client.post('/OCRBatch',data='x',content_type='text/plain').status_code,415)
    def test_large_payload_and_missing_length_are_bounded(self):
        response=self.client.post('/OCRBatch',data=b'x'*(24*1024**2+1),content_type='application/json')
        self.assertEqual(response.status_code,413); self.assertEqual(response.json['code'],'LIMIT')
        response=self.client.post('/OCRBatch',data='{}',content_type='application/json',environ_overrides={'CONTENT_LENGTH':''})
        self.assertEqual(response.status_code,411); self.assertEqual(self.service.calls,[])
    def test_busy_backpressure_keeps_health_responsive(self):
        self.service.block=threading.Event(); responses=[]
        def worker():
            with self.app.test_client() as client: responses.append(client.post('/OCRBatch',json=self.good))
        thread=threading.Thread(target=worker); thread.start(); self.assertTrue(self.service.enter.wait(1))
        try:
            self.assertEqual(self.client.post('/OCRBatch',json=self.good).status_code,429)
            self.assertTrue(self.client.get('/healthz').json['busy']); self.assertEqual(len(self.service.calls),1)
        finally: self.service.block.set(); thread.join(2)
        self.assertFalse(thread.is_alive()); self.assertEqual(responses[0].status_code,200)
        self.assertFalse(self.client.get('/healthz').json['busy'])
    def test_busy_upload_is_drained_in_bounded_chunks_without_decoding(self):
        class Stream(io.BytesIO):
            largest=0
            def read(self,size=-1): self.largest=max(self.largest,size); return super().read(size)
            def readinto(self,buffer): self.largest=max(self.largest,len(buffer)); return super().readinto(buffer)
        self.service.block=threading.Event()
        def worker():
            with self.app.test_client() as client: client.post('/OCRBatch',json=self.good)
        thread=threading.Thread(target=worker); thread.start(); self.assertTrue(self.service.enter.wait(1))
        stream=Stream(b'not-json'*131072)
        try:
            response=self.client.post('/OCRBatch',environ_overrides={'wsgi.input':stream,
                'CONTENT_LENGTH':str(len(stream.getvalue())),'CONTENT_TYPE':'application/json'})
            self.assertEqual(response.status_code,429); self.assertEqual(stream.tell(),1048576)
            self.assertLessEqual(stream.largest,65536); self.assertEqual(len(self.service.calls),1)
        finally: self.service.block.set(); thread.join(2)
        self.assertFalse(thread.is_alive())
    def test_per_image_failure_and_backend_fault_never_claim_success(self):
        value=dict(images=[dict(self.good['images'][0],imageBase64=base64.b64encode(b'bad').decode())])
        response=self.client.post('/OCRBatch',json=value)
        self.assertEqual(response.status_code,200); self.assertEqual(response.json['status'],'partial')
        self.service.fail=True
        self.assertEqual(self.client.post('/OCRBatch',json=self.good).status_code,503)
        self.service.fail=False
        self.assertEqual(self.client.post('/OCRBatch',json=self.good).status_code,200)
        self.service.available=False
        self.assertEqual(self.client.get('/healthz').status_code,503)
        self.assertEqual(self.client.post('/OCRBatch',json=self.good).status_code,503)

    def test_absolute_upload_deadline_releases_gate_without_inference(self):
        with patch('serve_http.time.monotonic',side_effect=[0,6]):
            response=self.client.post('/OCRBatch',json=self.good)
        self.assertEqual(response.status_code,504); self.assertEqual(self.service.calls,[])
        self.assertEqual(self.client.post('/OCRBatch',json=self.good).status_code,200)

    def test_werkzeug_wrapped_socket_timeout_is_still_504(self):
        from werkzeug.exceptions import ClientDisconnected
        class TimedOut(io.BytesIO):
            def readinto(self,buffer): raise ClientDisconnected()
        with patch('serve_http.time.monotonic',side_effect=[0,.1,6]):
            response=self.client.post('/OCRBatch',environ_overrides={
                'wsgi.input':TimedOut(b'x'),'CONTENT_LENGTH':'1','CONTENT_TYPE':'application/json'})
        self.assertEqual(response.status_code,504); self.assertEqual(self.service.calls,[])
        self.assertFalse(self.client.get('/healthz').json['busy'])

    def test_duplicate_json_keys_and_invalid_unicode_never_reach_models(self):
        for body in (b'{"images":[],"images":[]}',b'{"images":"\\ud800"}',b'{"images":NaN}'):
            self.assertEqual(self.client.post('/OCRBatch',data=body,content_type='application/json').status_code,400)
        self.assertEqual(self.service.calls,[])

    def test_result_keeps_reservation_until_wsgi_closes_the_body(self):
        response=self.client.post('/OCRBatch',json=self.good,buffered=False)
        self.assertEqual(response.status_code,200)
        state=self.client.get('/healthz').json
        self.assertTrue(state['busy']); self.assertEqual(state['front_reservation_bytes'],256*1024**2)
        self.assertEqual(self.client.post('/OCRBatch',json=self.good).status_code,429)
        self.assertEqual(len(self.service.calls),1)
        response.close(); response.close()
        self.assertEqual(self.client.get('/healthz').json['front_reservation_bytes'],0)
        self.assertEqual(self.client.post('/OCRBatch',json=self.good).status_code,200)


if __name__=='__main__': unittest.main()
