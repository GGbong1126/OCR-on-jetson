import copy
import json
import socket
import sys
import time
import unittest

import ipc_wire as w
import stage_contract as s


def base(op,**fields):
    return dict(schema='ocr.stage/1',op=op,session='a'*32,remaining_ms=5000,**fields)


def raw(meta): return json.dumps(meta,separators=(',',':')).encode()


def ready(image='a',call=1,width=704,model='det'):
    return base('model.ready',image_id=image,image_type=3,call_id=call,model=model,
                inputs=[dict(name='x',handle=call,dtype='float32',shape=[1,3,960,width],bytes=1*3*960*width*4)])


class StageTests(unittest.TestCase):
    def test_limits_and_reservation_arithmetic(self):
        self.assertFalse(s.validate_limits(s.LIMITS)['det_batch_enabled'])
        self.assertEqual(s.LIMITS['explicit_budget_bytes'],768*1024**2)
        for changes in ({'gpu_dispatchers':2},{'page_reservation_bytes':1},{'input_workers':True},
                        {'explicit_budget_bytes':1},{'arbitrary':1},{'result_queue_bytes':2**30}):
            with self.assertRaises(w.ProtocolError): s.validate_limits(dict(s.LIMITS,**changes))

    def test_submit_and_no_filesystem_paths(self):
        good=base('image.submit',image_id='img_1',image_type=1,format='encoded')
        self.assertEqual(s.validate_stage(raw(good),1024),good)
        for value,n in ((dict(good,path='/tmp/x'),1),(good,0),(good,16*1024**2+1),
                        (dict(good,image_type=True),1),(dict(good,image_id='../x'),1)):
            with self.assertRaises(w.ProtocolError): s.validate_stage(raw(value),n)
        bgr=base('image.submit',image_id='x',image_type=3,format='bgr8',shape=[2,4,3],stride=12)
        s.validate_stage(raw(bgr),24)
        with self.assertRaises(w.ProtocolError): s.validate_stage(raw(bgr),23)

    def test_tensor_specs_checked_without_allocating_tensors(self):
        value=ready(); s.validate_stage(raw(value),0)
        for field,new in (('bytes',True),('bytes',1),('shape',[2,3,960,704]),
                          ('shape',[1,3,961,704]),('shape',[1,3,960,999999]),('dtype','float16')):
            changed=copy.deepcopy(value); changed['inputs'][0][field]=new
            with self.assertRaises(w.ProtocolError): s.validate_stage(raw(changed),0)
        changed=dict(value,model='ner')
        with self.assertRaises(w.ProtocolError): s.validate_stage(raw(changed),0)

    def test_binding_sets_and_rec_width(self):
        for model,size in (('layout',800),('cell',640)):
            value=base('model.ready',image_id='a',image_type=3,call_id=1,model=model,inputs=[
                dict(name='image',handle=1,dtype='float32',shape=[1,3,size,size],bytes=12*size*size),
                dict(name='im_shape',handle=2,dtype='float32',shape=[1,2],bytes=8),
                dict(name='scale_factor',handle=3,dtype='float32',shape=[1,2],bytes=8)])
            s.validate_stage(raw(value),0)
            value['inputs'][2]['name']='im_shape'
            with self.assertRaises(w.ProtocolError): s.validate_stage(raw(value),0)
        value=ready(model='rec'); value['inputs'][0].update(shape=[1,3,48,2304],bytes=12*48*2304)
        s.validate_stage(raw(value),0)
        for width in (2305,3200):
            value['inputs'][0].update(shape=[1,3,48,width],bytes=12*48*width)
            s.validate_stage(raw(value),0)
        value['inputs'][0].update(shape=[1,3,48,3201],bytes=12*48*3201)
        with self.assertRaises(w.ProtocolError): s.validate_stage(raw(value),0)

    def test_batch_opt_in_exact_shape_and_distinct_images(self):
        a,b=ready(),ready('b',2); request=base('model.run',call_ids=[1,2]); rows={1:a,2:b}
        with self.assertRaises(w.ProtocolError): s.validate_run(raw(request),rows)
        s.validate_run(raw(request),rows,allow_det_batch=True)
        for changed in (ready('a',2),ready('b',2,672),dict(b,image_type=1),dict(b,session='b'*32)):
            with self.assertRaises(w.ProtocolError): s.validate_run(raw(request),{1:a,2:changed},allow_det_batch=True)
        with self.assertRaises(w.ProtocolError): s.validate_run(raw(base('model.run',call_ids=[1,1])),rows,allow_det_batch=True)
        with self.assertRaises(w.ProtocolError): s.validate_run(raw(base('model.run',call_ids=[3])),rows)

    def test_resume_collect_cancel_and_result_limits(self):
        for op in ('image.collect','image.cancel'):
            s.validate_stage(raw(base(op,image_id='a')),0)
        s.validate_stage(raw(base('model.resume',call_id=1)),0)
        s.validate_stage(raw(base('model.done',call_ids=[1,2],output_bytes=[10,10])),0)
        result=base('image.result',image_id='a',status='ok'); s.validate_stage(raw(result),100)
        for meta,n in ((result,8*1024**2+1),(dict(result,status='error'),1),
                       (base('model.done',call_ids=[1],output_bytes=[10,20]),0)):
            with self.assertRaises(w.ProtocolError): s.validate_stage(raw(meta),n)

    def test_strict_json_and_session(self):
        value=base('image.collect',image_id='a'); encoded=raw(value)
        for data in (encoded[:-1]+b',"op":"image.collect"}',raw(dict(value,op='Extract')),
                     raw(dict(value,session='x')),raw(dict(value,remaining_ms=True)),raw(dict(value,extra=1))):
            with self.assertRaises(w.ProtocolError): s.validate_stage(data,0)

    @unittest.skipUnless(sys.platform=='linux','Unix socketpair')
    def test_stage_codec_is_explicit_and_probe_default_unchanged(self):
        metadata=raw(base('image.collect',image_id='a')); frame=w.Frame(7,metadata,b'')
        a,b=socket.socketpair()
        try:
            with self.assertRaises(w.ProtocolError): w.send_frame(a,frame,time.monotonic()+1)
            w.send_frame(a,frame,time.monotonic()+1,validator=s.validate_stage)
            self.assertEqual(w.receive_frame(b,time.monotonic()+1,validator=s.validate_stage),frame)
        finally: a.close(); b.close()


if __name__=='__main__': unittest.main()
