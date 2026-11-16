import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import ipc_wire as wire
import stage_contract as contract
from stage_session import Session, encoded
from test_stage_contract import base, ready


def metadata_cases():
    valid=[(base('session.ready'),0),
           (base('channel.bind',channel='input'),0),(base('channel.bound',channel='events'),0),
           (base('rpc.reply',for_op='image.submit',status='ok',code='OK'),0),
           (base('rpc.reply',for_op='image.collect',status='error',code='STATE'),0),
           (base('image.submit',image_id='a',image_type=3,format='encoded'),1),
           (base('image.submit',image_id='a',image_type=1,format='bgr8',shape=[2,4,3],stride=12),24),
           (ready(),0),(base('model.run',call_ids=[1,2]),0),
           (base('model.resume',call_id=1),0),(base('image.collect',image_id='a'),0),
           (base('image.cancel',image_id='a'),0),
           (base('model.done',call_ids=[1,2],output_bytes=[0,64*1024**2]),0),
           (base('model.failed',call_ids=[1],code='GPU_FAILED'),0),
           (base('image.done',image_id='a',status='ok'),0),
           (base('image.done',image_id='a',status='error',code='STATE'),0),
           (base('image.done',image_id='a',status='cancelled',code='CANCELLED'),0),
           (base('image.result',image_id='a',status='ok'),18),
           (base('image.result',image_id='a',status='error',code='LIMIT'),0),
           (base('image.result',image_id='a',status='cancelled',code='CANCELLED'),0)]
    for model,size in (('layout',800),('cell',640)):
        valid.append((base('model.ready',image_id='a',image_type=3,call_id=1,model=model,inputs=[
            dict(name='image',handle=1,dtype='float32',shape=[1,3,size,size],bytes=12*size*size),
            dict(name='im_shape',handle=2,dtype='float32',shape=[1,2],bytes=8),
            dict(name='scale_factor',handle=3,dtype='float32',shape=[1,2],bytes=8)]),0))
    for model,name,shape in (('rec','x',[1,3,48,2304]),('yolo','images',[1,3,64,640]),
                             ('enhance','images',[1,3,512,512])):
        size=4
        for n in shape: size*=n
        valid.append((base('model.ready',image_id='a',image_type=3,call_id=2,model=model,inputs=[
            dict(name=name,handle=5,dtype='float32',shape=shape,bytes=size)]),0))
    for width in (2305,3200):
        m=ready(model='rec'); m['inputs'][0].update(shape=[1,3,48,width],bytes=12*48*width)
        valid.append((m,0))
    cases=[]
    for i,(meta,n) in enumerate(valid):
        cases.append((f'valid_{i}',encoded(meta),n,True))
        for key in ('schema','op','session','remaining_ms'):
            bad=dict(meta); del bad[key]
            cases.append((f'missing_{i}_{key}',encoded(bad),n,False))
        for label,extra in (('extra',{'extra':1}),('session',{'session':'A'*32}),
                            ('bool_ms',{'remaining_ms':True}),('zero_ms',{'remaining_ms':0}),
                            ('float_ms',{'remaining_ms':1.0}),('large_ms',{'remaining_ms':300001}),
                            ('extract',{'op':'Extract'})):
            cases.append((f'{label}_{i}',encoded(dict(meta,**extra)),n,False))
    for key,value in (('bytes',1),('bytes',True),('shape',[2,3,960,704]),
                      ('shape',[1,3,961,704]),('shape',[8192]*4),('dtype','float16'),
                      ('handle',0),('handle',2**63),('name','other')):
        m=ready(); m['inputs'][0][key]=value
        cases.append((f'tensor_{key}_{value}',encoded(m),0,False))
    nested=encoded(ready()).replace(b'"handle":1',b'"handle":1,"handle":1')
    common=encoded(base('session.ready'))
    malformed=[b'\xff',b'{}',b'[]',b'['*2000+b']'*2000,b'x'*4097,
               b'\xef\xbb\xbf'+common,common+b'{}',common[:-1]+b',"op":"session.ready"}',
               common[:-1]+b',"\\u006fp":"session.ready"}',common.replace(b'5000',b'1e999'),
               common.replace(b'5000',b'NaN'),nested,common+b'\0',common+b'\0{}',
               b'/*comment*/'+common,common[:-1]+b',}']
    cases.extend((f'parse_{i}',raw,0,False) for i,raw in enumerate(malformed))
    cases.append(('forbidden_payload',common,1,False))
    cases.append(('wire_oversize',common,wire.MAX_PAYLOAD+1,False))
    m=ready(model='rec'); m['inputs'][0].update(shape=[1,3,48,3201],bytes=12*48*3201)
    cases.append(('rec_over_original_preprocess_cap',encoded(m),0,False))
    return cases


def result_cases():
    within=b'{"x":'+b'['*63+b'0'+b']'*63+b'}'
    too_deep=b'{"x":'+b'['*64+b'0'+b']'*64+b'}'
    return [('object',b'{}',True),('unicode',b'{"x":"\xe4\xb8\xad\\u0000"}',True),
            ('depth64',within,True),('depth65',too_deep,False),('array',b'[]',False),
            ('nan',b'{"x":NaN}',False),('overflow',b'{"x":1e999}',False),
            ('duplicate',b'{"x":{"a":1,"a":2}}',False),('surrogate',b'{"x":"\\ud800"}',False),
            ('invalid_utf8',b'{"x":"\xff"}',False),('trailing',b'{}{}',False),('empty',b'',False),
            ('raw_nul',b'{}\0',False),('max_bytes',b' '*(8*1024**2-2)+b'{}',True),
            ('oversize_bytes',b' '*(8*1024**2-1)+b'{}',False)]


class CorpusTests(unittest.TestCase):
    def test_python_metadata_fixture_expectations(self):
        for label,raw,n,expected in metadata_cases():
            with self.subTest(label=label):
                try: contract.validate_stage(raw,n); got=True
                except wire.ProtocolError: got=False
                self.assertEqual(got,expected)

    def test_result_integrity_limits(self):
        for label,raw,expected in result_cases():
            with self.subTest(label=label):
                try: contract.validate_result(raw); got=True
                except wire.ProtocolError: got=False
                self.assertEqual(got,expected)


@unittest.skipUnless(sys.platform=='linux' and os.environ.get('AUTOOCR_STAGE_PROBE'),'Compiled Linux C++ stage client required')
class CppStageTests(unittest.TestCase):
    def test_cpp_metadata_corpus_matches_python(self):
        for label,raw,n,expected in metadata_cases():
            with self.subTest(label=label):
                p=subprocess.run([os.environ['AUTOOCR_STAGE_PROBE'],'--metadata',str(n)],input=raw,capture_output=True,timeout=5)
                self.assertEqual(p.returncode,0 if expected else 1)
                self.assertEqual(json.loads(p.stdout if expected else p.stderr)['accepted'],expected)

    def test_cpp_result_corpus_matches_python(self):
        for label,raw,expected in result_cases():
            with self.subTest(label=label):
                p=subprocess.run([os.environ['AUTOOCR_STAGE_PROBE'],'--result'],input=raw,capture_output=True,timeout=5)
                self.assertEqual(p.returncode,0 if expected else 1)
                self.assertEqual(json.loads(p.stdout if expected else p.stderr)['accepted'],expected)

    def test_cpp_self_test(self):
        p=subprocess.run([os.environ['AUTOOCR_STAGE_PROBE'],'--self-test'],capture_output=True,text=True,timeout=5)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(json.loads(p.stdout)['status'],'cpp_stage_self_test_passed')

    def exchange(self, mode):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path=Path(folder)/'stage.sock'
            with wire.private_listener(path) as listener:
                listener.settimeout(3)
                p=subprocess.Popen([os.environ['AUTOOCR_STAGE_PROBE'],str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                session=Session(); event_id=0; requests=0
                try:
                    with listener.accept()[0] as connection:
                        wire.check_peer(connection)
                        def send_event(meta):
                            nonlocal event_id
                            event_id+=1
                            wire.send_frame(connection,wire.Frame(event_id,encoded(meta),b''),time.monotonic()+2,validator=contract.validate_stage)
                        if mode=='timeout':
                            out,err=p.communicate(timeout=4)
                            self.assertEqual(p.returncode,1); self.assertEqual(json.loads(err)['kind'],'timeout')
                            return
                        send_event(session.hello())
                        while requests<9:
                            frame=wire.receive_frame(connection,time.monotonic()+2,validator=contract.validate_stage)
                            meta=contract.validate_stage(frame.metadata,len(frame.payload)); op=meta['op']; requests+=1
                            channel='input' if op=='image.submit' else ('gpu' if op=='model.run' else 'result')
                            reply=session.handle(frame,channel)
                            if mode!='normal':
                                bad=contract.validate_stage(reply.frame.metadata,0); rid=frame.request_id
                                if mode=='wrong_session': bad['session']='b'*32
                                elif mode=='wrong_id': rid+=1
                                elif mode=='wrong_for_op': bad['for_op']='image.cancel'
                                if mode=='invalid_before_payload':
                                    connection.sendall(wire.header(rid,len(encoded(bad)),wire.MAX_PAYLOAD)+encoded(bad))
                                elif mode=='duplicate_key':
                                    raw=encoded(bad)[:-1]+b',"code":"OK"}'
                                    connection.sendall(wire.header(rid,len(raw),0)+raw)
                                else: wire.send_frame(connection,wire.Frame(rid,encoded(bad),b''),time.monotonic()+2,validator=contract.validate_stage)
                                out,err=p.communicate(timeout=4)
                                self.assertEqual(p.returncode,1)
                                self.assertEqual(json.loads(err)['kind'],'protocol_or_system')
                                return
                            wire.send_frame(connection,reply.frame,time.monotonic()+2,validator=contract.validate_stage)
                            if reply.delivery: session.delivered(reply)
                            if op=='image.submit' and session.stats()['images']==2:
                                for image in ('b','a'):
                                    page=session.page(image); session.activate(page)
                                    send_event(session.ready(page,'det',[('x',[1,3,32,32],bytes(12288))]))
                            elif op=='model.run':
                                send_event(session.complete(session.running,[(bytes(4),)]))
                            elif op=='model.resume':
                                page=next(page for page in session._pages.values() if page.call and page.call.meta['call_id']==meta['call_id'])
                                session.release_call(page); session.finish(page,encoded(dict(synthetic=True,image_id=page.image_id)))
                        self.assertEqual(session.stats()['images'],0)
                    out,err=p.communicate(timeout=4); self.assertEqual(p.returncode,0,err)
                    result=json.loads(out); self.assertEqual(result['images'],2); self.assertEqual(result['requests'],9)
                    self.assertTrue(result['reverse_collection'])
                finally:
                    session.disconnect()
                    if p.poll() is None: p.kill()
                    p.communicate(timeout=4)
            self.assertFalse(path.exists())

    def test_real_cpp_session_stage_roundtrip(self): self.exchange('normal')
    def test_wrong_session(self): self.exchange('wrong_session')
    def test_wrong_id(self): self.exchange('wrong_id')
    def test_wrong_operation_correlation(self): self.exchange('wrong_for_op')
    def test_duplicate_keys(self): self.exchange('duplicate_key')
    def test_invalid_metadata_rejected_before_body(self): self.exchange('invalid_before_payload')
    def test_timeout(self): self.exchange('timeout')


if __name__=='__main__': unittest.main()
