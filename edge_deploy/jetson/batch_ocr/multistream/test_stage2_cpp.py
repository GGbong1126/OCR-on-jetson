import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'ipc'))
import ipc_wire as wire
from stage_session import encoded
from stage2 import route, validate
from test_cpp_stage import metadata_cases


def cases():
    rows=[]
    for _,raw,n,valid in metadata_cases():
        if not valid: continue
        m=json.loads(raw); m['schema']='ocr.stage/2'
        if m['op']=='session.ready': m.update(rec_slots=4,rec_queue_items=8,rec_queue_bytes=16*1024**2)
        if m['op']=='model.ready' and m['model']=='rec':
            m.update(group_id=1,crop_id=0,box_index=0,route=route(m['inputs'][0]['shape'][3]))
        rows.append((encoded(m),n,True))
        for key in m:
            bad=copy.deepcopy(m); del bad[key]; rows.append((encoded(bad),n,False))
        rows.append((encoded(dict(m,extra=True)),n,False))
    root=dict(schema='ocr.stage/2',session='a'*32,remaining_ms=5000)
    for slot in (0,1,2,3,4,5,6,True):
        rows.append((encoded(dict(root,op='model.run',call_ids=[1],slot=slot)),0,type(slot) is int and 1<=slot<=5))
        rows.append((encoded(dict(root,op='rec.consumed',call_id=1,slot=slot)),0,type(slot) is int and 1<=slot<=5))
    return rows


class CorpusTests(unittest.TestCase):
    def test_python_corpus(self):
        for raw,n,expected in cases():
            try: validate(raw,n); accepted=True
            except (wire.ProtocolError,KeyError,IndexError,TypeError): accepted=False
            self.assertEqual(accepted,expected,raw)
    @unittest.skipUnless(sys.platform=='linux' and os.environ.get('AUTOOCR_STAGE_PROBE'),'Compiled candidate schema consumer required')
    def test_cpp_same_corpus(self):
        for raw,n,expected in cases():
            p=subprocess.run([os.environ['AUTOOCR_STAGE_PROBE'],'--metadata',str(n)],input=raw,capture_output=True,timeout=5)
            self.assertEqual(p.returncode,0 if expected else 1,raw)


if __name__=='__main__': unittest.main()
