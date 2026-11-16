"""Selected real service: validation/BUSY and clean restart, no Extract requests."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request

import controller as c


def main():
    out=c.S/'acceptance_r1.json'; assert not out.exists()
    state=json.loads(c.OUT.read_text()); assert state['status']=='selected_threaded_ocr_deployed'
    selected=state['selected']; pid=state['selected_master_pid']; c.idle()
    report=dict(status='running',selected=selected,old_master_pid=pid,extract_requests=0)
    def save(): out.write_text(json.dumps(report,indent=2)+'\n')
    env=dict(os.environ,AUTOOCR_THREAD_SOURCE=str(c.NEW),AUTOOCR_BATCH_APP=str(c.NEW/'app/autoocr_batch_app'))
    with (c.S/'final_unit_tests.log').open('xb') as log:
        commands=[[str(c.PY),'-B','-m','unittest','discover','-s',str(c.NEW/folder),'-p','test_*.py','-v']
                  for folder in ('ipc','multistream','app')]
        commands.append([str(c.PY),'-B',str(c.S/'check_pipeline.py')])
        for cmd in commands:
            assert subprocess.run(cmd,env=env,stdout=log,stderr=log).returncode==0
    def invalid(payload,expected):
        req=Request('http://127.0.0.1:5052/OCRBatch',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        try:
            with c.OPENER.open(req,timeout=10) as response: code=response.status
        except HTTPError as error: code=error.code
        assert code==expected,(code,expected)
        return dict(http_status=code)
    report['invalid']=[invalid({},400),invalid(dict(images=[]),400),
                       invalid(dict(images=[dict(imageId='x',imageType=3,imageBase64='?')]),400)]
    rows=json.loads((c.S/'inventory_r1.json').read_text())['selected']
    golden=json.loads((c.S/'reference_hashes.json').read_text())
    result=[]; errors=[]
    def valid():
        try: result.append(c.request([(row,3) for row in rows[:4]],golden))
        except Exception as e: errors.append(repr(e))
    worker=threading.Thread(target=valid); worker.start()
    end=time.monotonic()+20
    while time.monotonic()<end:
        if c.health()['busy']: break
        time.sleep(.01)
    else: raise RuntimeError('Could not observe BUSY')
    report['busy']=invalid(dict(images=[]),429)
    worker.join(330); assert not worker.is_alive() and not errors and len(result)==1
    report['concurrent_valid_request']=result[0]; report['idle_after_busy']=c.idle(); save()
    c.stop(pid,c.NEW)
    # Exercise the delivered selection-file launcher, not just environment variables in the benchmark.
    env=dict(os.environ)
    for k in ('AUTOOCR_REC_SLOTS','AUTOOCR_REC_THREADS','AUTOOCR_BATCH_DET','AUTOOCR_BATCH_JPEG','AUTOOCR_BATCH_RUNTIME_BASE'): env.pop(k,None)
    with (c.S/'selected_restart.log').open('xb') as log:
        child=subprocess.Popen(['bash',str(c.NEW/'start_threaded.sh')],env=env,cwd=c.NEW/'app',stdin=subprocess.DEVNULL,
                               stdout=log,stderr=log,start_new_session=True)
    report['new_master_pid']=child.pid; save(); end=time.monotonic()+260
    while time.monotonic()<end and child.poll() is None:
        try:
            if c.health()['status']=='ready': break
        except (OSError,ValueError): pass
        time.sleep(.2)
    else: raise RuntimeError('Selected launcher restart failed; inspect retained runtime')
    report['restart_checks']=c.request([(rows[i//2],(3,1)[i%2]) for i in range(4)],golden)
    h=c.idle(); t=h['models']['rec']['threaded']; assert t['owners']==t['alive']==selected and t['errors']==0 and t['context_shared']
    raw=(Path('/proc')/str(child.pid)/'environ').read_bytes().split(b'\0')
    assert b'AUTOOCR_REC_THREADS=1' in raw and ('AUTOOCR_REC_SLOTS='+str(selected)).encode() in raw
    assert b'AUTOOCR_BATCH_DET=1' in raw and b'AUTOOCR_BATCH_JPEG=1' in raw
    report.update(status='selected_http_boundary_and_restart_passed',final_health=h,final_snapshot=c.snapshot(child.pid)); save()
    print(json.dumps(dict(status=report['status'],master=child.pid,selected=selected)),flush=True)


if __name__=='__main__': main()
