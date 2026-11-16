"""Ordinary-user controller: real HTTP control, authorized sample, E167 restoration."""
import json
import os
from pathlib import Path
import subprocess
import time

from settings import B,S,SOURCE,PY,e167 as c


def main():
    assert os.geteuid()==1000
    out=S/'control.json'; assert not out.exists()
    expected=json.loads((c.S/'postflight_r1.json').read_text())['source_sha256']
    frozen=json.loads((B/'staging/multistream_e163_r1/preflight_r2.json').read_text())
    def fingerprints():
        assert all(c.sha(SOURCE/n)==v for n,v in expected.items())
        for root,key in ((B/'releases/jetson-r2','originals'),(B/'overlays/rec-wide-e124-r1','wide'),
                         (B/'overlays/batch-ocr-e161-r1','stable_source_sha256')):
            assert all(c.sha(root/n)==v for n,v in frozen[key].items())
        assert c.sha(B/'staging/det_batch_e153_r1/det_batch2.engine')==frozen['n2_sha256']
    fingerprints(); h=c.idle(); assert h['models']['rec']['threaded']['owners']==4
    parents=dict(tuple(map(int,l.split())) for l in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines())
    matches=[]
    for pid in parents:
        try:
            p=Path('/proc')/str(pid)
            if os.readlink(p/'cwd')==str(SOURCE/'app') and b'gunicorn' in (p/'cmdline').read_bytes(): matches.append(pid)
        except (FileNotFoundError,PermissionError): pass
    roots=[p for p in matches if parents[p] not in matches]; assert len(roots)==1; master=roots[0]
    assert subprocess.run(['sudo','-n','true'],stdin=subprocess.DEVNULL).returncode==0
    initial=dict(boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),units=c.units())
    state=dict(status='running',phase='normal_http_control',initial=initial,old_master_pid=master,
         source_sha256=expected,scripts_sha256={p.name:c.sha(p) for p in S.glob('*.py')},extract_requests=0)
    def save(): out.write_text(json.dumps(state,indent=2)+'\n')
    save()
    with (S/'tests.log').open('xb') as log:
        assert subprocess.run([str(PY),'-B','-m','unittest','discover','-s',str(S),'-p','test_*.py','-v'],stdout=log,stderr=log).returncode==0
    rows=json.loads((c.S/'inventory_r1.json').read_text())['selected']; golden=json.loads((c.S/'reference_hashes.json').read_text())
    state['ordinary_control']=c.cycle(rows,golden); save()
    c.stop(master,SOURCE); state['phase']='profile_real_http'; save()
    safe=False; error=None
    try:
        with (S/'profile_wrapper.log').open('xb') as log:
            child=subprocess.Popen(['sudo','-n',str(PY),'-u','-B',str(S/'sample_service.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log)
            state['profile_wrapper_pid']=child.pid; save(); code=child.wait()
        report=json.loads((S/'trace/profile.json').read_text())
        safe=report['closed']; state['profile_returncode']=code; save()
        assert safe,'Unknown GPU cleanup; no new model set'
        assert code==0 and report['status']=='real_http_trace_closed'
    except Exception as exc: error=repr(exc); state['error']=error; save()
    if not safe:
        state.update(status='unsafe_or_unknown_retain',phase='no_new_models'); save()
        raise RuntimeError('Inspect retained sample before restoring')
    state['phase']='restore_e167_four'; save()
    env=dict(os.environ)
    for k in ('AUTOOCR_REC_SLOTS','AUTOOCR_REC_THREADS','AUTOOCR_BATCH_DET','AUTOOCR_BATCH_JPEG','AUTOOCR_BATCH_RUNTIME_BASE','PYTHONPATH'):
        env.pop(k,None)
    with (S/'restored_start.log').open('xb') as log:
        restored=subprocess.Popen(['bash',str(SOURCE/'start_threaded.sh')],env=env,cwd=SOURCE/'app',stdin=subprocess.DEVNULL,
             stdout=log,stderr=log,start_new_session=True)
    state['restored_master_pid']=restored.pid; save(); end=time.monotonic()+260
    while restored.poll() is None and time.monotonic()<end:
        try:
            if c.health()['status']=='ready': break
        except (OSError,ValueError): pass
        time.sleep(.2)
    else: raise RuntimeError('E167 restoration failed; inspect new runtime')
    state['restored_check']=c.request([(rows[i//2],(3,1)[i%2]) for i in range(4)],golden)
    state['health_restored']=c.idle(); t=state['health_restored']['models']['rec']['threaded']
    assert t['owners']==t['alive']==4 and t['errors']==0 and t['context_shared']
    state['snapshot_restored']=c.snapshot(restored.pid)
    raw=(Path('/proc')/str(restored.pid)/'environ').read_bytes().split(b'\0')
    assert all(k in raw for k in (b'AUTOOCR_REC_THREADS=1',b'AUTOOCR_REC_SLOTS=4',b'AUTOOCR_BATCH_DET=1',b'AUTOOCR_BATCH_JPEG=1'))
    fingerprints(); assert initial==dict(boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),units=c.units())
    state.update(restored=True,originals_unchanged=True,phase='analyze'); save()
    if error is None:
        with (S/'analysis.log').open('xb') as log:
            assert subprocess.run([str(PY),'-B',str(S/'analysis.py')],stdout=log,stderr=log).returncode==0
        state['analysis']=json.loads((S/'trace/analysis.json').read_text())
    state.update(status='actual_http_trace_and_e167_restore_complete' if error is None else 'sample_failed_e167_restored',phase='complete'); save()
    print(json.dumps(dict(status=state['status'],restored_master_pid=restored.pid,
         peak=state.get('analysis',{}).get('max_distinct_rec_owners_over_threshold'))),flush=True)


if __name__=='__main__': main()
