"""Validate a complete capture after the exact idle-Stop observer timeout; retain failures."""
import json
import os
from pathlib import Path
import tarfile
import time
from settings import S,SOURCE,e167 as c
from analysis import analyze


def main():
    end=time.monotonic()+300
    while time.monotonic()<end:
        state=json.loads((S/'control.json').read_text())
        if state['phase']=='complete': break
        time.sleep(1)
    else: raise RuntimeError('Original controller has not finished restoration')
    assert state['status']=='sample_failed_e167_restored' and state['restored'] and state['error']=='AssertionError()'
    assert state['profile_returncode']==1
    report=json.loads((S/'trace/profile.json').read_text())
    assert report['status']=='failed' and report['error']=="TimeoutError('Diagnostic startup/gate deadline; retain evidence')"
    assert report['closed'] and report['profiler_exit']==0 and report['http_exit']==report['cpp_exit']==0
    assert report['backend_exit']['status']=='stopped' and report['backend_exit']['phase']=='stopped'
    assert report['backend_exit']['close_error'] is False
    assert json.loads((S/'trace/gate.json').read_text())['phase']=='stopped'
    assert "gate('stop','stopped')" in (S/'profile_wrapper.log').read_text()
    assert not (S/'trace/factory_error.json').exists()
    assert all(not c.alive(report[n]) for n in ('backend_pid','http_pid','cpp_pid'))
    golden=json.loads((c.S/'reference_hashes.json').read_text()); total=0
    for cycle,pages in ((state['ordinary_control'],32),(report['warmup'],32),(report['measured'],32),(state['restored_check'],4)):
        requests=[cycle] if pages==4 else cycle['requests']
        rows=[r for request in requests for r in request['records']]; assert len(rows)==pages
        for r in rows: assert r['equal'] and r['public_sha256']==golden[r['image_sha256']+':'+str(r['image_type'])]
        total+=pages
    pid=state['restored_master_pid']; health=c.idle()
    assert os.readlink(Path('/proc')/str(pid)/'cwd')==str(SOURCE/'app') and not c.alive(state['old_master_pid'])
    assert state['initial']==dict(boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),units=c.units())
    for n,v in state['source_sha256'].items(): assert c.sha(SOURCE/n)==v,n
    for n,v in state['scripts_sha256'].items(): assert c.sha(S/n)==v,n
    t=health['models']['rec']['threaded']; assert t['owners']==t['alive']==4 and t['errors']==0 and t['context_shared']
    assert health['models']['ner']['calls']==1
    cpp=[p for p in c.tree(pid) if (Path('/proc')/str(p)/'cmdline').read_bytes().split(b'\0')[0]==str(SOURCE/'app/autoocr_batch_app').encode()]
    assert len(cpp)==1 and c.sha(Path('/proc')/str(cpp[0])/'exe')==state['source_sha256']['app/autoocr_batch_app']
    raw=(S/'trace/gpu.csv').read_text()
    assert not any(s in raw.lower() for s in ('dropped','err_nvgpuctrperm','application received signal','== error:'))
    result=analyze(raw,json.loads((S/'trace/mapping.json').read_text())); del raw
    ap=S/'trace/analysis.json'; assert not ap.exists(); ap.write_text(json.dumps(result,indent=2)+'\n')
    reconciliation=dict(status='complete_capture_validated_after_stop_observer_timeout',original_control_status=state['status'],
         original_profile_status=report['status'],original_wrapper_returncode=1,profiler_returncode=0,
         stop_confirmed=True,backend_safely_closed=True,verified_pages=total,
         note='Only the idle ProfilerStop acknowledgement exceeded 10s. All32 profiled HTTP pages, later stopped gate, native exit0 and safe closure verified; original failed reports unchanged. No GPU rerun.')
    rp=S/'reconciliation.json'; assert not rp.exists(); rp.write_text(json.dumps(reconciliation,indent=2)+'\n')
    post=dict(status='capture_reconciled_e167_four_ready_audited',master_pid=pid,cpp_pid=cpp[0],health=health,
         source_sha256=state['source_sha256'],originals_unchanged=state['originals_unchanged'],extract_requests=0,
         reconciliation=reconciliation,analysis=result)
    pp=S/'postflight.json'; assert not pp.exists(); pp.write_text(json.dumps(post,indent=2)+'\n')
    files={str(p.relative_to(S)):p for p in S.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json','.log')}
    files.update({str(p.relative_to(S)):p for p in (S/'trace').iterdir() if p.is_file() and p.suffix in ('.json','.csv','.log')})
    files['reference_hashes.json']=c.S/'reference_hashes.json'
    failed=S.parent/'http_rec_trace_e168_r1'
    for name in ('control.json','controller.log','profile_wrapper.log','tests.log','backend_trace.py','profile.py','run.py','settings.py',
                 'trace/profile.json','trace/backend_exit.json','trace/gpu.csv','trace/runtime/status.json'):
        files['failed_r1/'+name]=failed/name
    for n in state['source_sha256']:
        if n!='app/autoocr_batch_app': files['deployed_source/'+n]=SOURCE/n
    for n in ('selection.json','installation.json'): files['deployed_source/'+n]=SOURCE/n
    manifest={n:dict(bytes=p.stat().st_size,sha256=c.sha(p)) for n,p in files.items()}
    mp=S/'manifest.json'; assert not mp.exists(); mp.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=S/'technical_evidence_r1.tgz'; assert not archive.exists()
    with tarfile.open(archive,'w:gz') as tar:
        for n,p in sorted(files.items()): tar.add(p,arcname=n,recursive=False)
        tar.add(mp,arcname='manifest.json',recursive=False)
    digest=c.sha(archive); archive.with_suffix('.sha256').write_text(digest+'  '+archive.name+'\n')
    print(json.dumps(dict(status=post['status'],sha256=digest,bytes=archive.stat().st_size,members=len(files)+1,
        master_pid=pid,peak=result['max_distinct_rec_owners_over_threshold'],at_least_us=result['at_least_us'],
        at_least_percent=result['at_least_percent_of_rec_active_union'])),flush=True)


if __name__=='__main__': main()
