"""Post-restoration audit and explicit technical-only evidence archive."""
import json
import os
from pathlib import Path
import tarfile
from settings import S,SOURCE,e167 as c


def main():
    state=json.loads((S/'control.json').read_text())
    assert state['status']=='actual_http_trace_and_e167_restore_complete' and state['restored']
    pid=state['restored_master_pid']; health=c.idle()
    assert os.readlink(Path('/proc')/str(pid)/'cwd')==str(SOURCE/'app')
    assert not c.alive(state['old_master_pid'])
    assert state['initial']==dict(boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),units=c.units())
    for n,v in state['source_sha256'].items(): assert c.sha(SOURCE/n)==v,n
    for n,v in state['scripts_sha256'].items(): assert c.sha(S/n)==v,n
    report=json.loads((S/'trace/profile.json').read_text())
    assert report['closed'] and report['status']=='real_http_trace_closed' and report['profiler_exit']==0
    assert not c.alive(report['backend_pid'])
    assert not any(c.alive(report[n]) for n in ('http_pid','cpp_pid'))
    assert health['models']['rec']['threaded']['alive']==4 and health['models']['rec']['threaded']['errors']==0
    assert health['models']['ner']['calls']==1 and health['runtime']['session']['images']==health['runtime']['session']['registered_bytes']==0
    cpp=[]
    for child in c.tree(pid):
        command=(Path('/proc')/str(child)/'cmdline').read_bytes().split(b'\0')
        if command[0]==str(SOURCE/'app/autoocr_batch_app').encode(): cpp.append(child)
    assert len(cpp)==1 and c.sha(Path('/proc')/str(cpp[0])/'exe')==state['source_sha256']['app/autoocr_batch_app']
    out=S/'postflight.json'; assert not out.exists()
    out.write_text(json.dumps(dict(status='sample_closed_e167_four_ready_audited',master_pid=pid,cpp_pid=cpp[0],health=health,
        source_sha256=state['source_sha256'],originals_unchanged=state['originals_unchanged'],extract_requests=0,
        note='Only the temporary diagnostic model processes ran as root; current E167 restored as xchq, no system/autostart changes.'),indent=2)+'\n')
    files={str(p.relative_to(S)):p for p in S.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json','.log')}
    files.update({str(p.relative_to(S)):p for p in (S/'trace').iterdir() if p.is_file() and p.suffix in ('.json','.csv','.log')})
    files['reference_hashes.json']=c.S/'reference_hashes.json'
    for n in state['source_sha256']:
        if n!='app/autoocr_batch_app': files['deployed_source/'+n]=SOURCE/n
    for name in ('selection.json','installation.json'): files['deployed_source/'+name]=SOURCE/name
    manifest={n:dict(bytes=p.stat().st_size,sha256=c.sha(p)) for n,p in files.items()}
    mp=S/'manifest.json'; assert not mp.exists(); mp.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=S/'technical_evidence_r1.tgz'; assert not archive.exists()
    with tarfile.open(archive,'w:gz') as t:
        for n,p in sorted(files.items()): t.add(p,arcname=n,recursive=False)
        t.add(mp,arcname='manifest.json',recursive=False)
    digest=c.sha(archive); archive.with_suffix('.sha256').write_text(digest+'  '+archive.name+'\n')
    print(json.dumps(dict(status='audited_and_archived',members=len(files)+1,bytes=archive.stat().st_size,sha256=digest,master_pid=pid)),flush=True)


if __name__=='__main__': main()
