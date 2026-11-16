"""Read-only final audit and privacy-safe technical evidence archive."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
from urllib.request import build_opener,ProxyHandler

from analyze import analyze
from controller import B,S,STABLE,SOURCE,units,sha,applications


def main():
    control=json.loads((S/'control_threads_r1.json').read_text())
    assert control['restored'] and control['phase']=='complete'
    assert all(r['equal'] for r in control['restored_checks']) and len(control['restored_checks'])==4
    with build_opener(ProxyHandler({})).open('http://127.0.0.1:5052/healthz',timeout=10) as response: h=json.load(response)
    assert h['status']=='ready' and not h['busy'] and h['runtime']['session']['images']==h['runtime']['session']['registered_bytes']==0
    assert h['models']['ner']['calls']==1
    initial=json.loads((S/'control_r1.json').read_text())['preflight']
    assert initial==dict(boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),units=units())
    master=Path('/proc')/str(control['restored_master_pid'])
    assert os.readlink(master/'cwd')==str(STABLE/'app') and b'gunicorn' in (master/'cmdline').read_bytes()
    env=(master/'environ').read_bytes().split(b'\0')
    assert b'AUTOOCR_BATCH_DET=1' in env and b'AUTOOCR_BATCH_JPEG=1' in env and not any(x.startswith(b'AUTOOCR_REC_SLOTS=') for x in env)
    apps=applications(); assert len(apps)==2 and all(str(STABLE) in a['command'] for a in apps)
    cpp=next(a for a in apps if a['command'].split()[0].endswith('/autoocr_batch_app'))
    binary_sha=sha(Path('/proc')/str(cpp['pid'])/'exe')
    assert binary_sha=='d123191b57fd861a00c1a47cbf7d533465f983eef7ccf9ae56d0d358ace8abf0'
    frozen=json.loads((B/'staging/multistream_e163_r1/preflight_r2.json').read_text())
    for root,key in ((B/'releases/jetson-r2','originals'),(B/'overlays/rec-wide-e124-r1','wide'),(STABLE,'stable_source_sha256')):
        assert all(sha(root/n)==v for n,v in frozen[key].items())
    assert sha(B/'staging/det_batch_e153_r1/det_batch2.engine')==frozen['n2_sha256']
    production=json.loads((SOURCE/'installation.json').read_text())['files_sha256']
    assert all(sha(SOURCE/n)==v for n,v in production.items())
    for control_name,folder in (('control_r1.json',S/'frozen_r1'),('control_threads_r1.json',S)):
        expected=json.loads((S/control_name).read_text())['scripts_sha256']
        assert all(sha(folder/name)==value for name,value in expected.items())
    summaries=[]
    for folder in (S/'timeline_r1',S/'timeline_threads_r1',S/'timeline_positive_r2'):
        if not folder.exists(): continue
        for path in sorted(folder.glob('*.json')):
            r=json.loads(path.read_text())
            if not r.get('trace',{}).get('stopped') or not r.get('closed'): continue
            if r['status'] not in ('rec_outputs_consistent','threaded_rec_outputs_consistent','positive_control_completed_NOT_REC'): continue
            a=analyze(path.with_suffix('.csv').read_text(),r['trace'])
            summaries.append(dict(file=str(path.relative_to(S)),mode=r.get('kind',r.get('mode')),width=r.get('width'),**a))
    assert any(r['max_rec_streams_over_threshold']>=4 and r['mode']!='positive_control_NOT_REC' for r in summaries)
    perf=[]
    for path in sorted(S.glob('perf_*.json')):
        r=json.loads(path.read_text()); assert r['closed'] and r['safe_quiescence_confirmed']
        assert r['status'] in ('rec_outputs_consistent','threaded_rec_outputs_consistent')
        groups={}
        for p in r['passes']:
            count=p['slots']; group=groups.setdefault(count,dict(calls=0,seconds=0.,directions=[]))
            if 'rounds' in p:
                group['calls']+=count*len(p['rounds']); group['seconds']+=sum(v['wall_ms']/1000 for v in p['rounds'])
            else:
                group['calls']+=count*p['rounds_per_worker']; group['seconds']+=p['seconds']
            group['directions'].append(p['direction'])
        for group in groups.values(): group['inferences_per_second']=group['calls']/group['seconds']
        perf.append(dict(file=path.name,mode=r.get('kind',r.get('mode')),width=r['width'],allocated_slots=r['allocated_slots'],groups=groups))
    report=dict(status='finite_rec_experiments_audited_e162_restored',restored_master_pid=control['restored_master_pid'],
                restored_cpp_pid=cpp['pid'],restored_cpp_sha256=binary_sha,health=h,preflight_unchanged=True,
                source_sha256=production,originals_unchanged=True,script_versions_verified=True,
                timelines=summaries,performance=perf,
                note='Graph profiler crash preserved; positive control is not REC; finite synthetic experiments, no universal hardware upper bound or HTTP speed claim.')
    out=S/'postflight_r1.json'; assert not out.exists(); out.write_text(json.dumps(report,indent=2)+'\n')
    files=list(S.glob('*.py'))+list(S.glob('*.md'))+list(S.glob('*.json'))+list(S.glob('*.log'))
    files+=list((S/'frozen_r1').glob('*.py'))
    for folder in (S/'timeline_r1',S/'timeline_threads_r1',S/'timeline_positive_r2'):
        if not folder.exists(): continue
        files += [p for p in folder.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.log')]
    files=sorted(set(files)); assert all(p.resolve().is_relative_to(S) and not p.is_symlink() for p in files)
    manifest={str(p.relative_to(S)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    mp=S/'manifest_r1.json'; assert not mp.exists(); mp.write_text(json.dumps(manifest,indent=2)+'\n')
    target=S/'technical_evidence_r1.tgz'; assert not target.exists()
    with tarfile.open(target,'w:gz') as archive:
        for p in files+[mp]: archive.add(p,arcname=str(p.relative_to(S)),recursive=False)
    digest=sha(target); target.with_suffix('.sha256').write_text(digest+'  '+target.name+'\n')
    print(json.dumps(dict(status=report['status'],sha256=digest,bytes=target.stat().st_size,members=len(files)+1,
                         max_observed_rec=max(r['max_rec_streams_over_threshold'] for r in summaries if r['mode']!='positive_control_NOT_REC'))),flush=True)


if __name__=='__main__': main()
