"""Audit all HTTP arms and installed source, then archive technical metadata only."""
import json
import os
from pathlib import Path
import tarfile

import controller as c


def main():
    out=c.S/'postflight_r1.json'; assert not out.exists()
    state=json.loads(c.OUT.read_text()); acceptance=json.loads((c.S/'acceptance_r1.json').read_text())
    assert state['status']=='selected_threaded_ocr_deployed' and acceptance['status']=='selected_http_boundary_and_restart_passed'
    assert acceptance['extract_requests']==0
    assert state['selected']==acceptance['selected']
    h=c.idle(); t=h['models']['rec']['threaded']; assert t['owners']==t['alive']==state['selected'] and t['context_shared'] and t['errors']==0
    pid=acceptance['new_master_pid']; assert os.readlink(Path('/proc')/str(pid)/'cwd')==str(c.NEW/'app')
    cpp=[]
    for child in c.tree(pid):
        command=(Path('/proc')/str(child)/'cmdline').read_bytes().split(b'\0')
        if command[0]==str(c.NEW/'app/autoocr_batch_app').encode(): cpp.append(child)
    assert len(cpp)==1 and c.sha(Path('/proc')/str(cpp[0])/'exe')=='f4baeef9a403324f67e49e2cb05c219d51302f7aa28ae60a4aec71de8c228b1c'
    fixes=json.loads((c.S/'safety_tests_r1.json').read_text()); assert all(v for k,v in fixes.items() if k!='gpu_calls') and fixes['gpu_calls']==0
    initial=state['preflight']; assert initial==dict(boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),units=c.units())
    frozen=json.loads((c.B/'staging/multistream_e163_r1/preflight_r2.json').read_text())
    for root,key in ((c.B/'releases/jetson-r2','originals'),(c.B/'overlays/rec-wide-e124-r1','wide'),(c.OLD,'stable_source_sha256')):
        assert all(c.sha(root/n)==v for n,v in frozen[key].items())
    base=json.loads((c.PREVIOUS/'installation.json').read_text())['files_sha256']
    assert all(c.sha(c.PREVIOUS/n)==v for n,v in base.items())
    patch=json.loads((c.S/'owner_activation_r2.json').read_text()); assert patch['phase']=='http_arm_0' and not patch['arms']
    expected=dict(state['source_sha256'])
    for name,item in patch['files'].items():
        assert expected[name]==item['before_sha256'] and c.sha(c.S/'frozen_owner_r1'/name)==item['before_sha256']
        expected[name]=item['after_sha256']
    assert all(c.sha(c.NEW/n)==v for n,v in expected.items())
    assert all(c.sha(c.S/('frozen_controller_r2/controller.py' if n=='controller.py' else n))==v for n,v in state['scripts_sha256'].items())
    assert all(c.sha(c.S/n)==v for n,v in state['resumed_scripts_sha256'].items())
    golden=json.loads((c.S/'reference_hashes.json').read_text()); arms=[]
    for index in range(5):
        arm=json.loads((c.S/('arm_%d.json'%index)).read_text()); assert all(arm['gates'].values())
        assert len(arm['passes'])==3 and arm['warmup']['pages']==32
        for p in [arm['warmup']]+arm['passes']:
            assert p['pages']==32 and len(p['requests'])==8
            assert sum(r['pages'] for r in p['requests'])==32
            for request in p['requests']:
                assert len(request['records'])==request['pages']
                assert all(r['equal'] and r['public_sha256']==golden[r['image_sha256']+':'+str(r['image_type'])] for r in request['records'])
        seconds=sum(p['seconds'] for p in arm['passes']); assert arm['pages_per_second']==96/seconds
        arms.append(dict(index=index,slots=arm['slots'],pages=96,seconds=seconds,pages_per_second=96/seconds))
    candidates=[r for r in arms if r['slots']]; fastest=max(r['pages_per_second'] for r in candidates)
    selected=min(r['slots'] for r in candidates if r['pages_per_second']>=.98*fastest); assert selected==state['selected']
    baseline_seconds=(arms[0]['seconds']+arms[4]['seconds'])/2
    chosen=next(r for r in candidates if r['slots']==selected)
    report=dict(status='threaded_http_deployment_audited',arms=arms,selected=selected,
                baseline_seconds_mean=baseline_seconds,throughput_gain_percent=(baseline_seconds/chosen['seconds']-1)*100,
                baseline_drift_seconds=arms[4]['seconds']-arms[0]['seconds'],
                final_master_pid=acceptance['new_master_pid'],final_cpp_pid=cpp[0],health=h,originals_unchanged=True,
                source_sha256=expected,script_versions_verified=True,extract_requests=0,
                note='16 selected real JPEG images, ordinary/enhanced; finite loopback HTTP experiment, no GT/24h/locked clocks or new kernel timeline.')
    out.write_text(json.dumps(report,indent=2)+'\n')
    files=list(c.S.glob('*.py'))+list(c.S.glob('*.md'))+list(c.S.glob('*.json'))+list(c.S.glob('*.log'))
    files+=list((c.S/'frozen_r1').glob('*.py'))+list((c.S/'frozen_controller_r2').glob('*.py'))+list((c.S/'frozen_owner_r1').rglob('*.py'))
    manifest={str(p.relative_to(c.S)):dict(sha256=c.sha(p),bytes=p.stat().st_size) for p in sorted(set(files))}
    mp=c.S/'manifest_r1.json'; assert not mp.exists(); mp.write_text(json.dumps(manifest,indent=2)+'\n')
    target=c.S/'technical_evidence_r1.tgz'; assert not target.exists()
    # Explicit source namespace includes the actual new service, not image/weight/runtime files.
    with tarfile.open(target,'w:gz') as archive:
        for p in sorted(set(files)|{mp}): archive.add(p,arcname=str(p.relative_to(c.S)),recursive=False)
        for name in sorted(expected):
            if name=='app/autoocr_batch_app': continue
            archive.add(c.NEW/name,arcname='deployed_source/'+name,recursive=False)
        for name in ('selection.json','installation.json','start_threaded.sh'):
            if name not in expected: archive.add(c.NEW/name,arcname='deployed_source/'+name,recursive=False)
    digest=c.sha(target); target.with_suffix('.sha256').write_text(digest+'  '+target.name+'\n')
    print(json.dumps(dict(status=report['status'],sha256=digest,bytes=target.stat().st_size,selected=selected,
                         gain=report['throughput_gain_percent'],master=report['final_master_pid'])),flush=True)


if __name__=='__main__': main()
