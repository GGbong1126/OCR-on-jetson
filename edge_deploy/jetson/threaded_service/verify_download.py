"""Independent Windows check of installed sources, HTTP data hashes and selection."""
import hashlib
import json
from pathlib import Path,PurePosixPath
import tarfile


def main():
    folder=Path('temp/threaded_e167_r1'); p=folder/'technical_evidence_r1.tgz'
    digest=(folder/'technical_evidence_r1.sha256').read_text().split()[0]
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
    with tarfile.open(p,'r:gz') as archive:
        members=archive.getmembers(); names=[m.name for m in members]
        assert len(names)==len(set(names))
        for m in members:
            assert m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts
            assert ':' not in m.name and '\\' not in m.name
        data={m.name:archive.extractfile(m).read() for m in members}
    manifest=json.loads(data['manifest_r1.json'])
    assert set(data)==set(manifest)|{'manifest_r1.json'}|{n for n in data if n.startswith('deployed_source/')}
    for name,v in manifest.items(): assert len(data[name])==v['bytes'] and hashlib.sha256(data[name]).hexdigest()==v['sha256']
    state=json.loads(data['control_r1.json']); post=json.loads(data['postflight_r1.json']); accept=json.loads(data['acceptance_r1.json'])
    assert state['status']=='selected_threaded_ocr_deployed' and accept['status']=='selected_http_boundary_and_restart_passed'
    assert accept['extract_requests']==post['extract_requests']==0 and post['originals_unchanged']
    for n,v in state['scripts_sha256'].items():
        path='frozen_controller_r2/controller.py' if n=='controller.py' else n
        assert hashlib.sha256(data[path]).hexdigest()==v
    for n,v in state['resumed_scripts_sha256'].items(): assert hashlib.sha256(data[n]).hexdigest()==v
    patch=json.loads(data['owner_activation_r2.json']); assert patch['phase']=='http_arm_0' and not patch['arms']
    for n,v in post['source_sha256'].items():
        if n=='app/autoocr_batch_app':
            assert v=='f4baeef9a403324f67e49e2cb05c219d51302f7aa28ae60a4aec71de8c228b1c'; continue
        assert hashlib.sha256(data['deployed_source/'+n]).hexdigest()==v
        local=Path('edge_deploy/jetson/threaded_service/start_threaded.sh') if n=='start_threaded.sh' else Path('edge_deploy/jetson/batch_ocr')/n
        assert local.is_file() and hashlib.sha256(local.read_bytes()).hexdigest()==v,n
    for n,item in patch['files'].items():
        assert hashlib.sha256(data['frozen_owner_r1/'+n]).hexdigest()==item['before_sha256']
        assert post['source_sha256'][n]==item['after_sha256']
    golden=json.loads(data['reference_hashes.json']); assert len(golden)==32
    inventory=json.loads(data['inventory_r1.json']); assert inventory['header_valid']==673 and len(inventory['selected'])==16
    arms=[]
    for i in range(5):
        arm=json.loads(data['arm_%d.json'%i]); assert all(arm['gates'].values()) and len(arm['passes'])==3
        for cycle in [arm['warmup']]+arm['passes']:
            assert cycle['pages']==32 and len(cycle['requests'])==8
            for request in cycle['requests']:
                assert request['pages']==4 and request['seconds']>0 and request['prepare_seconds']>=0
                for r in request['records']: assert r['equal'] and r['public_sha256']==golden[r['image_sha256']+':'+str(r['image_type'])]
        seconds=sum(p['seconds'] for p in arm['passes']); assert 96/seconds==arm['pages_per_second']
        arms.append(dict(index=i,slots=arm['slots'],seconds=seconds,pages_per_second=96/seconds))
    fastest=max(a['pages_per_second'] for a in arms if a['slots'])
    selected=min(a['slots'] for a in arms if a['slots'] and a['pages_per_second']>=fastest*.98)
    assert selected==state['selected']==post['selected']==accept['selected']
    deployed=json.loads(data['deployed_source/selection.json']); assert deployed['selected_rec_threads']==selected
    base=(arms[0]['seconds']+arms[4]['seconds'])/2; chosen=next(a for a in arms if a['slots']==selected)
    gain=(base/chosen['seconds']-1)*100
    assert gain==post['throughput_gain_percent']
    for r in accept['restart_checks']['records']+accept['concurrent_valid_request']['records']:
        assert r['equal'] and r['public_sha256']==golden[r['image_sha256']+':'+str(r['image_type'])]
    assert len(accept['invalid'])==3 and all(r['http_status']==400 for r in accept['invalid']) and accept['busy']['http_status']==429
    h=post['health']; assert h['status']=='ready' and not h['busy'] and h['models']['ner']['calls']==1
    assert h['runtime']['session']['images']==h['runtime']['session']['registered_bytes']==0
    out=folder/'verification_final.json'; assert not out.exists()
    result=dict(status='members_installed_sources_http_quality_selection_restart_and_originals_verified',sha256=digest,
                members=len(members),selected=selected,arms=arms,baseline_seconds_mean=base,throughput_gain_percent=gain,
                final_master_pid=post['final_master_pid'],production_source_files=len(post['source_sha256'])-1,
                note='Finite 16-image loopback experiment, not all673/GT/24h or new kernel concurrency proof.')
    out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
