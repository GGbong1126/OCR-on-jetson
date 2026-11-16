"""Windows independent member/source/quality/raw-GPU overlap reconstruction."""
import hashlib
import json
from pathlib import Path,PurePosixPath
import sys
import tarfile

from analysis import analyze


def main():
    folder=Path('temp/http_rec_trace_e168_r2'); p=folder/'technical_evidence_r1.tgz'
    digest=p.with_suffix('.sha256').read_text().split()[0]
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
    with tarfile.open(p) as t:
        members=t.getmembers(); assert len({m.name for m in members})==len(members)
        for m in members:
            name=PurePosixPath(m.name)
            assert m.isfile() and not name.is_absolute() and '..' not in name.parts and ':' not in m.name and '\\' not in m.name
        data={m.name:t.extractfile(m).read() for m in members}
    manifest=json.loads(data['manifest.json']); assert set(data)==set(manifest)|{'manifest.json'}
    for n,v in manifest.items(): assert len(data[n])==v['bytes'] and hashlib.sha256(data[n]).hexdigest()==v['sha256'],n
    state=json.loads(data['control.json']); post=json.loads(data['postflight.json']); profile=json.loads(data['trace/profile.json'])
    assert state['status']=='actual_http_trace_and_e167_restore_complete' and state['restored']
    assert post['status']=='sample_closed_e167_four_ready_audited' and post['master_pid']==state['restored_master_pid']
    assert state['extract_requests']==post['extract_requests']==profile['extract_requests']==0
    assert post['originals_unchanged'] and profile['closed'] and profile['profiler_exit']==0 and profile['backend_exit']['phase']=='stopped'
    golden=json.loads(data['reference_hashes.json']); assert len(golden)==32
    total=0
    for cycle,pages in [(state['ordinary_control'],32),(profile['warmup'],32),(profile['measured'],32),(state['restored_check'],4)]:
        if pages==4: requests=[cycle]
        else: requests=cycle['requests']; assert cycle['pages']==32
        rows=[r for req in requests for r in req['records']]; assert len(rows)==pages
        if pages==32: assert {r['image_sha256']+':'+str(r['image_type']) for r in rows}==set(golden)
        for r in rows: assert r['equal'] and r['public_sha256']==golden[r['image_sha256']+':'+str(r['image_type'])]
        total+=pages
    for n,v in post['source_sha256'].items():
        if n=='app/autoocr_batch_app':
            assert v=='f4baeef9a403324f67e49e2cb05c219d51302f7aa28ae60a4aec71de8c228b1c'; continue
        assert hashlib.sha256(data['deployed_source/'+n]).hexdigest()==v
        local=Path('edge_deploy/jetson/threaded_service/start_threaded.sh') if n=='start_threaded.sh' else Path('edge_deploy/jetson/batch_ocr')/n
        assert hashlib.sha256(local.read_bytes()).hexdigest()==v,n
    for n,v in state['scripts_sha256'].items():
        assert hashlib.sha256(data[n]).hexdigest()==v
        local=Path('edge_deploy/jetson/rec_concurrency/analyze.py') if n=='analyze.py' else Path(__file__).parent/n
        assert hashlib.sha256(local.read_bytes()).hexdigest()==v
    result=analyze(data['trace/gpu.csv'].decode(),json.loads(data['trace/mapping.json']))
    assert json.loads(json.dumps(result))==json.loads(data['trace/analysis.json'])==state['analysis']
    # Independently intersect the raw-row witness, not merely its declared peak.
    witness=result['peak_witness']
    assert len({r['owner'] for r in witness})==result['max_distinct_rec_owners_over_threshold']
    assert min(r['end_us'] for r in witness)-max(r['start_us'] for r in witness)>.1
    health=post['health']; assert health['status']=='ready' and not health['busy'] and health['models']['rec']['threaded']['alive']==4
    assert health['models']['ner']['calls']==1
    out=folder/'verification_final.json'; assert not out.exists()
    value=dict(status='all_members_sources_100_http_pages_raw_rec_owner_intervals_and_restore_verified',sha256=digest,
        members=len(data),bytes=p.stat().st_size,production_source_files=len(post['source_sha256'])-1,
        verified_pages=total,restored_master_pid=post['master_pid'],analysis=result,
        note='16 independent images, profiler/root diagnostic window, not normal-run duty cycle/throughput/GT/24h.')
    out.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k!='analysis'})); print(json.dumps(result))


if __name__=='__main__': main()
