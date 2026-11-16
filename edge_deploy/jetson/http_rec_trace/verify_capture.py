"""Independent Windows audit, retaining both failed orchestration records verbatim."""
import hashlib
import json
from pathlib import Path,PurePosixPath
import tarfile
from analysis import analyze


def main():
    folder=Path('temp/http_rec_trace_e168_r2'); p=folder/'technical_evidence_r1.tgz'
    digest=p.with_suffix('.sha256').read_text().split()[0]; assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
    with tarfile.open(p) as tar:
        members=tar.getmembers(); assert len({m.name for m in members})==len(members)
        for m in members:
            n=PurePosixPath(m.name)
            assert m.isfile() and not n.is_absolute() and '..' not in n.parts and ':' not in m.name and '\\' not in m.name
        data={m.name:tar.extractfile(m).read() for m in members}
    manifest=json.loads(data['manifest.json']); assert set(data)==set(manifest)|{'manifest.json'}
    for n,v in manifest.items(): assert len(data[n])==v['bytes'] and hashlib.sha256(data[n]).hexdigest()==v['sha256'],n
    state=json.loads(data['control.json']); post=json.loads(data['postflight.json']); report=json.loads(data['trace/profile.json'])
    assert state['status']=='sample_failed_e167_restored' and state['restored'] and state['error']=='AssertionError()'
    assert state['profile_returncode']==1 and report['status']=='failed'
    assert report['error']=="TimeoutError('Diagnostic startup/gate deadline; retain evidence')"
    assert report['closed'] and report['profiler_exit']==0 and report['http_exit']==report['cpp_exit']==0
    assert report['backend_exit']['status']=='stopped' and report['backend_exit']['phase']=='stopped'
    assert report['backend_exit']['close_error'] is False and json.loads(data['trace/gate.json'])['phase']=='stopped'
    assert b"gate('stop','stopped')" in data['profile_wrapper.log'] and 'trace/factory_error.json' not in data
    assert post['status']=='capture_reconciled_e167_four_ready_audited' and post['master_pid']==state['restored_master_pid']
    assert json.loads(data['reconciliation.json'])==post['reconciliation']
    assert post['reconciliation']['status']=='complete_capture_validated_after_stop_observer_timeout'
    assert state['extract_requests']==post['extract_requests']==report['extract_requests']==0 and post['originals_unchanged']
    golden=json.loads(data['reference_hashes.json']); assert len(golden)==32; total=0
    for cycle,pages in ((state['ordinary_control'],32),(report['warmup'],32),(report['measured'],32),(state['restored_check'],4)):
        requests=[cycle] if pages==4 else cycle['requests']; rows=[r for req in requests for r in req['records']]
        assert len(rows)==pages
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
        assert hashlib.sha256(local.read_bytes()).hexdigest()==v,n
    for name in ('finalize.py','verify_capture.py'):
        assert data[name]==(Path(__file__).parent/name).read_bytes()
    raw=data['trace/gpu.csv'].decode()
    assert not any(s in raw.lower() for s in ('dropped','err_nvgpuctrperm','application received signal','== error:'))
    result=analyze(raw,json.loads(data['trace/mapping.json']))
    assert json.loads(json.dumps(result))==json.loads(data['trace/analysis.json'])==post['analysis']
    witness=result['peak_witness']; peak=result['max_distinct_rec_owners_over_threshold']
    assert len({r['owner'] for r in witness})==peak
    assert min(r['end_us'] for r in witness)-max(r['start_us'] for r in witness)>.1
    failed=json.loads(data['failed_r1/control.json'])
    assert failed['status']=='sample_failed_e167_restored' and failed['restored'] and failed['restored_master_pid']==state['old_master_pid']
    h=post['health']; assert h['status']=='ready' and not h['busy'] and h['models']['rec']['threaded']['alive']==4 and h['models']['ner']['calls']==1
    out=folder/'verification_final.json'; assert not out.exists()
    value=dict(status='complete_raw_capture_sources_quality_failed_records_and_restore_independently_verified',sha256=digest,
        bytes=p.stat().st_size,members=len(data),production_source_files=len(post['source_sha256'])-1,
        verified_r2_pages=total,restored_master_pid=post['master_pid'],analysis=result,
        note='Initial factory failure and Stop observer timeout remain failed; full32 captured pages, stopped gate, native exit0 and safe restoration validated, no GPU rerun. 16 independent images; not normal-run throughput/GT/24h.')
    out.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k!='analysis'})); print(json.dumps(result))


if __name__=='__main__': main()
