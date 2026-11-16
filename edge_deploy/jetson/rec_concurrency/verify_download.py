"""Independently verify downloaded evidence and recompute all kernel concurrency."""
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import tarfile

from analyze import analyze


def main():
    folder=Path('temp/rec_concurrency_e166_r1'); path=folder/'technical_evidence_r1.tgz'
    expected=(folder/'technical_evidence_r1.sha256').read_text().split()[0]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers(); names=[m.name for m in members]
        assert len(names)==len(set(names))
        for m in members:
            assert m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts
            assert ':' not in m.name and '\\' not in m.name
        data={m.name:archive.extractfile(m).read() for m in members}
    manifest=json.loads(data['manifest_r1.json'])
    assert set(data)==set(manifest)|{'manifest_r1.json'}
    for name,meta in manifest.items():
        assert len(data[name])==meta['bytes'] and hashlib.sha256(data[name]).hexdigest()==meta['sha256']
    for control_name,prefix in (('control_r1.json','frozen_r1/'),('control_threads_r1.json','')):
        control=json.loads(data[control_name])
        assert all(hashlib.sha256(data[prefix+name]).hexdigest()==sha for name,sha in control['scripts_sha256'].items())
    post=json.loads(data['postflight_r1.json']); control=json.loads(data['control_threads_r1.json'])
    assert control['restored'] and control['phase']=='complete' and len(control['restored_checks'])==4
    assert all(r['equal'] for r in control['restored_checks'])
    assert post['status']=='finite_rec_experiments_audited_e162_restored' and post['preflight_unchanged'] and post['script_versions_verified']
    for name,sha in post['source_sha256'].items():
        if name=='app/autoocr_batch_app':
            assert sha=='f4baeef9a403324f67e49e2cb05c219d51302f7aa28ae60a4aec71de8c228b1c'
            continue  # Board-only executable, separately pinned; not a Windows source file.
        p=Path('edge_deploy/jetson/batch_ocr')/name
        assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==sha,name
    timelines=[]
    for row in post['timelines']:
        quality=json.loads(data[row['file']]); csv_name=row['file'][:-5]+'.csv'
        value=analyze(data[csv_name].decode(),quality['trace'])
        canonical=json.loads(json.dumps(value))
        assert all(row[k]==v for k,v in canonical.items() if k not in ('duration_us_by_distinct_stream_count','at_least_us'))
        assert {int(k):v for k,v in row['duration_us_by_distinct_stream_count'].items()}==value['duration_us_by_distinct_stream_count']
        assert {int(k):v for k,v in row['at_least_us'].items()}==value['at_least_us']
        timelines.append(dict(file=row['file'],mode=row['mode'],width=row['width'],peak=value['max_rec_streams_over_threshold'],
                              at_least_us=value['at_least_us'],duration_us_by_count=value['duration_us_by_distinct_stream_count']))
    for row in post['performance']:
        r=json.loads(data[row['file']]); assert r['closed'] and r['safe_quiescence_confirmed']
        assert r['status'] in ('rec_outputs_consistent','threaded_rec_outputs_consistent')
        assert r['allocated_slots']==row['allocated_slots']
        for count,g in row['groups'].items():
            passes=[p for p in r['passes'] if p['slots']==int(count)]
            assert len(passes)==2 and {p['direction'] for p in passes}=={'forward','reverse'}
            calls=sum(int(count)*(len(p['rounds']) if 'rounds' in p else p['rounds_per_worker']) for p in passes)
            seconds=sum(sum(x['wall_ms']/1000 for x in p['rounds']) if 'rounds' in p else p['seconds'] for p in passes)
            assert calls==g['calls'] and seconds==g['seconds'] and calls/seconds==g['inferences_per_second']
    result=dict(status='archive_members_sources_script_versions_timelines_and_restoration_verified',sha256=expected,
                members=len(members),production_source_files=len(post['source_sha256'])-1,
                restored_master_pid=post['restored_master_pid'],timelines=timelines,performance=post['performance'])
    out=folder/'verification_final.json'; assert not out.exists(); out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],members=len(members),
                         max_rec=max(r['peak'] for r in timelines if r['mode']!='positive_control_NOT_REC'))))


if __name__=='__main__': main()
