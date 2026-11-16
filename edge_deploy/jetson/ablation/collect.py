"""Archive technical-only E165 reports/source after verified E162 restoration."""
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from urllib.request import build_opener,ProxyHandler

from analyze import analyze
from controller import owners,units

B=Path('/home/xchq/work/autoocr'); S=B/'staging/ablation_e165_r1'


def main():
    result=analyze(S)
    analysis=S/'analysis_r1.json'
    if not analysis.exists():
        with analysis.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    else:
        assert json.loads(analysis.read_text())==result
    control=json.loads((S/'control_r3.json').read_text())
    frozen=json.loads((B/'staging/multistream_e163_r1/preflight_r2.json').read_text())
    assert units()==control['preflight']['units']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==control['preflight']['boot']
    stable=B/'overlays/batch-ocr-e161-r1'
    stable_binary=hashlib.sha256((stable/'app/autoocr_batch_app').read_bytes()).hexdigest()
    assert stable_binary=='d123191b57fd861a00c1a47cbf7d533465f983eef7ccf9ae56d0d358ace8abf0'
    for parent,key in ((stable,'stable_source_sha256'),(B/'releases/jetson-r2','originals'),(B/'overlays/rec-wide-e124-r1','wide')):
        assert all(hashlib.sha256((parent/n).read_bytes()).hexdigest()==v for n,v in frozen[key].items())
    assert hashlib.sha256((B/'staging/det_batch_e153_r1/det_batch2.engine').read_bytes()).hexdigest()==frozen['n2_sha256']
    pid=control['restored_master_pid'];proc=Path('/proc')/str(pid)
    assert os.readlink(proc/'cwd')==str(stable/'app')
    raw=(proc/'environ').read_bytes().split(b'\0')
    assert b'AUTOOCR_BATCH_DET=1' in raw and b'AUTOOCR_BATCH_JPEG=1' in raw and not any(v.startswith(b'AUTOOCR_REC_SLOTS=') for v in raw)
    assert len(owners())==2 and all(str(stable) in line for line in owners())
    with build_opener(ProxyHandler({})).open('http://127.0.0.1:5052/healthz',timeout=5) as response: health=json.load(response)
    assert health['status']=='ready' and not health['busy'] and health['runtime']['session']['images']==health['runtime']['session']['registered_bytes']==0
    assert health['models']['ner']['calls']==1 and all(v['errors']==0 for v in health['models'].values())
    memory={k:int(v.split()[0]) for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k in ('MemAvailable','NvMapMemUsed','SwapFree')}
    assert memory['MemAvailable']>=2*1024**2
    with (S/'postflight.json').open('x') as f:
        json.dump(dict(status='originals_units_models_unchanged_e162_ready',restored_master_pid=pid,stable_binary_sha256=stable_binary,memory_kib=memory,health=health,units=units(),model_owners=owners()),f,indent=2)
    source=B/'overlays/batch-ocr-plan2-e164-r1'
    files={}
    for p in S.iterdir():
        if p.is_file() and p.suffix in ('.json','.py','.log'):
            files['reports/'+p.name]=p
    for name,expected in control['source_files'].items():
        p=source/name; assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
        if name=='app/autoocr_batch_app': continue
        files['source/'+name]=p
    for name in ('bench_s0_r18.json','measurement_r18.json','postflight_r2.json'):
        files['e164_reference/'+name]=B/'staging/multistream_e164_r1'/name
    manifest={n:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in sorted(files.items())}
    archive=S/'technical_evidence_r1.tgz'; assert not archive.exists()
    with tarfile.open(archive,'w:gz') as tar:
        raw=(json.dumps(dict(kind='technical_only_no_images_bodies_weights_credentials',files=manifest),indent=2)+'\n').encode()
        entry=tarfile.TarInfo('manifest.json'); entry.size=len(raw); entry.mode=0o644
        tar.addfile(entry,io.BytesIO(raw))
        for name,p in sorted(files.items()): tar.add(p,arcname=name,recursive=False)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    with (S/'technical_evidence_r1.sha256').open('x') as f: f.write(digest+'  technical_evidence_r1.tgz\n')
    print(json.dumps(dict(sha256=digest,members=len(files)+1,bytes=archive.stat().st_size,restored_master_pid=result['restored_master_pid'])))


if __name__=='__main__': main()
