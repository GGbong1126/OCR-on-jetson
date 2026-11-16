"""Independent local validation of the complete downloaded E165 archive."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

from analyze import analyze


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    archive=a.directory/'technical_evidence_r1.tgz'
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest==(a.directory/'technical_evidence_r1.sha256').read_text().split()[0]
    with tarfile.open(archive,'r:gz') as tar, tempfile.TemporaryDirectory(dir=a.directory,prefix='verify_') as folder:
        members=tar.getmembers(); assert len({m.name for m in members})==len(members)
        assert all(m.isfile() and not Path(m.name).is_absolute() and not Path(m.name).drive and ':' not in m.name and '..' not in Path(m.name).parts and '\\' not in m.name for m in members)
        manifest=json.load(tar.extractfile('manifest.json'))['files']
        assert set(manifest)=={m.name for m in members if m.name!='manifest.json'}
        for name,row in manifest.items():
            raw=tar.extractfile(name).read()
            assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
            if name.startswith('reports/'):
                # Generated technical verification artifacts, never executable extraction.
                path=Path(folder)/name[8:]; assert path.resolve().is_relative_to(Path(folder).resolve())
                path.write_bytes(raw)
        result=analyze(Path(folder))
        assert result==json.loads((Path(folder)/'analysis_r1.json').read_text())
        control=json.loads((Path(folder)/'control_r3.json').read_text())
        reference=json.load(tar.extractfile('e164_reference/bench_s0_r18.json'))
        current=json.loads((Path(folder)/'00_sync.json').read_text())
        assert {(r['image_sha256'],r['image_type']):r['candidate_sha256'] for r in reference['rows']} == {(r['image_sha256'],r['image_type']):r['candidate_sha256'] for r in current['rows']}
        for name in ('experiment.py','controller.py','check_pipeline.py'):
            expected=control['scripts_sha256'][name]
            assert manifest['reports/'+name]['sha256']==expected
            assert hashlib.sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()==expected
        post=json.loads((Path(folder)/'postflight.json').read_text())
        assert post['status']=='originals_units_models_unchanged_e162_ready' and post['restored_master_pid']==result['restored_master_pid']
        root=Path(__file__).resolve().parents[1]/'batch_ocr'
        for name,row in manifest.items():
            if name.startswith('source/'):
                assert hashlib.sha256((root/name[7:]).read_bytes()).hexdigest()==row['sha256']
        target=a.directory/'verification_final.json'
        with target.open('x') as f:
            json.dump(dict(status='sha_members_sources_eight_arms_resources_and_e162_restoration_verified',
                           archive_sha256=digest,members=len(members),analysis=result),f,indent=2,allow_nan=False)
        print(json.dumps({k:v for k,v in result.items() if k!='arms'},indent=2))


if __name__=='__main__': main()
