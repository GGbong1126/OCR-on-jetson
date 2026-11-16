"""Repeat CPU/IPC and JPEG checks with every existing native test entry enabled."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile

import controller as c


def main():
    out=c.S/'supplement_r1.json'; assert not out.exists()
    accepted=json.loads((c.S/'acceptance_r1.json').read_text())
    assert accepted['status']=='selected_http_boundary_and_restart_passed'
    pid=accepted['new_master_pid']; before=c.idle()
    assert os.readlink(Path('/proc')/str(pid)/'cwd')==str(c.NEW/'app')
    build=c.B/'staging/multistream_e164_r1/cmake_r16/ipc'
    env=dict(os.environ,AUTOOCR_THREAD_SOURCE=str(c.NEW),
             AUTOOCR_BATCH_APP=str(c.NEW/'app/autoocr_batch_app'),
             AUTOOCR_IPC_PROBE=str(build/'ipc_probe'),
             AUTOOCR_STAGE_PROBE=str(build/'stage_client_probe'),AUTOOCR_TEST_JPEG='1',
             LD_LIBRARY_PATH=str(c.B/'lib/cuda10.2')+':/usr/local/cuda-10.2/targets/aarch64-linux/lib')
    native={k:dict(path=env[k],sha256=c.sha(Path(env[k])))
            for k in ('AUTOOCR_BATCH_APP','AUTOOCR_IPC_PROBE','AUTOOCR_STAGE_PROBE')}
    log=c.S/'supplement_tests_r1.log'
    commands=[[str(c.PY),'-B','-m','unittest','discover','-s',str(c.NEW/f),'-p','test_*.py','-v']
              for f in ('ipc','multistream','app')]
    commands.append([str(c.PY),'-B',str(c.S/'check_pipeline.py')])
    with log.open('xb') as stream:
        for command in commands:
            result=subprocess.run(command,env=env,stdout=stream,stderr=stream,timeout=90)
            assert result.returncode==0,command
    text=log.read_text(); counts=[int(n) for n in re.findall(r'^Ran (\d+) tests? in ',text,re.M)]
    assert counts==[113,40,29,3] and 'skipped' not in text and 'FAILED' not in text
    after=c.idle()
    assert before['models']['rec']['calls']==after['models']['rec']['calls']
    assert before['models']['ner']['calls']==after['models']['ner']['calls']==1
    assert os.readlink(Path('/proc')/str(pid)/'cwd')==str(c.NEW/'app')
    report=dict(status='all_185_checks_passed_zero_skips',counts=counts,native=native,
                service_master_pid=pid,rec_calls_unchanged=after['models']['rec']['calls'],
                extract_requests=0,gpu_model_calls=0,jpeg_hardware_test_enabled=True,
                note='Supplement to immutable R1: initial final_unit_tests.log skipped 17+1+1 native checks; no timed arm or production source changed.')
    out.write_text(json.dumps(report,indent=2)+'\n')
    files=[out,log,Path(__file__)]
    manifest={p.name:dict(sha256=c.sha(p),bytes=p.stat().st_size) for p in files}
    mp=c.S/'supplement_manifest_r1.json'; assert not mp.exists()
    mp.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=c.S/'supplement_evidence_r1.tgz'; assert not archive.exists()
    with tarfile.open(archive,'w:gz') as tar:
        for p in files+[mp]: tar.add(p,arcname=p.name,recursive=False)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.sha256').write_text(digest+'  '+archive.name+'\n')
    print(json.dumps(dict(status=report['status'],sha256=digest,bytes=archive.stat().st_size)),flush=True)


if __name__=='__main__': main()
