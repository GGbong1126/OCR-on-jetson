"""Authorized normal-stop, isolated ablations, then restore unchanged E162."""
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import build_opener, ProxyHandler, Request

B = Path('/home/xchq/work/autoocr'); S = B/'staging/ablation_e165_r1'
OLD = B/'staging/multistream_e164_r1'; SOURCE = B/'overlays/batch-ocr-plan2-e164-r1'
STABLE = B/'overlays/batch-ocr-e161-r1'; PYTHON = B/'envs/business-py310/bin/python'
ORDER = ('sync', 'full', 'one_inflight', 'blocking_submit', 'no_fanout', 'inline_ctc', 'full', 'sync')
OUT = S/'control_r3.json'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def alive(pid):
    try: return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError: return False
def units():
    return {u: subprocess.check_output(['systemctl', 'show', u, '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID', '-p', 'UnitFileState'], text=True).splitlines()
            for u in ('autoocr-jetson-dev.service', 'gdm.service', 'ssh.service', 'frpc-jetson.service')}
def owners():
    return [x.strip() for x in subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True).splitlines()
            if any(v in x for v in ('/serve_backend.py', '/autoocr_batch_app ', '/verify_ocr_workers.py', str(S/'experiment.py')))]


def main():
    assert not OUT.exists()
    state = dict(status='running',phase='preflight',order=ORDER,arms=[],started=time.monotonic(),restored=False)
    def save(): OUT.write_text(json.dumps(state, indent=2, allow_nan=False))
    stop_requested = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.__setitem__(0, True))
    opener = build_opener(ProxyHandler({}))
    def health():
        with opener.open('http://127.0.0.1:5052/healthz', timeout=5) as r: return json.load(r)
    frozen = json.loads((B/'staging/multistream_e163_r1/preflight_r2.json').read_text())
    install = json.loads((SOURCE/'installation.json').read_text())
    source_sha = install['files_sha256']
    binary = OLD/'cmake_r16/app/autoocr_batch_app'
    original = dict(boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),units=units())
    assert original['boot'] == frozen['boot'] and original['units'] == frozen['units']
    def fingerprints():
        assert all(sha(B/'releases/jetson-r2'/n) == v for n, v in frozen['originals'].items())
        assert all(sha(B/'overlays/rec-wide-e124-r1'/n) == v for n, v in frozen['wide'].items())
        assert sha(B/'staging/det_batch_e153_r1/det_batch2.engine') == frozen['n2_sha256']
        assert all(sha(STABLE/n) == v for n, v in frozen['stable_source_sha256'].items())
        assert all(sha(SOURCE/n) == v for n, v in source_sha.items())
        assert sha(binary) == 'f4baeef9a403324f67e49e2cb05c219d51302f7aa28ae60a4aec71de8c228b1c'
    fingerprints()
    h = health(); assert h['status'] == 'ready' and not h['busy'] and h['runtime']['session']['images'] == 0
    masters = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal(): continue
        try:
            if os.readlink(proc/'cwd') == str(STABLE/'app') and b'gunicorn' in (proc/'cmdline').read_bytes():
                masters.append(int(proc.name))
        except (FileNotFoundError, PermissionError): pass
    parents = dict(tuple(map(int, line.split())) for line in subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True).splitlines())
    roots = [p for p in masters if parents[p] not in masters]; assert len(roots) == 1
    pid = roots[0]
    envraw = (Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
    assert b'AUTOOCR_BATCH_DET=1' in envraw and b'AUTOOCR_BATCH_JPEG=1' in envraw
    assert not any(v.startswith(b'AUTOOCR_REC_SLOTS=') for v in envraw)
    owned = {pid}
    while True:
        new = owned | {p for p, pp in parents.items() if pp in owned}
        if new == owned: break
        owned = new
    assert len(owned) == 4 and len(owners()) == 2 and all(str(STABLE) in x for x in owners())
    state.update(preflight=original,old_master_pid=pid,owned_pids=sorted(owned),health_before=h,
                 source_files=source_sha,binary_sha256=sha(binary),
                 scripts_sha256={p.name:sha(p) for p in S.glob('*.py')})
    save()
    env = dict(os.environ, LD_LIBRARY_PATH=str(B/'lib/cuda10.2')+':/usr/local/cuda-10.2/targets/aarch64-linux/lib',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONNOUSERSITE='1',
               AUTOOCR_BATCH_APP=str(binary), AUTOOCR_ABLATION_SOURCE=str(SOURCE))
    env.pop('AUTOOCR_CUPTI_LIBRARY', None)
    # No models: real C++ channel/retirement tests before any service downtime.
    for mode in ('full', 'one_inflight', 'blocking_submit', 'no_fanout', 'inline_ctc'):
        with (S/('pipeline_r3_'+mode+'.log')).open('xb') as log:
            result = subprocess.run([str(PYTHON), '-B', str(S/'check_pipeline.py'), '--source', str(SOURCE), '--mode', mode],
                                    env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL, timeout=90)
        assert result.returncode == 0, ('fake pipeline failure', mode)
    state['phase'] = 'normal_stop_e162'; save(); start = time.monotonic()
    # Recheck immediately before the stop; the old PID is not trusted by itself.
    assert health()['busy'] is False and os.readlink(Path('/proc')/str(pid)/'cwd') == str(STABLE/'app')
    os.kill(pid, signal.SIGTERM)
    while any(alive(p) for p in owned) and time.monotonic()-start < 60: time.sleep(.1)
    assert not any(alive(p) for p in owned), 'Stop safety unknown; no new models'
    assert not owners()
    state['normal_stop_seconds'] = time.monotonic()-start
    state['phase'] = 'isolated_measurements'; save()
    safe = True; error = None
    try:
        golden = json.loads((OLD/'bench_s0_r18.json').read_text())
        golden = {(r['image_sha256'], r['image_type']):r['candidate_sha256'] for r in golden['rows']}
        for index, mode in enumerate(ORDER):
            if stop_requested[0]: raise RuntimeError('Requested graceful experiment abort')
            assert not owners()
            output = S/('%02d_%s.json' % (index, mode)); assert not output.exists()
            with output.with_suffix('.log').open('xb') as log:
                child = subprocess.Popen([str(PYTHON), '-u', '-B', str(S/'experiment.py'), '--source', str(SOURCE),
                                          '--mode', mode, '--output', str(output)], env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL)
                arm = dict(index=index,mode=mode,pid=child.pid,output=str(output),status='running')
                state['arms'].append(arm); save(); safe = False
                code = child.wait()
            r = json.loads(output.read_text()) if output.exists() else {}
            safe = bool(r.get('closed')) and not owners()
            arm.update(returncode=code,status=r.get('status','no_report'),closed=r.get('closed'),resource_gate=r.get('resource_gate',{})); save()
            assert safe and code == 0 and r['status'] == 'full_python_ocr_workers_consistent_not_http_or_gt'
            assert all(r['resource_gate'].values()) and len(r['passes']) == 4
            assert all(p['full_data_equal_count'] == 26 for p in r['passes'])
            assert all(v['reference_rec_widths'] == v['candidate_rec_widths'] for v in r['rows'])
            assert {(v['image_sha256'],v['image_type']):v['candidate_sha256'] for v in r['rows']} == golden
            host = json.loads(output.with_suffix('.host.json').read_text())
            assert host['completed'] and host['snapshot_count'] == 2
            measured = [p for p in r['passes'] if not p['warmup']]
            arm.update(seconds=sum(p['seconds'] for p in measured),pages=78,
                       pages_per_second=78/sum(p['seconds'] for p in measured),pass_seconds=[p['seconds'] for p in measured])
            save(); print('Finished', mode, arm['pages_per_second'], flush=True)
    except Exception as failure:
        error = type(failure).__name__; state['error'] = error
    finally:
        state['measurements_completed'] = error is None and len(state['arms']) == len(ORDER)
        state['phase'] = 'restore_e162' if safe else 'unsafe_candidate_exit_manual_review_required'; save()
        if safe:
            assert not owners(); fingerprints()
            assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == original['boot'] and units() == original['units']
            for key in ('AUTOOCR_REC_SLOTS','AUTOOCR_BATCH_DET','AUTOOCR_BATCH_JPEG','AUTOOCR_BATCH_RUNTIME_BASE'):
                env.pop(key, None)
            with (S/'restored_e162.log').open('xb') as log:
                master = subprocess.Popen(['bash',str(STABLE/'app/start_batch.sh')],env=env,cwd=STABLE/'app',
                                          stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
            state['restored_master_pid'] = master.pid; save(); end = time.monotonic()+250
            while time.monotonic() < end and master.poll() is None:
                try:
                    h = health()
                    if h['status'] == 'ready' and not h['busy']: break
                except (OSError, ValueError): pass
                time.sleep(.2)
            else: raise RuntimeError('E162 restore failed; no force kill')
            raw = (Path('/proc')/str(master.pid)/'environ').read_bytes().split(b'\0')
            assert b'AUTOOCR_BATCH_DET=1' in raw and b'AUTOOCR_BATCH_JPEG=1' in raw and not any(v.startswith(b'AUTOOCR_REC_SLOTS=') for v in raw)
            plan = json.loads((B/'staging/det_batch_e153_r1/plan.json').read_text()); images = []; keys = []
            for row in plan['rows'][:2]:
                data = (Path(plan['images'])/row['file']).read_bytes(); assert hashlib.sha256(data).hexdigest() == row['id']
                for kind in (3,1):
                    images.append(dict(imageId='image'+str(len(images)),imageType=kind,imageBase64=base64.b64encode(data).decode()))
                    keys.append((row['id'],kind))
            req = Request('http://127.0.0.1:5052/OCRBatch',data=json.dumps(dict(images=images)).encode(),headers={'Content-Type':'application/json'})
            with opener.open(req,timeout=320) as r: value = json.load(r)
            assert value['status'] == 'ok' and len(value['results']) == 4
            checks = []
            for item, key in zip(value['results'],keys):
                digest = hashlib.sha256(json.dumps(item['data'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
                assert item['status'] == 'ok' and digest == golden[key]
                checks.append(dict(image_sha256=key[0],image_type=key[1],equal=True,public_sha256=digest))
            for _ in range(60):
                h = health()
                if not h['busy'] and h['runtime']['session']['images'] == h['runtime']['session']['registered_bytes'] == 0: break
                time.sleep(.1)
            assert h['status'] == 'ready' and not h['busy'] and h['runtime']['session']['images'] == h['runtime']['session']['registered_bytes'] == 0
            fingerprints(); assert units() == original['units']
            state.update(restored=True,restored_checks=checks,restored_health=h,phase='complete',finished=time.monotonic(),
                         status='ablation_complete_e162_restored' if state['measurements_completed'] else 'experiment_failed_e162_restored')
            save()
    assert state['measurements_completed'] and state['restored']


if __name__ == '__main__': main()
