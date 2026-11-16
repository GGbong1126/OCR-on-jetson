"""Same E151 build contract via installed trtexec, observable and SSH-independent.

The launcher detaches this one job; no service, boot, GPU clock or BSP changes.
Completed CPU evidence is copied byte-for-byte and independently validated.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import det_batch2 as b

CLI=Path('/usr/src/tensorrt/bin/trtexec')
TIMEOUT_SECONDS=1800


def command(stage):
    shape=lambda s:'x:'+ 'x'.join(map(str,s))
    return [str(CLI),'--onnx='+str(b.SOURCE),
            '--minShapes='+shape(b.PROFILE[0]),'--optShapes='+shape(b.PROFILE[1]),
            '--maxShapes='+shape(b.PROFILE[2]),'--workspace=256','--avgTiming=8',
            '--noTF32','--profilingVerbosity=detailed','--buildOnly','--verbose',
            '--timingCacheFile='+str(stage/'timing.cache'),
            '--saveEngine='+str(stage/'det_batch2.engine')]


def stop_owned(process):
    if process.poll() is not None:return True
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:process.wait(timeout=15);return True
    except subprocess.TimeoutExpired:pass
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    try:process.wait(timeout=15);return True
    except subprocess.TimeoutExpired:return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args();stage=a.stage.resolve()
    if b.platform.machine()!='aarch64' or not stage.is_relative_to(b.q.BASE/'staging') or not stage.is_dir():
        p.error('Existing Jetson project stage required')
    for name in ('build.json','cli_build.log','det_batch2.engine','timing.cache','build_started.json'):
        if (stage/name).exists():raise FileExistsError('Preserve previous build artifact: '+name)
    plan=b.load_plan(stage);cpu=b.require(stage,'cpu_check.json','cpu_batch_independent')
    reuse=json.loads((stage/'reuse.json').read_text())
    if set(reuse['copied_sha256'])!={'plan.json','cpu_check.json','det_batch2.py','quantize_yolo_ptq.py'}:
        raise ValueError('Exact original plan/CPU/code evidence required')
    prior=Path(reuse['prior_stage']).resolve()
    if not prior.is_relative_to(b.q.BASE/'staging') or prior==stage:raise ValueError('Separate original evidence required')
    for name,expected in reuse['copied_sha256'].items():
        if Path(name).name!=name or b.q.digest(stage/name)!=expected or b.q.digest(prior/name)!=expected:
            raise ValueError('CPU evidence/source copy differs')
    if cpu['boxes_equal_count']!=13 or cpu['numeric_allclose_count']!=13 or not cpu['original_fingerprints_unchanged']:
        raise ValueError('Original CPU acceptance incomplete')
    if b.q.memory()['MemAvailable_kib']<plan['min_available_build_kib'] or b.shutil.disk_usage(stage).free<plan['min_free_disk_bytes']:
        raise RuntimeError('Resource reserve failed')
    state=subprocess.check_output(['systemctl','show','autoocr-jetson-dev.service','--property=ActiveState'],text=True)
    if state.strip()!='ActiveState=inactive':raise RuntimeError('Original service must stay stopped')
    library=ctypes.CDLL('libnvinfer.so.8');library.getInferLibVersion.restype=ctypes.c_int
    if library.getInferLibVersion()!=8201:raise RuntimeError('Expected original native TRT8.2.1')
    report=dict(status='building',plan_sha256=b.q.digest(stage/'plan.json'),
                reused_cpu_sha256=b.q.digest(stage/'cpu_check.json'),reuse_sha256=b.q.digest(stage/'reuse.json'),
                script_sha256=b.q.digest(__file__),core_sha256=b.q.digest(b.__file__),
                command=command(stage),cli_sha256=b.q.digest(CLI),native_trt_version=8201,
                pid=os.getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                max_build_seconds=TIMEOUT_SECONDS,memory_before=b.q.memory(),
                requested_precision=dict(fp16=False,int8=False,tf32=False),
                workspace_bytes=256*1024**2,avg_timing_iterations=8,
                settings_source='fixed CLI argv, not an internal builder-mask readback',
                note='Same ONNX/profile/FP32 policy; trtexec --buildOnly, no inference acceptance. '
                     'CPU evidence reused, not re-executed. Independent job; no driver/service/clock changes.')
    b.q.save(stage/'build_started.json',report)
    process=None;start=time.monotonic()
    try:
        with (stage/'cli_build.log').open('xb') as log:
            process=subprocess.Popen(report['command'],stdin=subprocess.DEVNULL,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            b.q.save(stage/'cli_process.json',dict(pid=process.pid,boot_id=report['boot_id'],
                       start_ticks=Path('/proc/{}/stat'.format(process.pid)).read_text().split()[21]))
            try:returncode=process.wait(timeout=TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                report.update(status='timed_out',child_stopped=stop_owned(process));returncode=process.returncode
            report['returncode']=returncode
        if report['status']=='timed_out':raise TimeoutError('Bounded build exceeded time limit')
        engine=stage/'det_batch2.engine'
        if returncode!=0 or not engine.is_file() or not engine.stat().st_size:
            raise RuntimeError('trtexec did not produce a successful engine')
        report.update(status='built_not_validated',engine_sha256=b.q.digest(engine),engine_bytes=engine.stat().st_size,
                      timing_cache_sha256=b.q.digest(stage/'timing.cache') if (stage/'timing.cache').is_file() else None)
    except BaseException as exc:
        if report['status']!='timed_out':report['status']='failed'
        report['error']=type(exc).__name__
        if process is not None and process.poll() is None and 'child_stopped' not in report:
            report['child_stopped']=stop_owned(process)
        raise
    finally:
        report.update(build_s=time.monotonic()-start,memory_after=b.q.memory())
        try:report['original_fingerprints_unchanged']=b.fingerprints()==plan['fingerprints']
        except Exception as exc:report.update(original_fingerprints_unchanged=False,post_audit_error=type(exc).__name__)
        if not report['original_fingerprints_unchanged']:report['status']='failed'
        b.q.save(stage/'build.json',report)
    print(json.dumps({k:report[k] for k in ('status','build_s','engine_bytes','engine_sha256')}),flush=True)


if __name__=='__main__':main()
