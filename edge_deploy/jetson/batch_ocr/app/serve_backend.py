"""One persistent Python model worker group behind the C++ application.

Private Unix sockets only. Does not create HTTP or touch activation/Extract.
"""
import argparse
import contextlib
import json
import logging
import os
from pathlib import Path
import signal
import sys
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'ipc'))
import ipc_wire as wire
from channel_io import ChannelServer
from ocr_worker_adapter import OCRRuntime
from stage_session import require
from worker_bridge import WorkerBridge

BASE=Path('/home/xchq/work/autoocr'); RELEASE=BASE/'releases/jetson-r2'; WIDE=BASE/'overlays/rec-wide-e124-r1'


def status(path, data):
    temp=path.with_suffix('.new')
    with temp.open('w',encoding='utf-8') as f: json.dump(data,f,allow_nan=False)
    os.replace(temp,path)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True); args=p.parse_args()
    batch_enabled=os.environ.get('AUTOOCR_BATCH_DET','0')=='1'
    if os.environ.get('AUTOOCR_BATCH_DET','0') not in ('0','1'): raise ValueError('Expected batch flag 0/1')
    rec_slots=os.environ.get('AUTOOCR_REC_SLOTS','0')
    if rec_slots not in ('0','1','2','4'): raise ValueError('Expected rec slots 0/1/2/4')
    rec_slots=int(rec_slots)
    rec_threads=os.environ.get('AUTOOCR_REC_THREADS','0')
    if rec_threads not in ('0','1') or (rec_threads=='1' and not rec_slots):
        raise ValueError('REC threads require flag 0/1 and nonzero rec slots')
    runtime_type=OCRRuntime; bridge_type=WorkerBridge; channel_type=ChannelServer
    if rec_slots:
        sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'multistream'))
        from runtime import AsyncRuntime
        from bridge import MultiBridge
        from stage2 import MultiChannel
        runtime_type=AsyncRuntime; bridge_type=MultiBridge; channel_type=MultiChannel
        if rec_threads=='1':
            from threaded_rec import runtime_class
            runtime_type=runtime_class()
    root=args.runtime.resolve(); require(root.is_relative_to(BASE) and root.is_dir())
    require(root.stat().st_mode & 0o077==0 and root.stat().st_uid==os.geteuid())
    require(not (root/'status.json').exists())
    mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    require(int(mem['MemAvailable'].split()[0])>=8*1024**2)
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(RELEASE),AUTOOCR_BASE_DIR=str(RELEASE),
                      AUTOOCR_RUNTIME_DIR=str(root/'business'),AUTOOCR_TEMP_DIR=str(root/'business/temp'),
                      AUTOOCR_LOG_FILE=os.devnull,AUTOOCR_REC_WIDE_CONFIG=str(WIDE/'config.json'),
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_AUTOINSTALL='false')
    sys.path.insert(0,str(WIDE))
    from offline_business import load_business
    from rec_wide_adapter import close_models
    quit=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:quit.set())
    marker=root/'status.json'; runtime=[]; logging.disable(logging.CRITICAL)
    status(marker,dict(status='starting',pid=os.getpid(),auth_mode='jetson_dev_no_license',scope='batch_ocr_no_extract'))
    def factory():
        def batch_factory():
            from trt82_backend import Session as TRTSession
            return TRTSession(BASE/'staging/det_batch_e153_r1/det_batch2.engine',
                '692064bd77441c918bd2fce877e90e898ce365b0ebe071022bc48b35100f3e3e','det_batch2')
        gpu=runtime_type(load_business,close_models,root,det_batch_factory=batch_factory if batch_enabled else None,public_response=True,
                         **({'rec_slots':rec_slots} if rec_slots else {}))
        runtime.append(gpu); gpu.server.log=lambda *a,**k:None
        loaded=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(loaded['MemAvailable'].split()[0])<2*1024**2:
            gpu.synchronize(); gpu.close(); raise RuntimeError('loaded reserve')
        return gpu
    with open(os.devnull,'w') as quiet,contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet),wire.private_listener(root/'backend.sock') as listener:
        channel=channel_type(listener,allow_det_batch=batch_enabled,transport_bytes=64*1024**2,
                             **({'rec_slots':rec_slots} if rec_slots else {}))
        def algorithm(payload,kind,api): return runtime[0].algorithm(payload,kind,api)
        bridge=bridge_type(channel,algorithm,factory); next_status=0
        try:
            while not quit.is_set() and not channel.stopped:
                channel.poll(.02); bridge.step()
                now=time.monotonic()
                if now>=next_status:
                    state='error' if bridge.init_error or channel.session.gpu_failed else ('ready' if bridge.initialized else 'starting')
                    status(marker,dict(status=state,pid=os.getpid(),monotonic=now,auth_mode='jetson_dev_no_license',
                        scope='batch_ocr_no_extract',session=channel.session.stats(),worker=bridge.stats(),io=channel.stats(),
                        models=runtime[0].stats() if bridge.initialized and runtime else {},
                        recent_pages=runtime[0].metrics() if bridge.initialized and runtime else {}))
                    next_status=now+1
        finally:
            status(marker,dict(status='stopping',pid=os.getpid(),scope='batch_ocr_no_extract'))
            # Unknown GPU safety is retained, never force-freed or reset after a timeout.
            while not bridge.close(1):
                status(marker,dict(status='quarantined',pid=os.getpid(),scope='batch_ocr_no_extract',worker=bridge.stats()))
                if bridge.closed: break
            channel.close()
            status(marker,dict(status='stopped',pid=os.getpid(),scope='batch_ocr_no_extract',worker=bridge.stats()))


if __name__=='__main__': main()
