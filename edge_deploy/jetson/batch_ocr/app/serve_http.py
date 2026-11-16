"""Independent loopback batch OCR front. One backend/model set and C++ app.

No original /OCR or /Extract route is replaced; no license/counter files touched.
"""
import atexit
import base64
import binascii
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from batch_front_client import exchange
from stage_contract import integer, token, LIMITS, validate_result

BASE=Path('/home/xchq/work/autoocr')
HERE=Path(__file__).resolve().parent
MAX_HTTP=24*1024**2
FRONT_RESERVATION=256*1024**2


def read_upload(*, discard=False):
    remaining=request.content_length
    if remaining is None or not 0<=remaining<=MAX_HTTP: raise ValueError('Bounded Content-Length required')
    sock=request.environ.get('gunicorn.socket'); timeout=sock.gettimeout() if sock is not None else None
    end=time.monotonic()+5; pieces=[]
    try:
        while remaining:
            left=end-time.monotonic()
            if left<=0: raise TimeoutError('Upload deadline')
            if sock is not None: sock.settimeout(left)
            try: data=request.stream.read(min(65536,remaining))
            except (OSError,BadRequest) as error:
                if isinstance(error,TimeoutError) or time.monotonic()>=end: raise TimeoutError('Upload deadline') from error
                raise
            if not data: raise ValueError('Truncated upload')
            remaining-=len(data)
            if not discard: pieces.append(data)
        return b'' if discard else b''.join(pieces)
    finally:
        if sock is not None: sock.settimeout(timeout)


class Supervisor:
    def __init__(self):
        if sys.platform!='linux' or os.environ.get('AUTOOCR_ENABLE_DEV_NO_AUTH')!='1': raise RuntimeError('Explicit Jetson development mode required')
        root=Path(os.environ.get('AUTOOCR_BATCH_RUNTIME_BASE',str(BASE/'runtime/batch-ocr'))).resolve()
        if not root.is_relative_to(BASE): raise ValueError('Project runtime root required')
        root.mkdir(parents=True,exist_ok=True)
        self.root=Path(tempfile.mkdtemp(prefix='run_',dir=root)); self.runtime_base=root
        self.backend=self.cpp=None; self.logs=[]; self.last_error=''
        rec_slots=os.environ.get('AUTOOCR_REC_SLOTS','0')
        if rec_slots not in ('0','1','2','4'): raise ValueError('Expected rec slots 0/1/2/4')
        env=dict(os.environ,LD_LIBRARY_PATH=str(BASE/'lib/cuda10.2')+':/usr/local/cuda-10.2/targets/aarch64-linux/lib',
                 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
        try:
            log=(self.root/'backend.log').open('xb'); self.logs.append(log)
            self.backend=subprocess.Popen([sys.executable,'-B',str(HERE/'serve_backend.py'),'--runtime',str(self.root)],
                env=env,cwd=HERE,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            end=time.monotonic()+240
            while time.monotonic()<end:
                if self.backend.poll() is not None: raise RuntimeError('Backend startup failed')
                state=self.snapshot()
                if state.get('status')=='ready': break
                if state.get('status') in ('error','quarantined','stopped'): raise RuntimeError('Backend not healthy')
                time.sleep(.05)
            else: raise TimeoutError('Backend startup deadline')
            log=(self.root/'cpp.log').open('xb'); self.logs.append(log)
            options=[]
            if rec_slots!='0': options.append('--rec-slots='+rec_slots)
            for env,flag in (('AUTOOCR_BATCH_DET','--det-batch'),('AUTOOCR_BATCH_JPEG','--jpeg')):
                value=os.environ.get(env,'0')
                if value not in ('0','1'): raise ValueError('Expected feature flag 0/1')
                if value=='1': options.append(flag)
            self.cpp=subprocess.Popen([str(HERE/'autoocr_batch_app'),str(self.root/'backend.sock'),str(self.root/'front.sock'),*options],
                cwd=HERE,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            end=time.monotonic()+5
            while not (self.root/'front.sock').exists() and self.cpp.poll() is None and time.monotonic()<end: time.sleep(.01)
            if not (self.root/'front.sock').exists(): raise RuntimeError('C++ startup failed')
            exchange(self.root/'front.sock')
        except Exception:
            self.close(); raise
        atexit.register(self.close)

    def snapshot(self):
        try:
            path=self.root/'status.json'
            if path.stat().st_size>64*1024: return {}
            return json.loads(path.read_text(encoding='utf-8'))
        except (FileNotFoundError,ValueError,OSError): return {}

    def health(self):
        state=self.snapshot()
        ready=bool(self.backend and self.cpp and self.backend.poll() is None and self.cpp.poll() is None and
                   state.get('status')=='ready' and state.get('pid')==self.backend.pid and time.monotonic()-state.get('monotonic',0)<10)
        return dict(status='ready' if ready else 'error',scope='batch_ocr_no_extract',auth_mode='jetson_dev_no_license',
                    backend=state.get('status','unknown'),models=state.get('models',{}),
                    recent_pages=state.get('recent_pages',{}),
                    runtime={k:state.get(k,{}) for k in ('session','worker','io')})

    def infer(self, images): return exchange(self.root/'front.sock',images,remaining_ms=300000)

    def close(self):
        safe=True; failed=False
        for child in (self.cpp,self.backend):
            if child is None or child.poll() is not None: continue
            child.terminate()
            try: child.wait(timeout=12)
            except subprocess.TimeoutExpired:
                safe=False; self.last_error='Process retained for safe GPU finalization'
        for child in (self.cpp,self.backend):
            if child is not None and child.poll() not in (None,0): failed=True
        if self.snapshot().get('worker',{}).get('close_error'): failed=True
        for log in self.logs: log.close()
        if safe and not failed and self.root.exists():
            if self.root.parent!=self.runtime_base or not self.root.name.startswith('run_'): raise RuntimeError('Unexpected cleanup target')
            shutil.rmtree(self.root)
        # Failed runs and unknown GPU owners retain their private evidence; never SIGKILL/reset.


def create_app(supervisor=None):
    service=supervisor or Supervisor(); app=Flask(__name__); app.config['MAX_CONTENT_LENGTH']=MAX_HTTP
    app.json.ensure_ascii=False
    gate=threading.BoundedSemaphore(1); busy=threading.Event()
    def error(code,status): return jsonify(status='error',code=code),status
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_): return error('LIMIT',413)
    @app.get('/healthz')
    def health():
        value=service.health(); value['busy']=busy.is_set()
        value.update(front_reservation_bytes=FRONT_RESERVATION if busy.is_set() else 0,
                     total_explicit_budget_bytes=LIMITS['explicit_budget_bytes']+FRONT_RESERVATION)
        return jsonify(value),200 if value['status']=='ready' else 503
    @app.post('/OCRBatch')
    def ocr_batch():
        if not gate.acquire(blocking=False):
            # Drain bounded uploads so an early 429 does not reset a still-sending TCP peer.
            remaining=request.content_length
            if remaining is not None and 0<=remaining<=MAX_HTTP:
                try: read_upload(discard=True)
                except (OSError,ValueError,BadRequest): pass
            return error('BUSY',429)
        busy.set()
        deferred=False
        try:
            if service.health()['status']!='ready': return error('BACKEND_UNAVAILABLE',503)
            if request.environ.get('CONTENT_LENGTH') in (None,'') or request.content_length is None: return error('LENGTH_REQUIRED',411)
            if request.content_length>MAX_HTTP: return error('LIMIT',413)
            if not request.is_json: return error('JSON_REQUIRED',415)
            try: payload=validate_result(read_upload(),max_bytes=MAX_HTTP)
            except (ValueError,UnicodeError,BadRequest): return error('INVALID_REQUEST',400)
            if type(payload) is not dict or set(payload)!={'images'}: return error('INVALID_REQUEST',400)
            rows=payload['images']
            if type(rows) is not list or not 1<=len(rows)<=4: return error('LIMIT',400)
            images=[]; seen=set(); total=0; base64_times={}
            for row in rows:
                if type(row) is not dict or set(row)!={'imageId','imageType','imageBase64'}: return error('INVALID_REQUEST',400)
                name=row['imageId']; kind=row['imageType']; value=row['imageBase64']
                if not token(name) or name in seen or not integer(kind,1,3) or type(value) is not str: return error('INVALID_REQUEST',400)
                if not 1<=len(value)<=((16*1024**2+2)//3)*4: return error('LIMIT',400)
                start=time.monotonic()
                try: data=base64.b64decode(value,validate=True)
                except (ValueError,binascii.Error): return error('INVALID_BASE64',400)
                base64_times[name]=(time.monotonic()-start)*1000
                if not 1<=len(data)<=16*1024**2: return error('LIMIT',400)
                seen.add(name); total+=len(data)
                if total>64*1024**2: return error('LIMIT',400)
                images.append((name,kind,data))
            result=service.infer(images)
            result['execution']['base64_ms']=base64_times
            good=all(row['status']=='ok' for row in result['results'])
            response=jsonify(status='ok' if good else 'partial',results=result['results'],execution=result['execution'])
            sock=request.environ.get('gunicorn.socket'); previous=sock.gettimeout() if sock is not None else None
            if sock is not None: sock.settimeout(5)
            close_lock=threading.Lock(); closed=False
            def delivered():
                nonlocal closed
                with close_lock:
                    if closed: return
                    closed=True
                    try:
                        if sock is not None: sock.settimeout(previous)
                    except OSError: pass
                    finally: busy.clear(); gate.release()
            response.call_on_close(delivered); deferred=True
            return response
        except RequestEntityTooLarge: return error('LIMIT',413)
        except TimeoutError: return error('UPLOAD_TIMEOUT',504)
        except Exception: return error('BACKEND_FAILED',503)
        finally:
            if not deferred: busy.clear(); gate.release()
    return app
