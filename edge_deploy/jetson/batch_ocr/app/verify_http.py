"""Own three loopback epochs: default hardware/N2, software/N1 rollback, default restart.

Compare real HTTP data with the accepted public-format report. Never save OCR
bodies, call Extract inference, replace systemd units, or kill unknown GPU owners.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, ProxyHandler, build_opener

from smoke_batch import statistics

BASE=Path('/home/xchq/work/autoocr')


def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--golden',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--leave-running',action='store_true'); args=p.parse_args()
    output=args.output.resolve(); stage=output.parent; source=args.source.resolve()
    assert stage.is_relative_to(BASE/'staging') and source.is_relative_to(BASE) and not output.exists()
    golden=json.loads(args.golden.read_text())
    assert golden['public_response'] and golden['status']=='full_python_ocr_workers_consistent_not_http_or_gt'
    assert all(golden['resource_gate'].values())
    expected={(r['image_sha256'],r['image_type']):r['candidate_sha256'] for r in golden['rows']}
    plan=json.loads((BASE/'staging/det_batch_e153_r1/plan.json').read_text()); images={}
    for row in plan['rows']:
        data=(Path(plan['images'])/row['file']).read_bytes(); assert hashlib.sha256(data).hexdigest()==row['id']
        images[row['id']]=data
    first,second=[r['id'] for r in plan['rows'][:2]]
    basic=[(first,3),(first,1),(second,3),(second,1)]
    cohort=[(r['image_sha256'],r['image_type']) for r in reversed(golden['rows'])]
    runtime=stage/(output.stem+'_runtime'); runtime.mkdir(mode=0o700)
    assert len(str(runtime/'run_abcdefgh/backend.sock'))<108
    binary=source/'app/autoocr_batch_app'
    if not binary.exists(): shutil.copy2(args.binary,binary)
    assert hashlib.sha256(binary.read_bytes()).digest()==hashlib.sha256(args.binary.read_bytes()).digest()
    opener=build_opener(ProxyHandler({})); process=None; log=None; phase='preflight'
    report=dict(status='running',golden_sha256=hashlib.sha256(args.golden.read_bytes()).hexdigest(),epochs=[],mismatches=[])

    def request(path,data=None):
        req=Request('http://127.0.0.1:5052'+path,data=data,headers={'Content-Type':'application/json'} if data is not None else {})
        try:
            with opener.open(req,timeout=320 if data is not None else 3) as response: return response.status,json.load(response)
        except HTTPError as error:
            try: value=json.load(error)
            except ValueError: value={}
            return error.code,value
    def body(cases):
        data=json.dumps(dict(images=[dict(imageId='image'+str(i),imageType=kind,
            imageBase64=base64.b64encode(images[key]).decode()) for i,(key,kind) in enumerate(cases)])).encode()
        assert len(data)<=24*1024**2
        return data
    def compare(value,cases):
        assert value['status']=='ok' and [v['image_id'] for v in value['results']]==['image'+str(i) for i in range(len(cases))]
        rows=[]
        for item,key in zip(value['results'],cases):
            h=canonical(item['data'])
            if h!=expected[key]: report['mismatches'].append(dict(source_sha256=key[0],image_type=key[1],expected=expected[key],actual=h))
            assert item['status']=='ok' and h==expected[key]
            rows.append(dict(image_id=item['image_id'],source_sha256=key[0],image_type=key[1],canonical_sha256=h,expected_equal=True))
        return rows
    def idle(expected_det_calls=None):
        end=time.monotonic()+6
        while time.monotonic()<end:
            code,state=request('/healthz')
            s=state.get('runtime',{}).get('session',{})
            if code==200 and not state['busy'] and s.get('images')==0 and s.get('registered_bytes')==0 and (
                expected_det_calls is None or state['models']['det']['calls']>=expected_det_calls): return state
            time.sleep(.1)
        raise TimeoutError('Service did not return credits')
    def stop():
        nonlocal process,log
        if process is not None:
            if process.poll() is None: process.terminate(); process.wait(40)
            assert process.returncode==0
        if log: log.close()
        end=time.monotonic()+3
        while list(runtime.glob('run_*')) and time.monotonic()<end: time.sleep(.05)
        assert not list(runtime.glob('run_*'))
        process=log=None
    def raw(headers):
        with socket.create_connection(('127.0.0.1',5052),timeout=10) as peer:
            peer.settimeout(10); peer.sendall(headers); data=peer.recv(4096)
            return int(data.split(b' ',2)[1])

    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            probe.bind(('127.0.0.1',5052))
        for epoch,features in enumerate((True,False,True)):
            phase='startup_'+str(epoch)
            if epoch==2 and args.leave_running:
                runtime=BASE/'runtime/batch-ocr'; runtime.mkdir(mode=0o700,parents=True,exist_ok=True)
                assert runtime.stat().st_uid==os.getuid() and not runtime.stat().st_mode&0o077 and not list(runtime.glob('run_*'))
            env=dict(os.environ,AUTOOCR_BATCH_RUNTIME_BASE=str(runtime))
            for name in ('AUTOOCR_BATCH_DET','AUTOOCR_BATCH_JPEG'): env.pop(name,None)
            if not features: env.update(AUTOOCR_BATCH_DET='0',AUTOOCR_BATCH_JPEG='0')
            log=output.with_name(output.stem+'_epoch'+str(epoch)+'.log').open('xb')
            process=subprocess.Popen(['bash',str(source/'app/start_batch.sh')],env=env,cwd=source/'app',
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            end=time.monotonic()+250
            while time.monotonic()<end and process.poll() is None:
                try:
                    code,state=request('/healthz')
                    if code==200 and state['status']=='ready': break
                except (OSError,ValueError): pass
                time.sleep(.1)
            else: raise TimeoutError('Startup failed')
            assert state['scope']=='batch_ocr_no_extract' and state['auth_mode']=='jetson_dev_no_license'
            assert ('det_batch' in state['models'])==features and state['total_explicit_budget_bytes']==1024**3
            entry=dict(epoch=epoch,features=features,master_pid=process.pid,rows=[]); report['epochs'].append(entry)
            phase='boundaries_'+str(epoch)
            assert request('/Extract',b'{}')[0]==request('/OCR',b'{}')[0]==404
            assert request('/OCRBatch',b'{"images":[]}')[0]==400
            assert request('/OCRBatch',b'{"images":[],"images":[]}')[0]==400
            assert raw(b'POST /OCRBatch HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 25165825\r\nConnection: close\r\n\r\n')==413
            before=idle(); start=time.monotonic()
            assert raw(b'POST /OCRBatch HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 100\r\nConnection: close\r\n\r\n{')==504
            assert 4<=time.monotonic()-start<=9
            after=idle(); assert {k:v['calls'] for k,v in before['models'].items()}=={k:v['calls'] for k,v in after['models'].items()}
            phase='quality_'+str(epoch); result=[]; errors=[]
            def submit():
                try: result.append(request('/OCRBatch',body(basic)))
                except Exception as error: errors.append(type(error).__name__)
            worker=threading.Thread(target=submit); worker.start(); end=time.monotonic()+10
            while time.monotonic()<end:
                _,state=request('/healthz')
                if state['busy']: break
                time.sleep(.01)
            assert state['busy'] and request('/OCRBatch',body(basic))[0]==429
            worker.join(320); assert not worker.is_alive() and not errors and result[0][0]==200
            entry['rows'].extend(compare(result[0][1],basic)); idle()
            for offset in range(0,len(cohort),4):
                cases=cohort[offset:offset+4]; code,value=request('/OCRBatch',body(cases)); assert code==200
                entry['rows'].extend(compare(value,cases)); idle()
                print('HTTP epoch',epoch+1,'cohort pages',min(offset+4,len(cohort)),'/',len(cohort),flush=True)
            code,value=request('/OCRBatch',body(basic[:3])); assert code==200
            entry['rows'].extend(compare(value,basic[:3])); idle()
            code,value=request('/OCRBatch',body([(second,2)])); assert code==200
            alias=compare(value,[(second,1)]); alias[0].update(image_type=2,reference_image_type=1)
            entry['rows'].extend(alias); idle()
            entry['quality_execution']=value['execution']
            execution=entry['quality_execution']
            assert execution['det_batch_enabled']==execution['hardware_jpeg_enabled']==features
            if features:
                assert execution['det_pairs']>0 and execution['jpeg_used']>0 and execution['jpeg_software']>0 and execution['jpeg_failed']==0
            else: assert execution['det_pairs']==execution['jpeg_used']==0
            from PIL import Image
            before_png=idle()['models']['det']['calls']
            with Image.open(io.BytesIO(images[first])) as original:
                image=original.convert('RGB'); image.thumbnail((1500,1500),Image.Resampling.LANCZOS)
                png=io.BytesIO(); image.save(png,format='PNG'); size=image.size; image.close()
            png_body=json.dumps(dict(images=[dict(imageId='image0',imageType=3,imageBase64=base64.b64encode(png.getvalue()).decode())])).encode()
            code,value=request('/OCRBatch',png_body); assert code==200
            summary=statistics(value,1); assert (summary['rows'][0]['width'],summary['rows'][0]['height'])==size
            entry['png_format_smoke']=summary['rows'][0]; entry['png_note']='Derived PNG format/bounds smoke, not GT accuracy.'
            state=idle(before_png+1); assert state['recent_pages']['image0']['image_source']=='encoded'
            bad=dict(images=[dict(imageId='image0',imageType=3,imageBase64=base64.b64encode(b'bad').decode())])
            code,value=request('/OCRBatch',json.dumps(bad).encode()); assert code==200 and value['status']=='partial' and value['results'][0]['status']=='error'
            entry['health_after']=idle(); assert entry['health_after']['models']['ner']['calls']==1
            if features: assert entry['health_after']['models']['det_batch']['calls']>0
            entry['last_execution']=result[0][1]['execution']
            print('HTTP epoch passed',epoch+1,'strict pages',len(entry['rows']),flush=True)
            if epoch<2 or not args.leave_running:
                phase='shutdown_'+str(epoch); stop(); entry['stopped_cleanly']=True
        report.update(status='http_public_quality_boundaries_restart_rollback_passed',left_running=bool(args.leave_running))
        if args.leave_running:
            report['master_pid']=process.pid
            report['active_runtime_root']=str(runtime)
            with output.with_suffix('.pid').open('x') as file: file.write(str(process.pid)+'\n')
            log.close()
    except Exception as error:
        report.update(status='failed',phase=phase,error_type=type(error).__name__,error_errno=getattr(error,'errno',None))
        try: report['failure_health']=request('/healthz')[1]
        except Exception as health_error: report['failure_health_error']=type(health_error).__name__
        if process is not None:
            try: stop()
            except Exception as cleanup: report['cleanup_error']=type(cleanup).__name__
    finally:
        with output.open('x') as file: json.dump(report,file,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k!='epochs'}),flush=True)
    assert report['status']=='http_public_quality_boundaries_restart_rollback_passed'


if __name__=='__main__': main()
