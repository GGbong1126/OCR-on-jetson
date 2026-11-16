"""Frozen full OCR quality and optional predeclared pipeline benchmark; no HTTP/GT.

One set of original models: serial reference then two CPU worker candidates.
Images/text/results remain on the board in RAM/private temporary paths only.
"""
import argparse
import base64
import contextlib
import io
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import threading
import time

import numpy as np

import ipc_wire as wire
from channel_io import ChannelServer
from ocr_worker_adapter import OCRRuntime, ROLES
from stage_session import encoded, require
from verify_det_workers import client
from worker_bridge import WorkerBridge

BASE=Path('/home/xchq/work/autoocr')
RELEASE=BASE/'releases/jetson-r2'
WIDE=BASE/'overlays/rec-wide-e124-r1'


def sha(data): return hashlib.sha256(data).hexdigest()


def resource_snapshot(pids):
    memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    cpu=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    value=dict(monotonic=time.monotonic(),memory_kib={k:int(memory[k].split()[0]) for k in
        ('MemAvailable','SwapFree','NvMapMemUsed') if k in memory},cpu_total=sum(cpu),cpu_idle=cpu[3]+cpu[4],processes={})
    try: value['gpu_load_raw']=int(Path('/sys/devices/gpu.0/load').read_text())
    except OSError: value['gpu_load_raw']=None
    for pid in pids:
        if pid is None: continue
        try:
            path=Path('/proc')/str(pid)
            status=dict(line.split(':',1) for line in (path/'status').read_text().splitlines())
            stat=(path/'stat').read_text().rsplit(')',1)[1].split()
            value['processes'][str(pid)]={k:int(status[k].split()[0]) for k in ('VmRSS','VmHWM','VmSwap','Threads')}
            value['processes'][str(pid)].update(fd=len(list((path/'fd').iterdir())),cpu_ticks=int(stat[11])+int(stat[12]))
        except (OSError,KeyError): pass
    return value


def latency(values):
    return dict(count=len(values),mean_ms=float(np.mean(values)),p50_ms=float(np.percentile(values,50)),
                p95_ms=float(np.percentile(values,95)),note='Exploratory finite-sample wall latency, not stable tail/GT.')


def check_renderer(runtime):
    from PIL import Image
    from types import SimpleNamespace
    result=dict(success=True,data=dict(_processed_width=100,_processed_height=50,
        RawLines=[dict(text='synthetic',bbox=[-1,2,101,51])],MarkdownBodyLines=['synthetic body'],special={}))
    server=runtime.server; previous=server.pipeline
    stream=io.BytesIO(); Image.new('RGB',(200,100)).save(stream,format='PNG')
    try:
        server.pipeline=SimpleNamespace(process=lambda *a,**k:result)
        with server.app_online.test_request_context('/OCR',method='POST',json=dict(
                img=base64.b64encode(stream.getvalue()).decode(),imageType='3')):
            original=server.api_ocr().get_json()
        require(original['isSuccess'] and runtime.render(result,200,100)==original['msg'])
        require(original['msg']['pages'][0]['blocks'][0]==dict(text='synthetic',x1=0,y1=4,x2=200,y2=100))
        return dict(original_endpoint_assembly_equal=True,original_dimensions=[200,100],
                    scaled_and_clipped_coordinates=True,inference_calls=0,auth_endpoint_flow=False)
    finally: server.pipeline=previous


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-app',type=Path,help='Exercise the real C++ pools instead of the Python validation client')
    parser.add_argument('--all-images',action='store_true')
    parser.add_argument('--det-batch',action='store_true')
    parser.add_argument('--jpeg',action='store_true')
    parser.add_argument('--rec-slots',type=int,choices=(0,1,2,4),default=0)
    parser.add_argument('--trace',action='store_true',help='Separate own-process profiler window, never a benchmark')
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--public-response',action='store_true')
    args=parser.parse_args(); output=args.output.resolve(); require(not output.exists())
    require(not args.trace or (args.rec_slots and not args.benchmark))
    require(output.parent.is_relative_to(BASE/'staging'))
    require(subprocess_state()=='ActiveState=inactive')
    memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    require(int(memory['MemAvailable'].split()[0])>=8*1024**2,'LIMIT')
    plan=json.loads((BASE/'staging/det_batch_e153_r1/plan.json').read_text())
    require(1<=args.repeat<=10 and (args.batch_app is not None or not (args.all_images or args.det_batch or args.jpeg or args.repeat!=1)))
    require(not args.benchmark or (args.batch_app is not None and args.all_images and args.repeat==3))
    rows=plan['rows'] if args.all_images else plan['rows'][:2]
    require(len(rows) in (2,13) and len({r['id'] for r in rows})==len(rows))
    inputs=[]; kinds=[]; frozen=[]
    if args.all_images:
        captured=json.loads((BASE/'staging/det_batch_e153_r1/business_cross_image_r3/business_quality.json').read_text())
        shapes={(v['image_sha256'],v['image_type']):tuple(v['input_shape']) for v in captured['rows']}
        sequence=[(row,kind) for kind in (3,1) for row in sorted(rows,key=lambda r:shapes[(r['id'],kind)])]
    else: sequence=[(row,kind) for row in rows for kind in (3,1)]
    for row,kind in sequence:
        payload=(Path(plan['images'])/row['file']).read_bytes(); require(sha(payload)==row['id'])
        inputs.append(payload); kinds.append(kind); frozen.append(row['id'])
    count=len(inputs); passes=[]; serial_measurement={}; phase=['initializing']; samples=[]; sample_stop=threading.Event()
    resource_endpoints={}; sample_dropped=[0]; sample_thread=None; sample_errors=[]
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(RELEASE),AUTOOCR_BASE_DIR=str(RELEASE),
                      AUTOOCR_RUNTIME_DIR=str(output.parent/'runtime'),AUTOOCR_TEMP_DIR=str(output.parent/'runtime/temp'),
                      AUTOOCR_LOG_FILE=os.devnull,AUTOOCR_REC_WIDE_CONFIG=str(WIDE/'config.json'),
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_AUTOINSTALL='false')
    sys.path.insert(0,str(WIDE))
    from offline_business import load_business
    from rec_wide_adapter import close_models, digest
    sys.path.insert(0,str(BASE/'staging/det_batch_e153_r1'))
    from quantize_yolo_ptq import strict_compare
    def compare_data(a,b):
        if not args.public_response: return strict_compare(a,b)
        coordinates=lambda value:[[tuple(block[k] for k in ('x1','y1','x2','y2')) for block in page['blocks']] for page in value['pages']]
        return dict(passed=a==b,strict_equal=a==b,boxes_exactly_equal=coordinates(a)==coordinates(b),
                    box_count_reference=sum(len(p['blocks']) for p in a['pages']),box_count_candidate=sum(len(p['blocks']) for p in b['pages']))
    wide_fp={p.name:digest(p) for p in WIDE.iterdir() if p.is_file()}
    refs={}; ref_metrics={}; runtime=[]; results=[]; errors=[]; bridge=None; channel=None
    closed=False; boundary={}; stages={}; before_candidate=None; after_candidate=None; app=None; cpp_stats={}; renderer_check={}
    report=dict(status='running',frozen_images=frozen,image_types=kinds,
                note=f'{len(rows)} frozen images x ordinary/enhanced; det_batch={args.det_batch}, hardware_jpeg={args.jpeg}. '
                     'Original full Python business, one model set; serial reference then worker candidate. '
                     'Startup NER warmup separate; run_ner=False and no Extract/HTTP/auth. '
                     'No GT or whole-process RSS hard-limit claim; host model call timing is not pure GPU-kernel time.')
    runtime_type=OCRRuntime; bridge_type=WorkerBridge; channel_type=ChannelServer
    if args.rec_slots:
        sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'multistream'))
        from runtime import AsyncRuntime
        from bridge import MultiBridge
        from stage2 import MultiChannel
        runtime_type=AsyncRuntime; bridge_type=MultiBridge; channel_type=MultiChannel
    def factory():
        def batch_factory():
            from trt82_backend import Session
            return Session(BASE/'staging/det_batch_e153_r1/det_batch2.engine',
                '692064bd77441c918bd2fce877e90e898ce365b0ebe071022bc48b35100f3e3e','det_batch2')
        gpu=runtime_type(load_business,close_models,output.parent,metrics_limit=32,
                       det_batch_factory=batch_factory if args.det_batch else None,public_response=args.public_response,
                       **({'rec_slots':args.rec_slots} if args.rec_slots else {})); runtime.append(gpu)
        gpu.server.log=lambda *args,**kwargs:None
        try:
            if args.public_response: renderer_check.update(check_renderer(gpu))
            stages['warmup']=gpu.models.stats()
            loaded=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            require(int(loaded['MemAvailable'].split()[0])>=2*1024**2,'LIMIT')
            stages['loaded_mem_available_kib']=int(loaded['MemAvailable'].split()[0])
            # Preserve explicit out-of-profile CPU route; never resize/pad or error-fallback.
            before=gpu.models.stats()['rec']
            values=gpu.original['rec'](None,{'x':np.zeros((1,3,48,2305),np.float32)})
            after=gpu.models.stats()['rec']
            require(after['calls']==before['calls'] and after['cpu_fallback']['calls']==before['cpu_fallback']['calls']+1)
            require(all(np.isfinite(v).all() for v in values))
            boundary.update(input_shape=[1,3,48,2305],output_shapes=[list(v.shape) for v in values],
                            gpu_call_delta=after['calls']-before['calls'],cpu_call_delta=1,all_finite=True)
            values=None
            for index,(data,kind) in enumerate(zip(inputs,kinds)):
                refs['image'+str(index)],ref_metrics['image'+str(index)]=gpu.page(data,kind,'image'+str(index))
                print('Reference page',index+1,'/',count,file=sys.__stdout__,flush=True)
            if args.benchmark and not args.det_batch and not args.jpeg:
                phase[0]='serial_measured'; durations=[]; begin=time.monotonic()
                for index,(data,kind) in enumerate(zip(inputs,kinds)):
                    start=time.monotonic(); value,_=gpu.page(data,kind,'image'+str(index))
                    require(value==refs['image'+str(index)]); durations.append((time.monotonic()-start)*1000)
                elapsed=time.monotonic()-begin
                serial_measurement.update(seconds=elapsed,pages=count,pages_per_second=count/elapsed,latency=latency(durations))
                phase[0]='serial_finished'
            stages['after_reference']=gpu.models.stats(); gpu.synchronize()
            if args.rec_slots:
                from collections import Counter
                widths=Counter(w for metric in ref_metrics.values() for w in metric['rec_widths'])
                ledger=[]
                for width,calls in sorted(widths.items()):
                    if width>2304: continue
                    slot=gpu.rec[0]['narrow' if width<=1536 else 'wide']; model=slot.session
                    require(model.context.set_binding_shape(model.inputs[0].index,(1,3,48,width)))
                    shapes=[tuple(model.context.get_binding_shape(item.index)) for item in model.outputs]
                    ledger.append(dict(width=width,calls=calls,route='narrow' if width<=1536 else 'wide',
                        input_bytes=576*width,output_shapes=shapes,
                        output_bytes=sum(int(np.prod(shape))*4 for shape in shapes)))
                report['rec_shape_ledger']=ledger
                with output.with_suffix('.ledger.json').open('x') as file:
                    json.dump(dict(frozen_images=frozen,image_types=kinds,rec_shapes=ledger,
                                   fixed_io_bytes=sum(s.host_bytes+s.device_bytes for s in gpu.slots),
                                   rec_slots=args.rec_slots,core_reservation_bytes=768*1024**2,
                                   active_reallocation=dict(page_bytes=128*1024**2,pages=2,
                                                           slot_io_bytes=240*1024**2,prepare_bytes=16*1024**2)),file,indent=2)
                for slot in gpu.slots:
                    for ptr,_ in slot.session.buffers.values(): slot.session.cuda.call('cudaFree',ptr)
                    slot.session.buffers.clear()
            if args.trace:
                from gpu_trace import start
                report['trace']=start(gpu)
                close=gpu.close
                def traced_close():
                    gpu.cuda.call('cudaProfilerStop'); report['trace']['stopped']=True; close()
                gpu.close=traced_close
            return gpu
        except Exception:
            gpu.synchronize(); gpu.close(); raise
    def cpp_client(path):
        try:
            sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'app'))
            from batch_front_client import exchange
            end=time.monotonic()+5
            while not path.exists() and app.poll() is None and time.monotonic()<end: time.sleep(.01)
            require(path.exists())
            for repeat in range(args.repeat+int(args.benchmark)):
                phase[0]='cpp_warmup' if args.benchmark and repeat==0 else 'cpp_measured'
                order=list(range(count)); reverse=(repeat-int(args.benchmark))%2 if repeat else 0
                order=order[::-1] if reverse else order
                observed={}; durations=[]; begin=time.monotonic()
                for start in range(0,count,4):
                    indices=order[start:start+4]
                    data=exchange(path,[('image'+str(i),kinds[i],inputs[i]) for i in indices])
                    require(all(v['status']=='ok' for v in data['results']))
                    observed.update({v['image_id']:v['data'] for v in data['results']})
                    durations.extend(v['wall_ms'] for v in data['execution'].get('pages',[]))
                    print('C++ candidate pages',min(start+4,count),'/',count,'pass',repeat+1,file=sys.__stdout__,flush=True)
                equal=sum(observed[name]==refs[name] for name in refs)
                elapsed=time.monotonic()-begin
                passes.append(dict(pass_index=repeat,warmup=bool(args.benchmark and repeat==0),input_order=order,full_data_equal_count=equal,
                    seconds=elapsed,pages_per_second=count/elapsed,latency=latency(durations) if durations else {},
                    rows=[dict(image_id=name,full_data_equal=value==refs[name],
                        canonical_sha256=sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()),
                        reference_canonical_sha256=sha(json.dumps(refs[name],sort_keys=True,separators=(',',':'),allow_nan=False).encode()),
                        **compare_data(refs[name],value)) for name,value in observed.items()]))
                require(equal==count)
                if args.benchmark and repeat==0:
                    require(not args.det_batch or runtime[0].batch_calls>0)
                    resource_endpoints['warmed']=resource_snapshot([os.getpid(),app.pid])
                if repeat==args.repeat+int(args.benchmark)-1:
                    results.extend(observed['image'+str(i)] for i in reversed(range(count)))
                    resource_endpoints['measured_end']=resource_snapshot([os.getpid(),app.pid])
            cpp_stats.update(exchange(path)['execution'])
        except Exception as error:
            errors.append(type(error).__name__)
            try: cpp_stats.update(exchange(path)['execution'])
            except Exception: pass
    try:
        logging.disable(logging.CRITICAL)
        if args.benchmark:
            def sample():
                while not sample_stop.is_set():
                    try: row=resource_snapshot([os.getpid(),app.pid if app else None]); row['phase']=phase[0]
                    except Exception as error: sample_errors.append(type(error).__name__); return
                    if len(samples)<7200: samples.append(row)
                    else: sample_dropped[0]+=1
                    sample_stop.wait(.5)
            sample_thread=threading.Thread(target=sample,name='resource-sampler'); sample_thread.start()
        with open(os.devnull,'w') as quiet, contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet), \
                tempfile.TemporaryDirectory(prefix='ocr_',dir=output.parent) as folder, \
                wire.private_listener(Path(folder)/'ocr.sock') as listener:
            channel=channel_type(listener,allow_det_batch=args.det_batch,transport_bytes=64*1024**2,
                                 **({'rec_slots':args.rec_slots} if args.rec_slots else {}))
            def algorithm(payload,kind,api): return runtime[0].algorithm(payload,kind,api)
            bridge=bridge_type(channel,algorithm,factory)
            thread=None
            try:
                end=time.monotonic()+1200
                while not bridge.initialized and time.monotonic()<end: channel.poll(.01); bridge.step()
                require(bridge.initialized and not bridge.init_error)
                before_candidate=runtime[0].models.stats()
                if args.batch_app:
                    app=subprocess.Popen([str(args.batch_app.resolve()),str(Path(folder)/'ocr.sock'),str(Path(folder)/'front.sock'),
                        *(['--det-batch'] if args.det_batch else []),*(['--jpeg'] if args.jpeg else []),
                        *(['--rec-slots='+str(args.rec_slots)] if args.rec_slots else [])],
                        stdout=quiet,stderr=(output.parent/(output.stem+'_cpp.log')).open('x'))
                    thread=threading.Thread(target=cpp_client,args=(Path(folder)/'front.sock',),name='cpp-validation-front')
                else: thread=threading.Thread(target=client,args=(Path(folder)/'ocr.sock',inputs,results,errors,kinds),name='ocr-validation-client')
                thread.start(); end=time.monotonic()+1200
                while thread.is_alive() and time.monotonic()<end: channel.poll(.01); bridge.step()
                thread.join(1); require(not thread.is_alive() and not errors)
                after_candidate=runtime[0].models.stats()
                require(len(results)==count)
                metrics=runtime[0].page_stats
                compared=[]
                # Client collects in reverse input order by explicit image_id.
                for index,data in zip(reversed(range(count)),results):
                    name='image'+str(index); reference=refs[name]
                    delta=compare_data(reference,data)
                    compared.append(dict(image_id=name,image_sha256=frozen[index],image_type=kinds[index],
                        full_data_equal=reference==data,reference_sha256=sha(encoded(reference)),candidate_sha256=sha(encoded(data)),
                        calls_equal=metrics[name]['calls']==ref_metrics[name]['calls'],
                        reference_calls=ref_metrics[name]['calls'],candidate_calls=metrics[name]['calls'],
                        reference_rec_widths=ref_metrics[name]['rec_widths'],candidate_rec_widths=metrics[name]['rec_widths'],**delta))
                owners={m['cpu_thread'] for m in metrics.values()}
                require(len(owners)==2 and runtime[0].owner not in owners)
                require(after_candidate['ner']['calls']==before_candidate['ner']['calls'])
                require(all(v['errors']==0 and v.get('cpu_fallback',{}).get('errors',0)==0 for v in after_candidate.values()))
                expected_calls=(args.repeat+int(args.benchmark))*sum(sum(m['calls'].values()) for m in metrics.values())-runtime[0].batch_calls
                require(bridge.gpu_calls==expected_calls)
                report.update(status='full_python_ocr_workers_consistent_not_http_or_gt' if all(
                    r['full_data_equal'] and r['passed'] and r['boxes_exactly_equal'] and r['calls_equal'] for r in compared)
                    else 'business_differences_not_accepted',rows=compared,cpu_workers_observed=len(owners),
                    candidate_model_calls=bridge.gpu_calls,ner_business_delta=0,boundary_cpu_route=boundary,
                    models_before_candidate=before_candidate,models_after_candidate=after_candidate,stages=stages,
                    output_dtypes={k:sorted(v) for k,v in runtime[0].output_dtypes.items()})
                report.update(passes=passes,expected_count=count,repeat=args.repeat,det_pairs=runtime[0].batch_calls,
                              det_batch_stats=runtime[0].det_batch.stats() if runtime[0].det_batch else None,
                              page_metrics=metrics,serial_measurement=serial_measurement,resource_endpoints=resource_endpoints,
                              public_response=args.public_response,renderer_check=renderer_check)
                if args.rec_slots:
                    gpu=runtime[0]
                    slots=[dict(s.stats(),execution=s.session.stats()) for s in gpu.slots]
                    require(all(v['execution']['errors']==0 and not v['busy'] and not v['failed'] for v in slots))
                    report['async_execution']=dict(rec_slots=args.rec_slots,slots=slots,
                        fixed_io_bytes=sum(v['host_pinned_bytes']+v['device_buffer_bytes'] for v in slots),
                        page_credit_bytes=128*1024**2,rec_prepare_queue_limit=16*1024**2,
                        note='Route aliases share IO storage exclusively; their zero buffer byte entries are not missing allocations.')
                if args.batch_app:
                    require(channel.pid==app.pid and app.poll() is None)
                    require(cpp_stats['active_reserved']==cpp_stats['ingress_reserved']==cpp_stats['result_reserved']==0)
                    require(cpp_stats['input_workers_observed']==cpp_stats['result_workers_observed']==2 and cpp_stats['model_runs']==bridge.gpu_calls)
                    report.update(cpp_execution=cpp_stats,cpp_application_sha256=digest(args.batch_app),
                                  note=report['note'].replace('formal C++ dual-pool application or ','')+' C++ pools and input-order response exercised; no HTTP.')
            finally:
                if app is not None and app.poll() is None: app.terminate(); app.wait(8)
                closed=bridge.close(10)
                if thread and thread.is_alive(): thread.join(6)
                if closed: channel.close()
        require(closed)
        require({p.name:digest(p) for p in WIDE.iterdir() if p.is_file()}==wide_fp)
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,errors=errors,stages=stages,boundary_cpu_route=boundary,passes=passes,
                      cpp_execution=cpp_stats,det_pairs=runtime[0].batch_calls if runtime else None,
                      page_metrics=runtime[0].metrics() if runtime else {})
    finally:
        if sample_thread:
            sample_stop.set(); sample_thread.join(2); require(not sample_thread.is_alive())
            measured=[s for s in samples if s['phase']=='cpp_measured']
            if measured and all(k in resource_endpoints for k in ('warmed','measured_end')):
                warm=resource_endpoints['warmed']; end=resource_endpoints['measured_end']; own=str(os.getpid())
                gate=dict(no_samples_dropped=sample_dropped[0]==0,
                    no_sampler_error=not sample_errors,gpu_activity_available=all(s['gpu_load_raw'] is not None for s in measured),
                    warmed_reserve=warm['memory_kib']['MemAvailable']>=2*1024**2,
                    available_reserve=all(s['memory_kib']['MemAvailable']>=1024**2 for s in measured),
                    rss_growth=end['processes'][own]['VmRSS']-warm['processes'][own]['VmRSS']<=256*1024,
                    nvmap_growth=end['memory_kib']['NvMapMemUsed']-warm['memory_kib']['NvMapMemUsed']<=256*1024,
                    swap_growth=warm['memory_kib']['SwapFree']-end['memory_kib']['SwapFree']<=256*1024,
                    fd_growth=end['processes'][own]['fd']-warm['processes'][own]['fd']<=2,
                    thread_growth=end['processes'][own]['Threads']-warm['processes'][own]['Threads']<=2)
                cpp=str(app.pid)
                gate['cpp_resources']=all(end['processes'][cpp][k]-warm['processes'][cpp][k]<=limit for k,limit in
                    (('VmRSS',256*1024),('fd',2),('Threads',2)))
                report['resource_gate']=gate
                if not all(gate.values()): report['status']='resource_gate_failed_not_accepted'
            else:
                report['resource_gate']=dict(evidence_complete=False)
                if report['status']=='full_python_ocr_workers_consistent_not_http_or_gt': report['status']='missing_resource_evidence_not_accepted'
            report.update(resource_samples=samples,samples_dropped=sample_dropped[0],sampler_errors=sample_errors)
        report.update(closed=closed,workers=bridge.stats() if bridge else None,
                      session=channel.session.stats() if channel else None,io=channel.stats() if channel else None,
                      wide_fingerprints=wide_fp)
        with output.open('x',encoding='utf-8') as file: json.dump(report,file,indent=2,allow_nan=False)
    print(json.dumps(dict(status=report['status'],closed=closed,workers=report['workers'])),flush=True)
    require(report['status']=='full_python_ocr_workers_consistent_not_http_or_gt')


def subprocess_state():
    import subprocess
    return subprocess.check_output(['systemctl','show','autoocr-jetson-dev.service','--property=ActiveState'],text=True).strip()


if __name__=='__main__': main()
