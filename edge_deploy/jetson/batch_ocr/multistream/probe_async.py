"""Isolated P1 probe. Refuses to load beside the existing HTTP model set.

Synthetic original-shape FP32 tensors only; this is not full OCR, GT, or a
performance benchmark. Report has no recognized text or input/output tensors.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

from trt_async import Slot

B=Path('/home/xchq/work/autoocr')
R=B/'releases/jetson-r2'
NARROW='da2ff03a315a410ccd1d795af129c4220905d45ea2732da0d4441c8ea914c6ff'
WIDE='8e3f9167d1496011c8bf528df0c5eff006bc8a0d4f254e2d42a9a47009e7c95b'
DET='bf79b526cb3d59c60b3fb4d680c0116bed59c051f85698b45d4f58f287e0a271'


def memory():
    rows=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    return {k:int(rows[k].split()[0]) for k in ('MemAvailable','NvMapMemUsed','SwapFree')}


def tokens(array):
    ids=np.argmax(array[0],axis=1); out=[]; previous=-1
    for i in ids:
        if i==0: previous=-1; continue
        if i!=previous: out.append(int(i))
        previous=i
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--slots',type=int,choices=(1,2,4),required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); output=args.output.resolve()
    assert output.is_relative_to(B/'staging') and output.parent.is_dir() and not output.exists()
    with socket.socket() as sock:
        sock.settimeout(1)
        assert sock.connect_ex(('127.0.0.1',5052))!=0,'Stop the stable service in an authorized window first'
    assert memory()['MemAvailable']>=8*1024**2,'Do not duplicate a loaded model set'
    sys.path.insert(0,str(R/'edge_deploy/jetson'))
    from trt82_backend import Session
    sessions=[]; slots=[]; safe=False
    report=dict(status='running',slots=args.slots,memory_before=memory(),rows=[],max_unfinished_events=0,
        note='Independent engine per context/profile0; CUDA events include scheduling, not a kernel timeline. '
             'Synthetic FP32 original-shape tensors; CTC index consistency is not GT or full business quality. '
             'No repeated throughput benchmark; no GPU overlap claim without a kernel trace.')
    output.write_text(json.dumps(report,indent=2))
    try:
        def create(path,sha,role):
            s=Session(path,sha,role); sessions.append(s)
            slot=Slot(s); slots.append(slot); return slot
        det=create(R/'pth/tensorrt/ocr_det_fp32_hw32_960_r1.engine',DET,'async_det')
        rec=[]
        for i in range(args.slots):
            rec.append(dict(narrow=create(R/'pth/tensorrt/ocr_rec_fp32_w32_1536_r1.engine',NARROW,'async_rec_'+str(i)),
                            wide=create(B/'overlays/rec-wide-e124-r1/rec_wide_fp32.engine',WIDE,'async_wide_'+str(i))))
        report['allocation']=[s.stats() for s in slots]; report['memory_loaded']=memory()
        assert report['memory_loaded']['MemAvailable']>=2*1024**2,'Loaded memory reserve'
        rng=np.random.RandomState(163)
        feeds={w:{'x':rng.uniform(-1,1,(1,3,48,w)).astype(np.float32)} for w in (32,160,1536,1537,2081,2304)}
        det_feed={'x':rng.uniform(0,1,(1,3,960,704)).astype(np.float32)}
        refs={w:rec[0]['narrow' if w<=1536 else 'wide'].session.run(None,f)[0] for w,f in feeds.items()}
        det_ref=det.session.run(None,det_feed)[0]
        # Warm every independent context/route before simultaneous submission.
        for routes in rec:
            for route,w in (('narrow',160),('wide',2081)):
                slot=routes[route]; slot.submit(('warm',route),feeds[w]); slot.cuda.call('cudaStreamSynchronize',slot.stream)
                slot.collect()
        for pattern,widths in (
            ('repeat',[160]*args.slots),('wide_repeat',[2081]*args.slots),
            ('forward',[32,1536,1537,2304][:args.slots]),
            ('reverse',[2304,1537,1536,32][:args.slots])):
            active=[]; begin=time.monotonic()
            det.submit((pattern,'det'),det_feed); active.append((det,None,det_ref))
            for i,w in enumerate(widths):
                slot=rec[i]['narrow' if w<=1536 else 'wide']
                slot.submit((pattern,i,w),feeds[w]); active.append((slot,w,refs[w]))
                report['max_unfinished_events']=max(report['max_unfinished_events'],sum(not s.ready() for s,_,_ in active))
            end=time.monotonic()+120
            while active:
                assert time.monotonic()<end,'Event deadline'
                assert memory()['MemAvailable']>=1024**2,'Running memory reserve'
                progress=False
                for item in list(active):
                    slot,w,ref=item
                    if not slot.ready(): continue
                    ticket,values,timing=slot.collect(); value=values[0]
                    diff=np.abs(value.astype(np.float64)-ref.astype(np.float64))
                    row=dict(pattern=pattern,ticket=list(ticket),width=w,output_shape=list(value.shape),
                             output_bytes=value.nbytes,all_finite=bool(np.isfinite(value).all()),
                             max_abs_error=float(diff.max()),mae=float(diff.mean()),
                             allclose=bool(np.allclose(value,ref,rtol=1e-3,atol=1e-4)),timings=timing)
                    if w is not None: row['ctc_indices_equal']=tokens(value)==tokens(ref)
                    assert row['all_finite'] and row['allclose'] and row.get('ctc_indices_equal',True)
                    report['rows'].append(row); active.remove(item); progress=True
                if not progress: time.sleep(.001)
            report.setdefault('pattern_wall_s',{})[pattern]=time.monotonic()-begin
            output.write_text(json.dumps(report,indent=2))
        report['memory_end']=memory(); report['status']='isolated_async_outputs_consistent'
    except Exception as error:
        report.update(status='failed',error=type(error).__name__,detail=str(error))
        raise
    finally:
        try:
            for slot in slots: slot.quiesce()
            safe=True
            for slot in reversed(slots): slot.close()
            for session in reversed(sessions): session.close()
        finally:
            report['safe_quiescence_confirmed']=safe
            output.write_text(json.dumps(report,indent=2))
            # Unknown GPU safety must keep the process/resources alive for diagnosis.
            if not safe:
                print('QUARANTINED: synchronization failed; do not kill/reset',flush=True)
                while True: time.sleep(1)
    print(json.dumps({k:report[k] for k in ('status','slots','max_unfinished_events')},sort_keys=True))


if __name__=='__main__': main()
