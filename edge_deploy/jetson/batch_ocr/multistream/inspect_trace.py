"""Summarize own-process nvprof GPU activity. An empty/denied trace is not proof."""
import argparse
import csv
import json
import math
from pathlib import Path


def analyze(raw,mapping):
    if 'ERR_NVGPUCTRPERM' in raw: raise ValueError('GPU profiler permission denied; no overlap claim')
    lines=raw.splitlines(); start=next((i for i,s in enumerate(lines) if s.startswith('"Start","Duration"')),None)
    if start is None: raise ValueError('Missing GPU activity table')
    reader=csv.reader(lines[start:]); header=next(reader); units=next(reader)
    indices={name:header.index(name) for name in ('Start','Duration','Context','Stream','Name')}
    scale={'us':1.,'ms':1000.,'s':1e6,'ns':.001}
    if units[indices['Start']] not in scale or units[indices['Duration']] not in scale: raise ValueError('Unknown GPU time unit')
    labels={int(v):k for k,v in mapping['stream_ids'].items()}; events=[]
    for row in reader:
        if len(row)!=len(header): continue
        try:
            ctx=int(row[indices['Context']]); stream=int(row[indices['Stream']])
            a=float(row[indices['Start']])*scale[units[indices['Start']]]
            d=float(row[indices['Duration']])*scale[units[indices['Duration']]]
        except ValueError: continue
        if ctx!=mapping['context_id'] or stream not in labels: continue
        if not math.isfinite(a+d) or d<0: raise ValueError('Invalid GPU timestamps')
        name=row[indices['Name']]
        kind='copy' if name.startswith('[CUDA memcpy') else ('other' if name.startswith('[CUDA') else 'kernel')
        events.append(dict(start_us=a,end_us=a+d,stream=stream,role=labels[stream],kind=kind))
    if not any(x['kind']=='kernel' for x in events): raise ValueError('No mapped kernel activities')
    active=[]; counts=dict(det_rec_kernel_pairs=0,rec_rec_kernel_pairs=0,copy_compute_pairs=0); examples=[]
    for e in sorted(events,key=lambda x:x['start_us']):
        active=[p for p in active if p['end_us']>e['start_us']]
        for p in active:
            overlap=min(p['end_us'],e['end_us'])-e['start_us']
            if p['stream']==e['stream'] or overlap<=.1: continue
            key=None
            if e['kind']==p['kind']=='kernel':
                kinds={e['role'].split('_')[0][:3],p['role'].split('_')[0][:3]}
                if kinds=={'det','rec'}: key='det_rec_kernel_pairs'
                elif kinds=={'rec'}: key='rec_rec_kernel_pairs'
            elif {e['kind'],p['kind']}=={'kernel','copy'}: key='copy_compute_pairs'
            if key:
                counts[key]+=1
                if len(examples)<12: examples.append(dict(kind=key,roles=[p['role'],e['role']],overlap_us=overlap))
        active.append(e)
    return dict(status='gpu_activity_analyzed',mapped_activities=len(events),counts=counts,examples=examples,
                overlap_threshold_us=.1,note='Own candidate profiler window only; observed activity overlap is not a throughput guarantee or GT.')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--csv',type=Path,required=True); p.add_argument('--quality',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    if args.output.exists(): raise ValueError('Use a new output filename')
    report=json.loads(args.quality.read_text()); mapping=report['trace']
    if not mapping.get('stopped'): raise ValueError('Profiler window not confirmed stopped')
    result=analyze(args.csv.read_text(),mapping)
    args.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,sort_keys=True))


if __name__=='__main__': main()
