"""Distinct REC owner kernel overlap, reusing the already-tested E166 interval sweep."""
import csv
from collections import Counter
import json
import importlib.util
import math
from pathlib import Path
base=Path(__file__).with_name('analyze.py')
if not base.is_file(): base=Path(__file__).parent.parent/'rec_concurrency/analyze.py'
spec=importlib.util.spec_from_file_location('e166_intervals',base)
intervals_module=importlib.util.module_from_spec(spec); spec.loader.exec_module(intervals_module)
concurrency=intervals_module.concurrency


def analyze(raw,mapping):
    if 'ERR_NVGPUCTRPERM' in raw: raise ValueError('Profiler permission denied')
    owners=mapping['owners']; assert set(owners)=={'0','1','2','3'}
    contexts={r['context_id'] for r in owners.values()}; assert len(contexts)==1
    context=str(next(iter(contexts))); streams={}
    for owner,row in owners.items():
        assert set(row['stream_ids'])=={'narrow','wide'}
        for route,stream in row['stream_ids'].items():
            assert str(stream) not in streams
            streams[str(stream)]=(int(owner),route)
    lines=raw.splitlines(); start=next(i for i,s in enumerate(lines) if s.startswith('"Start","Duration"'))
    reader=csv.reader(lines[start:]); header=next(reader); units=next(reader)
    scale={'ns':.001,'us':1.,'ms':1000.,'s':1e6}
    sa=scale[units[header.index('Start')]]; sd=scale[units[header.index('Duration')]]
    intervals=[]; kernels=[]; counts=Counter(); copies=0; unmapped=0
    for row in reader:
        if len(row)!=len(header): continue
        r=dict(zip(header,row))
        if r['Context']!=context: continue
        if r['Stream'] not in streams:
            unmapped+=not r['Name'].startswith('[CUDA'); continue
        if r['Name'].startswith('[CUDA'):
            copies+=r['Name'].startswith('[CUDA memcpy'); continue
        a=float(r['Start'])*sa; d=float(r['Duration'])*sd
        if not math.isfinite(a+d) or d<0: raise ValueError('Invalid timestamp')
        owner,route=streams[r['Stream']]
        intervals.append((a,a+d,owner)); counts[owner]+=1
        kernels.append(dict(owner=owner,route=route,stream=int(r['Stream']),start_us=a,end_us=a+d,name=r['Name']))
    assert set(counts)=={0,1,2,3},'Each real REC owner must execute mapped kernels'
    result=concurrency(intervals); peak=result.pop('max_rec_streams_over_threshold')
    union=sum(v for k,v in result['duration_us_by_distinct_stream_count'].items() if k>0)
    result['duration_us_by_distinct_owner_count']=result.pop('duration_us_by_distinct_stream_count')
    for r in result['examples']: r['owner_ids']=r.pop('streams')
    witness=[]
    if result['examples']:
        example=result['examples'][-1]; middle=example['start_us']+example['duration_us']/2
        for owner in example['owner_ids']:
            witness.append(next(k for k in kernels if k['owner']==owner and k['start_us']<=middle<k['end_us']))
    result.update(status='actual_http_rec_owner_kernel_intervals_analyzed',max_distinct_rec_owners_over_threshold=peak,
        rec_kernel_count=len(intervals),rec_kernel_count_by_owner=dict(sorted(counts.items())),
        mapped_copy_count=copies,other_model_kernel_count=unmapped,rec_kernel_active_union_us=union,
        at_least_percent_of_rec_active_union={k:100*v/union for k,v in result['at_least_us'].items()},
        peak_witness=witness,note='Real HTTP profiler window; narrow/wide deduplicated per owner, copies/other models excluded. Not sustained parallelism or a throughput benchmark.')
    return result


def main():
    folder=Path(__file__).parent/'trace'
    report=json.loads((folder/'profile.json').read_text())
    assert report['status']=='real_http_trace_closed' and report['closed'] and report['profiler_exit']==0
    assert report['backend_exit']['phase']=='stopped'
    result=analyze((folder/'gpu.csv').read_text(),json.loads((folder/'mapping.json').read_text()))
    out=folder/'analysis.json'; assert not out.exists(); out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
