"""Count distinct REC streams with physically overlapping GPU kernel intervals."""
import csv
from collections import Counter
import math


def concurrency(intervals, threshold_us=.1):
    edges = []
    for start, end, stream in intervals:
        if not math.isfinite(start + end) or end < start:
            raise ValueError('Invalid interval')
        if end > start:
            edges.extend(((start, stream, 1), (end, stream, -1)))
    edges.sort()
    active = Counter(); durations = Counter(); peak = 0; examples = []
    i = 0; previous = edges[0][0] if edges else 0
    while i < len(edges):
        now = edges[i][0]; depth = len(active)
        if now > previous:
            durations[depth] += now - previous
            if now - previous > threshold_us and depth > peak:
                peak = depth
                examples.append(dict(start_us=previous, duration_us=now-previous, streams=sorted(active)))
        while i < len(edges) and edges[i][0] == now:
            _, stream, change = edges[i]; active[stream] += change
            if active[stream] == 0: del active[stream]
            if active.get(stream, 0) < 0: raise ValueError('Invalid lifecycle')
            i += 1
        previous = now
    return dict(max_rec_streams_over_threshold=peak, threshold_us=threshold_us,
                duration_us_by_distinct_stream_count=dict(sorted(durations.items())),
                at_least_us={n:sum(v for k,v in durations.items() if k>=n) for n in (2,3,4,8,16)},
                examples=examples)


def analyze(raw, mapping):
    if 'ERR_NVGPUCTRPERM' in raw: raise ValueError('Profiler denied')
    lines = raw.splitlines()
    try: start = next(i for i,s in enumerate(lines) if s.startswith('"Start","Duration"'))
    except StopIteration: raise ValueError('Missing GPU activities')
    reader = csv.reader(lines[start:]); header = next(reader); units = next(reader)
    scales = dict(ns=.001, us=1., ms=1000., s=1e6)
    scale_a = scales[units[header.index('Start')]]; scale_d = scales[units[header.index('Duration')]]
    ids = {str(v) for v in mapping['stream_ids'].values()}; context = str(mapping['context_id'])
    intervals = []; kernels = Counter(); names = Counter(); copies = 0
    for row in reader:
        if len(row) != len(header): continue
        r = dict(zip(header,row))
        if r['Context'] != context or r['Stream'] not in ids: continue
        if r['Name'].startswith('[CUDA'):
            copies += r['Name'].startswith('[CUDA memcpy'); continue
        a = float(r['Start'])*scale_a; d = float(r['Duration'])*scale_d
        intervals.append((a,a+d,int(r['Stream']))); kernels[r['Stream']] += 1; names[r['Name']] += 1
    if not intervals: raise ValueError('No mapped REC kernels')
    value = concurrency(intervals)
    value.update(status='mapped_rec_kernel_concurrency_analyzed',kernel_count=len(intervals),
                 kernel_count_by_stream=dict(kernels),mapped_copy_count=copies,
                 common_kernels=names.most_common(5),
                 note='Profiler window only, distinct REC streams, not outstanding requests or a universal hardware maximum.')
    return value
