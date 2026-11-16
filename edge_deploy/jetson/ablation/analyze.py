"""Read-only result comparison; no additive attribution across interventions."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

ORDER = ('sync', 'full', 'one_inflight', 'blocking_submit', 'no_fanout', 'inline_ctc', 'full', 'sync')


def analyze(folder):
    control = json.loads((folder/'control_r3.json').read_text())
    assert control['status'] == 'ablation_complete_e162_restored' and control['restored']
    assert control['measurements_completed'] and len(control['arms']) == 8
    assert control['order'] == list(ORDER)
    assert all(r['equal'] for r in control['restored_checks'])
    arms = []; golden = None
    for index, mode in enumerate(ORDER):
        assert control['arms'][index]['mode'] == mode
        p = folder/('%02d_%s.json' % (index,mode)); r = json.loads(p.read_text())
        host = json.loads(p.with_suffix('.host.json').read_text())
        assert r['closed'] and r['status'] == 'full_python_ocr_workers_consistent_not_http_or_gt'
        assert all(r['resource_gate'].values()) and all(v['full_data_equal_count'] == 26 for v in r['passes'])
        assert all(v['reference_rec_widths'] == v['candidate_rec_widths'] for v in r['rows'])
        assert host['completed'] and host['mode'] == mode and host['snapshot_count'] == 2
        hashes = {(v['image_sha256'],v['image_type']):v['candidate_sha256'] for v in r['rows']}
        if golden is None: golden = hashes
        assert golden == hashes
        passes = [v for v in r['passes'] if not v['warmup']]; assert len(passes) == 3
        seconds = sum(v['seconds'] for v in passes)
        assert seconds == control['arms'][index]['seconds']
        counters = host['measured']
        assert sum(v['calls'] for k,v in counters.items() if k in ('rec.context.execute_v2','rec.context.execute_async_v2')) == 3438
        assert not any(v['calls'] for k,v in counters.items() if k in ('rec.cuda.cudaMalloc','rec.cuda.cudaFree'))
        arms.append(dict(index=index,mode=mode,seconds=seconds,pages_per_second=78/seconds,
                         pass_seconds=[v['seconds'] for v in passes],host=counters,
                         high_inflight=r['workers'].get('rec_high_inflight',0),
                         report_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    sync = [a['seconds'] for a in arms if a['mode']=='sync']
    full = [a['seconds'] for a in arms if a['mode']=='full']
    baseline = mean(sync); candidate = mean(full); saved = baseline-candidate
    for arm in arms:
        arm['added_seconds_vs_full_mean'] = arm['seconds']-candidate
        arm['added_seconds_range_vs_full_controls'] = [arm['seconds']-max(full), arm['seconds']-min(full)]
        arm['fraction_of_observed_saved_time_removed'] = (arm['seconds']-candidate)/saved if saved>0 else None
    return dict(status='eight_arms_quality_resources_and_restoration_verified',arms=arms,
                current_sync_seconds_mean=baseline,current_full_seconds_mean=candidate,
                seconds_saved=saved,throughput_gain_percent=(baseline/candidate-1)*100,
                sync_control_drift_seconds=sync[-1]-sync[0],full_control_drift_seconds=full[-1]-full[0],
                restored_master_pid=control['restored_master_pid'],
                note='Each arm is 78 measured pages over 13 unique images / two types. Baseline means combine two independent processes. Interventions interact; removed fractions are NOT additive causal shares. CUDA counters cover only explicit Python-backend API calls, not vendor-internal allocations. Host calls include waits and nested API intervals; no measured kernel-overlap or GPU-idle claim.')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--directory',type=Path,required=True)
    args = parser.parse_args(); result = analyze(args.directory)
    target = args.directory/'analysis_r1.json'
    with target.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='arms'},indent=2))
    for row in result['arms']:
        print(row['mode'],round(row['seconds'],3),'seconds; delta vs full',round(row['added_seconds_vs_full_mean'],3))


if __name__ == '__main__': main()
