"""Validate E103 paired seven-model diagnostics, not select a global optimum."""
import argparse
import json
from pathlib import Path

import numpy as np
import profile_model_routes as source
import quantize_yolo_ptq as q


def hint(name):
    # Explicitly a name hint, not kernel or complete ONNX operator attribution.
    lower=name.lower()
    if 'foreignnode' in lower: return 'foreign_node'
    if 'grid' in lower: return 'grid_named'
    if 'reformat' in lower: return 'reformat_named'
    if 'conv' in lower: return 'conv_named'
    return 'other'


def pair(a,b):
    if (a['status']!='profiled_exact_not_optimized_or_gt' or b['status']!=a['status']
            or a['round']!='r1' or b['round']!='r2'):
        raise ValueError('Two completed ordered rounds required')
    for key in ('role','image_sha256','engine_sha256','script_sha256','source_calls','unique_shapes','warmup','repeat_bare','repeat_profile'):
        if a[key]!=b[key]: raise ValueError('Provenance/workload differs')
    ac={c['source_index']:c for c in a['cases']}; bc={c['source_index']:c for c in b['cases']}
    if not ac or set(ac)!=set(bc) or len(ac)!=len(a['cases']) or len(bc)!=len(b['cases']):
        raise ValueError('Case mismatch or duplicates')
    for report in (a,b):
        if report['bare_execution']['errors'] or report['profile_execution']['errors']:
            raise ValueError('Backend error')
        for c in report['cases']:
            if not c['bare_output_exact'] or not c['profile_output_exact']:
                raise ValueError('Output drift')
            if (len(c['bare_wall_ms'])!=report['repeat_bare']
                    or not np.isfinite(c['bare_wall_ms']).all() or min(c['bare_wall_ms'])<=0):
                raise ValueError('Invalid bare timing')
    rows=[]
    for i in sorted(ac):
        x,y=ac[i],bc[i]
        if x['inputs']!=y['inputs'] or x['outputs_sha256']!=y['outputs_sha256']:
            raise ValueError('Input or reference output differs')
        groups={}
        for c in (x,y):
            for layer in c['profile']['layers']:
                key=hint(layer['name'])
                groups[key]=groups.get(key,0)+layer['fraction_of_layer_sum']/2
        if not np.isclose(sum(groups.values()),1):
            raise ValueError('Invalid layer accounting')
        rows.append({'source_index':i,'input_shapes':{k:v['shape'] for k,v in x['inputs'].items()},
                     'bare_mean_ms_r1':float(np.mean(x['bare_wall_ms'])),
                     'bare_mean_ms_r2':float(np.mean(y['bare_wall_ms'])),
                     'layer_name_hint_fractions_mean':groups,
                     'top_layers_r1':x['profile']['layers'][:10],
                     'top_layers_r2':y['profile']['layers'][:10]})
    return {'role':a['role'],'engine_sha256':a['engine_sha256'],'cases':rows,
            'source_calls':a['source_calls'],'source_unique_shapes':a['unique_shapes'],
            'context_device_bytes':a['context_device_bytes'],
            'bare_calls':len(rows)*a['repeat_bare']*2,
            'profile_calls':len(rows)*a['repeat_profile']*2}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args(); rows=[]; hashes={}
    for role in source.ROLES:
        paths=[a.stage/(role+'_'+r+'.json') for r in ('r1','r2')]
        reports=[json.loads(path.read_text()) for path in paths]
        rows.append(pair(*reports))
        hashes.update({path.name:q.digest(path) for path in paths})
    result={'status':'seven_models_paired_diagnostics_validated','rows':rows,'reports_sha256':hashes,
            'bare_calls':sum(r['bare_calls'] for r in rows),'profile_calls':sum(r['profile_calls'] for r in rows),
            'note':'One source image ordinary/enhanced, representative cases only. Normal timing separate from profiler; name-based fractions are not exact kernel attribution. No engine optimized or deployment changed by profiling.'}
    q.save(a.stage/'summary.json',result)
    print(json.dumps({'status':result['status'],'bare_calls':result['bare_calls'],'profile_calls':result['profile_calls']}))


if __name__=='__main__': main()
