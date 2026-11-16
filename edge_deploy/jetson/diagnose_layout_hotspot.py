"""Postmortem of E110 precision-boundary rejection, without running inference."""
import argparse
import json
from pathlib import Path
import re

import evaluate_layout_hotspot as e


def diagnose(info, checked):
    rows=info.get('Layers',[])
    if (not rows or checked['status']!='failed' or checked.get('error')!='ValueError'
            or len(checked['rows'])!=6 or not all(r['repeat_exact'] for r in checked['rows'])):
        raise ValueError('Expected preserved six-case failed format check')
    targets=[]; unexpected=[]; half_rows=[]; grids=[]; coordinate_casts=[]
    for row in rows:
        tensors=row.get('Inputs',[])+row.get('Outputs',[])
        half=any(re.search(r'\b(?:half|fp16)\b',v.get('Format/Datatype','').lower()) for v in tensors)
        selected=any(e.token(row.get('Name',''),n) for n in e.build.TARGETS) and row.get('ParameterType')=='Convolution'
        if selected:
            if not e.dtype_is(row.get('Inputs',[]),'half') or not e.dtype_is(row.get('Outputs',[]),'half'):
                raise ValueError('Hotspot half contract not present')
            targets.append(row['Name'])
        if half:
            half_rows.append(row['Name'])
            if not selected and row.get('LayerType') not in ('Reformat','NoOp'):
                unexpected.append(row)
            if row.get('LayerType')=='Reformat' and e.dtype_is(row.get('Outputs',[]),'half'):
                source=[v['Name'] for v in row.get('Inputs',[]) if v['Name'] in ('im_shape','scale_factor')]
                if source: coordinate_casts.append(dict(layer=row['Name'],sources=source))
        if 'GridSample' in row.get('Name',''):
            grids.append(dict(name=row['Name'],fp32_io=e.dtype_is(row.get('Inputs',[]),'float') and e.dtype_is(row.get('Outputs',[]),'float')))
    if sorted(targets)!=list(e.build.TARGETS) or not unexpected:
        raise ValueError('Additional half computation not evidenced')
    numeric=[]
    for i,r in enumerate(checked['rows']):
        if not all(v['all_finite'] for v in r['raw'].values()): raise ValueError('Nonfinite case')
        numeric.append(dict(index=i,image_sha256=r['image_sha256'],image_type=r['image_type'],raw=r['raw']))
    return dict(status='precision_boundary_violated_not_accepted',accepted_for_release=False,
        layer_count=len(rows),requested_hotspots=targets,half_format_layer_count=len(half_rows),
        half_format_layers=half_rows,unexpected_half_compute=unexpected,coordinate_half_casts=coordinate_casts,
        grid_sample_count=len(grids),grid_sample_fp32_io_count=sum(v['fp32_io'] for v in grids),
        finite_repeat_cases=6,successful_inference_calls_inferred_from_rows=12,session_error_counter_recorded=False,
        numeric=numeric,full_business_cases=0,benchmark_calls=0,
        note='Actual three half convolutions plus an additional half-I/O fused computation and coordinate casts. Do not reinterpret as intended three-layer-only optimization. Inspector cannot reveal every internal accumulator. Raw aligned-row coordinate error is not matched-box displacement or GT accuracy. No original release changes; no business/performance gate bypass.')


def from_stage(stage):
    checked=json.loads((stage/'check.json').read_text()); info=json.loads((stage/'layers.json').read_text())
    result=diagnose(info,checked)
    result.update(script_sha256=e.q.digest(__file__),check_sha256=e.q.digest(stage/'check.json'),
                  layers_sha256=e.q.digest(stage/'layers.json'),build_sha256=e.q.digest(stage/'build.json'))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,type=Path)
    a=p.parse_args();r=from_stage(a.stage);e.q.save(a.stage/'precision_diagnostic.json',r)
    print(json.dumps({k:r[k] for k in ('status','layer_count','half_format_layer_count','grid_sample_count','grid_sample_fp32_io_count','finite_repeat_cases','full_business_cases','benchmark_calls')}))
