"""E107 full-model numeric, format, business and quality-gated timing checks.

Reuses frozen E104 business comparison/timing code through an explicit temporary
adapter. Its text/field gates are unchanged. Reports contain no OCR text.
"""
import argparse
import contextlib
import gc
import json
import logging
import os
from pathlib import Path
import platform
import re
import sys

import numpy as np
import build_det_cudnn as build
import evaluate_det_hotspot as legacy

q=build.q
old=build.old


def metadata(stage):
    now=build.provenance();b=json.loads((stage/'build.json').read_text())
    if (any(b[k]!=v for k,v in now.items()) or b['status']!='built_not_validated'
            or b['profile']!=[list(v) for v in build.original.PROFILE]
            or b['rewired_consumers']!=[dict(name='Conv.88',input_index=0),dict(name='Conv.91',input_index=0)]
            or b['flags']!=9281 or b['workspace_bytes']!=256*1024**2
            or q.digest(stage/'ranges.json')!=build.micro.probe.RANGES_SHA):raise ValueError('Build contract/provenance differs')
    return dict(script_sha256=q.digest(__file__),build_script_sha256=now['script_sha256'],
                build_sha256=q.digest(stage/'build.json'),engine_sha256=b['engine_sha256'],
                source_sha256=now['source_sha256'],deployment_sha256=now['deployment_sha256'],plan_sha256=now['plan_sha256'],
                comparison_helper_sha256=q.digest(legacy.__file__),business_helper_sha256=q.digest(q.__file__))


def candidate(stage):
    from trt82_backend import Session
    class Mixed(Session):
        def stats(self):return dict(super().stats(),precision='conv87_cudnn_half_with_fp32_boundary_other_compute_fp32')
    return Mixed(stage/'det_cudnn.engine',metadata(stage)['engine_sha256'],'det_cudnn_mixed')


def named(name,token):return re.search(r'(?<![\w.])'+re.escape(token)+r'(?![\w.])',name) is not None


def format_is(tensors,kind):
    pattern=r'\b(?:half|fp16)\b' if kind=='half' else r'\b(?:float|fp32)\b'
    return bool(tensors) and all(re.search(pattern,v.get('Format/Datatype','').lower()) for v in tensors)


def inspect(info,wanted):
    rows=info.get('Layers',[])
    def convolution(token):
        matches=[r for r in rows if named(r.get('Name',''),token) and r.get('ParameterType')=='Convolution']
        if len(matches)!=1:raise ValueError('Unique executed convolution missing: '+token)
        return matches[0]
    conv=convolution('Conv.87')
    if (conv['LayerType']!='CudnnConvolution' or not format_is(conv['Inputs'],'half') or not format_is(conv['Outputs'],'half')
            or build.micro.unsigned(int(conv['TacticValue'],16))!=build.micro.unsigned(wanted['tactic'])):raise ValueError('Executed half cuDNN tactic not evidenced')
    consumers={name:convolution(name) for name in ('Conv.88','Conv.91')}
    for row in consumers.values():
        if not format_is(row['Inputs'],'float') or not format_is(row['Outputs'],'float'):raise ValueError('Both consumer boundaries must be FP32')
    half_rows=[]
    for row in rows:
        half=any(re.search(r'\b(?:half|fp16)\b',t.get('Format/Datatype','').lower()) for t in row.get('Inputs',[])+row.get('Outputs',[]))
        if half:
            half_rows.append(row['Name'])
            if row is not conv and row['LayerType'] not in ('Reformat','NoOp'):raise ValueError('Unexpected additional half computation')
    return dict(layer_count=len(rows),hotspot=conv,consumers=consumers,half_format_layers=half_rows,
                note='Actual executed formats/tactic; not proof of internal accumulator precision.')


def check(a):
    target=a.stage/'check.json';old.fresh(target);old.fresh(a.stage/'layers.json')
    report=dict(metadata(a.stage),status='running',rows=[],synthetic=[])
    native=reference=None
    try:
        rows,cfg,execution=build.original.capture(a,3);gc.collect()
        expected=json.loads((a.stage/'ranges.json').read_text())['rows']
        if [build.original.describe(r) for r in rows]!=[{k:r[k] for k in build.original.describe(rows[0])} for r in expected]:raise ValueError('Original capture drift')
        native=candidate(a.stage)
        for row in rows:
            out=native.run(None,{'x':row['x']})[0];again=native.run(None,{'x':row['x']})[0]
            if not np.array_equal(out,again):raise ValueError('Candidate not repeatable')
            report['rows'].append(dict(build.original.describe(row),**legacy.compare(row['reference'],out),
                candidate_sha256=build.original.sha_tensor(out),repeat_exact=True))
        from trt82_backend import Session
        spec=cfg['engines']['det'];reference=Session(q.RELEASE/spec['path'],spec['sha256'],'det_shape_reference')
        for shape in build.original.PROFILE:
            x=np.random.default_rng(107).uniform(-1,1,shape).astype(np.float32)
            ref=reference.run(None,{'x':x})[0];out=native.run(None,{'x':x})[0]
            report['synthetic'].append(dict(shape=list(shape),input_sha256=build.original.sha_tensor(x),**legacy.compare(ref,out)))
        inspector=native.engine.create_engine_inspector();inspector.execution_context=native.context
        info=json.loads(inspector.get_engine_information(native.trt.LayerInformationFormat.JSON));q.save(a.stage/'layers.json',info)
        report['inspection']=inspect(info,build.provenance()['wanted'])
        report.update(status='finite_same_inputs_compared_not_business_accepted',execution=native.stats(),
            shape_reference_execution=reference.stats(),original_execution=execution,
            note='6 unchanged real tensors plus seeded synthetic min/opt/max. Numeric/mask differences diagnostic, no benchmark/GT/business acceptance.')
    except Exception as e:
        report.update(status='failed',error=type(e).__name__);raise
    finally:
        if native is not None:native.close()
        if reference is not None:reference.close()
        q.save(target,report)
    print(json.dumps(dict(status=report['status'],real_cases=len(report['rows']),synthetic_cases=len(report['synthetic']))),flush=True)


@contextlib.contextmanager
def adapter():
    """Local, single-threaded dependency injection; frozen helper file unchanged."""
    previous=(legacy.metadata,legacy.candidate)
    legacy.metadata=metadata;legacy.candidate=candidate
    try:yield
    finally:legacy.metadata,legacy.candidate=previous


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('check','evaluate','benchmark'))
    p.add_argument('--stage',type=Path,required=True);p.add_argument('--variant',choices=('reference','mixed'),default='mixed')
    p.add_argument('--round',choices=('r1','r2'),default='r1');a=p.parse_args();a.stage=a.stage.resolve()
    if platform.machine()!='aarch64' or not a.stage.is_relative_to(q.BASE/'staging') or not a.stage.is_dir():p.error('Isolated Jetson stage required')
    sys.path.insert(0,str(q.RELEASE/'edge_deploy/jetson'));sys.dont_write_bytecode=True
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(q.RELEASE),AUTOOCR_RUNTIME_DIR=str(a.stage/'runtime'),AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL)
    if a.mode=='check':check(a)
    else:
        with adapter():getattr(legacy,a.mode)(a)


if __name__=='__main__':main()
