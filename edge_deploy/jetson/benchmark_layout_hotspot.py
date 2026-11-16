"""E110 quality-gated synchronous layout calls, not full OCR/HTTP or kernel time."""
import argparse
import gc
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time

import evaluate_layout_hotspot as e


def run(a):
    quality=e.require(a.stage,'evaluation.json','all_26_business_consistent_not_gt')
    if len(quality['rows'])!=26 or not all(e.business_passed(r) for r in quality['rows']):
        raise ValueError('Incomplete quality gate')
    checked=e.require(a.stage,'check.json','finite_same_inputs_compared_not_business_accepted')
    path=a.stage/('benchmark_%s_%s.json'%(a.variant,a.round)); e.old.fresh(path)
    report=dict(e.metadata(a.stage),benchmark_script_sha256=e.q.digest(__file__),status='running',
        variant=a.variant,round=a.round,rows=[],warmup=3,repeat=10,
        note='6 original layout calls, same order, 3 warmup + 10 timed cycles in separate processes; rounds reverse variant order. Synchronous Session.run checks/copies/execution includes all 3 original outputs. Not pure kernel/full OCR/HTTP, 60 samples per process has exploratory p95, clocks not locked.')
    native=None
    try:
        rows,cfg,execution=e.build.capture(a); gc.collect()
        if [e.build.describe(r) for r in rows] != [{k:r[k] for k in e.build.describe(rows[0])} for r in checked['rows']]:
            raise ValueError('Original input/output drift')
        from trt82_backend import Session
        report.update(inputs=[e.build.describe(r) for r in rows],before=e.old.telemetry())
        spec=cfg['engines']['layout']
        native=(e.candidate(a.stage) if a.variant=='candidate' else
                Session(e.q.RELEASE/spec['path'],spec['sha256'],'layout_fp32_reference',[(e.build.PLUGIN,e.build.PLUGIN_SHA)]))
        expected=[r['candidate_sha256' if a.variant=='candidate' else 'reference_sha256'] for r in checked['rows']]
        def verify(row,wanted):
            values=e.named_outputs(native,row['feeds'])
            if {k:e.build.sha_tensor(v) for k,v in values.items()}!=wanted:
                raise RuntimeError('Timing output drift')
        for _ in range(3):
            for row,wanted in zip(rows,expected): verify(row,wanted)
        report['after_warmup']=e.old.telemetry()
        for cycle in range(10):
            for i,row in enumerate(rows):
                start=time.perf_counter(); native.run(None,row['feeds'])
                report['rows'].append(dict(cycle=cycle,index=i,wall_ms=(time.perf_counter()-start)*1000))
        for row,wanted in zip(rows,expected): verify(row,wanted)
        report.update(status='measured',pre_post_outputs_exact=True,execution=native.stats(),
                      after=e.old.telemetry(),context_device_bytes=native.engine.device_memory_size,
                      timing=e.old.summary([r['wall_ms'] for r in report['rows']]))
    except Exception as err:
        report.update(status='failed',error=type(err).__name__); raise
    finally:
        if native is not None: native.close()
        e.q.save(path,report)
    print(json.dumps(dict(variant=a.variant,round=a.round,timing=report['timing'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--stage',required=True,type=Path)
    p.add_argument('--variant',choices=('reference','candidate'),required=True); p.add_argument('--round',choices=('r1','r2'),required=True)
    a=p.parse_args(); a.stage=a.stage.resolve()
    if platform.machine()!='aarch64' or not a.stage.is_relative_to(e.q.BASE/'staging') or not a.stage.is_dir():
        p.error('Isolated board stage required')
    sys.dont_write_bytecode=True; sys.path.insert(0,str(e.q.RELEASE/'edge_deploy/jetson'))
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(e.q.RELEASE),AUTOOCR_RUNTIME_DIR=str(a.stage/'runtime'),AUTOOCR_LOG_FILE=os.devnull)
    logging.disable(logging.CRITICAL); run(a)
