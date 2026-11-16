"""Quality-only replay of actual ordinary/enhanced det tensors in original OCR.

Batch inference happens before candidate page replay; never time this as a
pipeline speedup. HTTP/Extract/license/counter paths are not called.
"""
import argparse
import contextlib
import json
import logging
import os
from pathlib import Path
import sys
import subprocess
import tempfile

import numpy as np
import det_batch2 as b


def business_groups(records):
    groups=[]
    for kind in (3,1):
        indices=[i for i,row in enumerate(records) if row['image_type']==kind]
        groups.extend([[indices[i] for i in group] for group in b.group_pairs([records[i] for i in indices])])
    if sorted(i for group in groups for i in group)!=list(range(len(records))):
        raise ValueError('Missing/duplicate business task')
    if any(len({records[i]['image_sha256'] for i in group})!=len(group) for group in groups):
        raise ValueError('Pair must contain different source images')
    return groups


def replay(index, records, outputs, seen, feeds):
    if index in seen or set(feeds) != {'x'}:
        raise ValueError('Duplicate page or unexpected input')
    np.testing.assert_array_equal(feeds['x'], records[index]['tensor'])
    value=outputs[index]
    if value.shape!=(1,1,*feeds['x'].shape[2:]) or value.dtype!=np.float32 or not np.isfinite(value).all():
        raise ValueError('Invalid per-page output')
    seen.add(index)
    return [value.copy()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args();stage=a.stage.resolve()
    if not stage.is_relative_to(b.q.BASE/'staging'):p.error('Isolated project stage required')
    plan=b.load_plan(stage);built=b.require(stage,'build.json','built_not_validated')
    b.require(stage,'check.json','gpu_det_boxes_consistent_not_full_business_validated')
    target=stage/'business_quality.json'
    if target.exists():raise FileExistsError(str(target))
    state=subprocess.check_output(['systemctl','show','autoocr-jetson-dev.service','--property=ActiveState'],text=True)
    if state.strip()!='ActiveState=inactive':raise RuntimeError('Keep original service stopped')
    if b.q.memory()['MemAvailable_kib']<8*1024**2:raise RuntimeError('Eight GiB reserve required')
    os.umask(0o077)
    wide=b.q.BASE/'overlays/rec-wide-e124-r1'
    wide_fingerprints={p.name:b.q.digest(p) for p in wide.iterdir() if p.is_file()}
    os.environ.update(AUTOOCR_SOURCE_ROOT=str(b.q.RELEASE),AUTOOCR_BASE_DIR=str(b.q.RELEASE),
                      AUTOOCR_RUNTIME_DIR=str(stage/'runtime'),AUTOOCR_TEMP_DIR=str(stage/'runtime/temp'),
                      AUTOOCR_LOG_FILE=os.devnull,AUTOOCR_REC_WIDE_CONFIG=str(wide/'config.json'))
    sys.path.insert(0,str(wide))
    from serve_rec_wide import load_business
    import rec_wide_adapter as adapter
    from trt82_backend import Session
    logging.disable(logging.CRITICAL)
    report=dict(status='running',script_sha256=b.q.digest(__file__),
                plan_sha256=b.q.digest(stage/'plan.json'),build_sha256=b.q.digest(stage/'build.json'),
                rows=[],memory_before=b.q.memory(),wide_fingerprints=wide_fingerprints,
                note='Original thirteen files x ordinary/enhanced; run_ner=False, startup NER warmup separate. '
                     'Actual det inputs captured unchanged, batched by shape, original per-page consumers replayed. '
                     'No timed pipeline/HTTP/GT acceptance or production scheduler claimed. '
                     'No recognized text/images/tensors persisted; only original source rules compared.')
    models=trial=base=original=None
    records=[];references={};counts={};outputs={};seen=set();active=-1
    cases=[(row,kind) for row in plan['rows'] for kind in (3,1)]
    try:
        with open(os.devnull,'w') as quiet:
            with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                server,pipeline,models,cfg=load_business()
            report['warmup_execution']=models.stats()
            if b.q.memory()['MemAvailable_kib']<2*1024**2:raise RuntimeError('Insufficient loaded-model reserve')
            base=models.models['det'].gpu;original=base.run
            def capture(names,feeds):
                if len(records)!=active:raise ValueError('Expected one det call per reference page')
                result=original(names,feeds)
                row,kind=cases[active];x=feeds['x'];b.validate(x)
                records.append(dict(id=row['id']+':'+str(kind),image_sha256=row['id'],image_type=kind,
                                    input_shape=list(x.shape),tensor=x.copy()))
                return result
            base.run=capture
            for active,(row,kind) in enumerate(cases):
                path=Path(plan['images'])/row['file']
                if b.q.digest(path)!=row['id']:raise ValueError('Source image changed')
                before=models.stats()
                with tempfile.TemporaryDirectory(prefix='ref-',dir=stage) as work:
                    with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                        result=pipeline.process(str(path),work_dir=work,image_type=kind,run_ner=False)
                after=models.stats()
                if not result['success'] or not models.healthy or len(records)!=active+1:raise RuntimeError('Reference OCR failed')
                counts[active]={k:after[k]['calls']-before[k]['calls'] for k in ('layout','cell','yolo','enhance','rec','ner')}
                if counts[active]['ner']!=0:raise ValueError('NER business unexpectedly executed')
                references[active]=result['data']
                print('Reference OCR',active+1,'/',len(cases),flush=True)
            base.run=original
            report['after_reference_execution']=models.stats()
            trial=Session(stage/'det_batch2.engine',built['engine_sha256'],'det_n2_business_quality')
            groups=business_groups(records);before_det=base.calls
            for indices in groups:
                x=np.concatenate([records[i]['tensor'] for i in indices])
                y=b.route(base,trial,x)
                for slot,index in enumerate(indices):outputs[index]=y[slot:slot+1].copy()
            report.update(groups=groups,batch_policy='Same image_type/shape, distinct source images; original N1 tails',
                          batched_execution=trial.stats(),single_tail_calls=base.calls-before_det)
            if trial.calls!=sum(len(g)==2 for g in groups) or base.calls-before_det!=sum(len(g)==1 for g in groups):
                raise ValueError('Actual batch/tail execution differs')
            def use_batch(names,feeds):return replay(active,records,outputs,seen,feeds)
            base.run=use_batch
            for active,(row,kind) in enumerate(cases):
                before=models.stats()
                with tempfile.TemporaryDirectory(prefix='cand-',dir=stage) as work:
                    with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
                        result=pipeline.process(str(Path(plan['images'])/row['file']),work_dir=work,image_type=kind,run_ner=False)
                after=models.stats()
                if not result['success'] or not models.healthy or trial.errors:raise RuntimeError('Candidate OCR failed')
                delta={k:after[k]['calls']-before[k]['calls'] for k in ('layout','cell','yolo','enhance','rec','ner')}
                if delta['ner']!=0:raise ValueError('NER business unexpectedly executed')
                compared=b.q.strict_compare(references[active],result['data'])
                report['rows'].append(dict(index=active,image_sha256=row['id'],image_type=kind,
                    input_shape=records[active]['input_shape'],input_sha256=b.tensor_sha(records[active]['tensor']),
                    counts_equal=delta==counts[active],reference_calls=counts[active],candidate_calls=delta,**compared))
                print('Candidate OCR',active+1,'/',len(cases),flush=True)
            if seen!=set(range(len(cases))):raise ValueError('Missing page replay')
            report.update(status='ocr_business_consistent_not_http_or_gt_validated' if all(
                r['passed'] and r['counts_equal'] for r in report['rows']) else 'completed_with_business_differences_not_accepted',
                after_candidate_execution=models.stats(),memory_after=b.q.memory(),replayed_pages=len(seen))
    except Exception as exc:
        report.update(status='failed',error=type(exc).__name__);raise
    finally:
        if base is not None and original is not None:base.run=original
        errors=[]
        if trial is not None:
            try:trial.close()
            except Exception as exc:errors.append(type(exc).__name__)
        if models is not None:
            try:adapter.close_models(models)
            except Exception as exc:errors.append(type(exc).__name__)
        if errors:report.update(status='failed',cleanup_errors=errors)
        report['fingerprints_unchanged']=b.fingerprints()==plan['fingerprints']
        report['wide_fingerprints_unchanged']={p.name:b.q.digest(p) for p in wide.iterdir() if p.is_file()}==wide_fingerprints
        b.q.save(target,report)
    print(json.dumps(dict(status=report['status'],pages=len(report['rows']),
                          passed=sum(r['passed'] and r['counts_equal'] for r in report['rows']))),flush=True)


if __name__=='__main__':main()
