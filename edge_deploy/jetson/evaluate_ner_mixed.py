"""E115 actual precision, logits and entity/business checks; no timing."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import ner_mixed_trial as n

q,old,h=n.q,n.old,n.h


def metadata(stage):
    meta=n.metadata(stage);b=n.read(stage/'build.json')
    if b['status']!='built_not_validated' or b['flags']!=1025 or any(b[k]!=v for k,v in meta.items()) or q.digest(stage/'ner_mixed.engine')!=b['candidate_engine_sha256']:raise ValueError('Build changed')
    return dict(meta,build_sha256=q.digest(stage/'build.json'),candidate_engine_sha256=b['candidate_engine_sha256'],evaluator_sha256=q.digest(__file__))


def candidate(stage):
    from trt82_backend import Session
    return Session(stage/'ner_mixed.engine',metadata(stage)['candidate_engine_sha256'],'ner_selective_half')


def compare(x,y,feeds):
    shape=(1,feeds['input_ids'].shape[1],43)
    if x.shape!=shape or y.shape!=shape or x.dtype!=np.float32 or y.dtype!=np.float32 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid logits')
    active=feeds['attention_mask'].astype(bool);diff=np.abs(x.astype(np.float64)-y.astype(np.float64))
    changed=(x.argmax(-1)!=y.argmax(-1))&active
    return dict(all_finite=True,shape=list(shape),active_tokens=int(active.sum()),changed_active_token_count=int(changed.sum()),
        active_token_labels_equal=not bool(changed.any()),mae=float(diff.mean()),max_abs_error=float(diff.max()),
        allclose_diagnostic=bool(np.allclose(x,y,rtol=1e-3,atol=1e-4)))


def inspect(info,protected):
    rows=info.get('Layers',[])
    if not rows or not all(isinstance(r,dict) for r in rows):raise ValueError('Detailed Inspector missing')
    def types(r):return [str(t.get('Format/Datatype','')).lower() for t in r.get('Inputs',[])+r.get('Outputs',[])]
    def half(r):return any(re.search(r'\b(half|fp16)\b',s) for s in types(r))
    half_rows=[r for r in rows if half(r)]
    missing_targets=[name for name in n.TARGETS if not any(name in r.get('Name','') and half(r) for r in rows)]
    missing_protected=[];wrong_protected=[]
    for name in protected:
        found=[r for r in rows if name in r.get('Name','')]
        if not found:missing_protected.append(name)
        elif any(not types(r) or any(not re.search(r'\b(float|fp32)\b',s) for s in types(r)) for r in found):wrong_protected.append(name)
    opaque=[r.get('Name','') for r in rows if 'ForeignNode' in r.get('Name','') or 'myelin' in r.get('LayerType','').lower()]
    # Compact fusion names/I-O do not disclose their internal normalization precision.
    passed=bool(half_rows) and not (missing_targets or missing_protected or wrong_protected or opaque)
    return dict(layers=len(rows),half_format_layers=len(half_rows),unexposed_half_targets=missing_targets,
        unexposed_protected_nodes=missing_protected,non_fp32_protected_nodes=wrong_protected,opaque_fusion_count=len(opaque),
        precision_boundary_verified=passed,
        note='Conservative visible-layer gate. OBEY declarations or a fused layer FP32 interface alone do not independently prove all protected internal arithmetic is FP32. Unexposed fusion is unverified, not proof of half arithmetic.')


def check(a):
    path=a.stage/'check.json';old.fresh(path);old.fresh(a.stage/'layers.json')
    report=dict(metadata(a.stage),status='running',rules=n.RULES,rows=[],boundaries=[]);trial=base=None
    try:
        rows,pages,execution=n.capture(a);plan=n.read(a.stage/'plan.json')
        if [n.describe(r) for r in rows]!=plan['rows'] or pages!=plan['pages']:raise ValueError('Original capture drift')
        trial=candidate(a.stage)
        for row in rows:
            y=trial.run(None,row['feeds'])[0];repeat=trial.run(None,row['feeds'])[0]
            if not np.array_equal(y,repeat):raise ValueError('Candidate repeat drift')
            report['rows'].append(dict(n.describe(row),metrics=compare(row['reference'],y,row['feeds']),repeat_exact=True,candidate_output_sha256=h.sha_tensor(y)))
        from trt82_backend import Session
        cfg=n.read(q.RELEASE/'edge_deploy/jetson/deployment.json');spec=cfg['engines']['ner']
        base=Session(q.RELEASE/spec['path'],spec['sha256'],'ner_original_boundary')
        for length in (2,128,512):
            ids=np.full((1,length),100,np.int64);ids[0,0]=101;ids[0,-1]=102
            feeds=dict(input_ids=ids,attention_mask=np.ones_like(ids),token_type_ids=np.zeros_like(ids))
            x=base.run(None,feeds)[0];y=trial.run(None,feeds)[0]
            report['boundaries'].append(dict(length=length,inputs=n.input_meta(feeds),metrics=compare(x,y,feeds)))
        inspector=trial.engine.create_engine_inspector();inspector.execution_context=trial.context
        info=json.loads(inspector.get_engine_information(trial.trt.LayerInformationFormat.JSON));q.save(a.stage/'layers.json',info)
        report.update(status='finite_checked_not_business_accepted',inspection=inspect(info,plan['graph']['protected_nodes']),original_execution=execution,
            note='Original real inputs repeated exactly; synthetic length2/128/512 only executable/numeric probes, not GT. Raw allclose diagnostic. Entity/field regression separate. No speedup/promotion.')
    except Exception as err:report.update(status='failed',error=type(err).__name__);raise
    finally:
        if trial is not None:report['candidate_execution']=trial.stats();trial.close()
        if base is not None:report['boundary_reference_execution']=base.stats();base.close()
        q.save(path,report)
    print(json.dumps(dict(status=report['status'],real_calls=len(report['rows']),changed_label_calls=sum(not r['metrics']['active_token_labels_equal'] for r in report['rows']),
        precision_boundary_verified=report['inspection']['precision_boundary_verified'],opaque_fusions=report['inspection']['opaque_fusion_count'])),flush=True)


def entity_signature(rows):
    # Scores remain diagnostic; threshold membership and semantic positions are exact.
    return [(str(v.get('entity_group','')),str(v.get('word','')),v.get('start'),v.get('end')) for v in rows]


def trace_compare(x,y):
    same=len(x)==len(y);raw=clean=membership=same;delta=0.
    for a,b in zip(x,y):
        raw=raw and a['raw_signature']==b['raw_signature'];clean=clean and a['clean_signature']==b['clean_signature']
        membership=membership and a['threshold_signature']==b['threshold_signature']
        if a['raw_signature']==b['raw_signature']:
            delta=max(delta,max((abs(i-j) for i,j in zip(a['scores'],b['scores'])),default=0.))
    return dict(postprocess_calls_reference=len(x),postprocess_calls_candidate=len(y),raw_entity_spans_equal=raw,
        cleaned_entity_spans_equal=clean,threshold_membership_equal=membership,max_aligned_entity_score_delta=delta)


def quality_passed(row):
    keys=('FullText','Text','MarkdownBodyLines','NER','raw_line_texts')
    return (all(row['strict_equal'][k] for k in keys) and row['same_ner_inputs'] and row['token_labels_equal']
        and all(row['entities'][k] for k in ('raw_entity_spans_equal','cleaned_entity_spans_equal','threshold_membership_equal'))
        and all(row['fields']['api_special_text_equal'].values()) and row['cpu_fallback_calls']['reference']==row['cpu_fallback_calls']['candidate'])


def quality(a, *, metadata_fn=None, candidate_fn=None):
    # E119 can reuse the exact business/NER gates with an independent FP32 build.
    path=a.stage/'quality.json';old.fresh(path);meta=(metadata_fn or metadata)(a.stage);checked=n.read(a.stage/'check.json')
    if checked['status']!='finite_checked_not_business_accepted' or any(checked[k]!=v for k,v in meta.items()):raise ValueError('Finite check incomplete')
    report=dict(meta,status='running',check_sha256=q.digest(a.stage/'check.json'),rows=[],
        note='26 original image/type pairs. If visible precision boundary is unverified, this remains diagnostic only even if all business rows match. No timing or release.')
    images=q.load_plan(q.BASE/'staging/quant_yolo_e92_r1');models=trial=base=original=post=None
    try:
        with open(os.devnull,'w') as quiet:
            with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):server,pipeline,models,cfg=old.setup(a)
            pp=sys.modules[server.predict_fields_ner.__module__]
            if server.cfg.NER_SCORE_THRESHOLD!=.98:raise ValueError('Threshold changed')
            trial=(candidate_fn or candidate)(a.stage);base=models.models['ner'];original=base.run;post=pp.postprocess_ner_entities
            active='reference';captured=traces=None
            def call(names,feeds):
                if names not in (None,[]) or trial.errors:raise ValueError('Unhealthy NER call')
                im=n.input_meta(feeds);out=(original if active=='reference' else trial.run)(names,feeds)
                captured[active].append(dict(inputs=im,output=out[0],mask=feeds['attention_mask'].copy()))
                return out
            def tracked(entities,text,score_threshold=.98):
                if score_threshold!=.98:raise ValueError('Postprocess threshold changed')
                scores=[float(r['score']) for r in entities]
                if not np.isfinite(scores).all():raise ValueError('Invalid entity confidence')
                before=entity_signature(entities);passing=entity_signature([r for r in entities if r['score']>=score_threshold])
                cleaned=post(entities,text,score_threshold)
                traces[active].append(dict(raw_signature=before,clean_signature=entity_signature(cleaned),threshold_signature=passing,scores=scores))
                return cleaned
            guard=h.GuardedCall(call);pg=h.GuardedCall(tracked);base.run=guard;pp.postprocess_ner_entities=pg
            for image in images['calibration']+images['holdout']:
                for kind in (3,1):
                    captured={v:[] for v in ('reference','candidate')};traces={v:[] for v in captured};values={};counts={}
                    order=('reference','candidate') if len(report['rows'])%2==0 else ('candidate','reference')
                    for active in order:
                        before=models.stats()
                        with contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):result=pipeline.process(str(Path(images['images'])/image['file']),image_type=kind,run_ner=True)
                        guard.require_healthy();pg.require_healthy();after=models.stats()
                        if not result['success'] or not models.healthy or trial.errors:raise RuntimeError('Business error')
                        values[active]=result['data'];counts[active]={k:after[k]['cpu_fallback']['calls']-before[k]['cpu_fallback']['calls'] for k in ('det','rec')}
                    hashes={b:[v['inputs'] for v in captured[b]] for b in captured};same=hashes['reference']==hashes['candidate']
                    labels=same and all(np.array_equal(x['output'].argmax(-1)[x['mask'].astype(bool)],y['output'].argmax(-1)[x['mask'].astype(bool)]) for x,y in zip(captured['reference'],captured['candidate']))
                    ref,out=values['reference'],values['candidate'];row=q.strict_compare(ref,out)
                    row.update(image_sha256=image['sha256'],image_type=kind,order=list(order),inputs=hashes,same_ner_inputs=same,token_labels_equal=labels,
                        entities=trace_compare(traces['reference'],traces['candidate']),fields=q.field_diagnostics(ref,out,server._layout_special_text),cpu_fallback_calls=counts,
                        ner_output_sha256={b:[h.sha_tensor(v['output']) for v in captured[b]] for b in captured},
                        business_sha256={b:hashlib.sha256(json.dumps({k:v[k] for k in ('FullText','Text','MarkdownBodyLines','NER')},sort_keys=True,ensure_ascii=False).encode()).hexdigest() for b,v in values.items()})
                    row['passed']=quality_passed(row);report['rows'].append(row);print('NER business',len(report['rows']),row['passed'],flush=True)
            allrows=len(report['rows'])==26 and all(r['passed'] for r in report['rows'])
            verified=checked['inspection']['precision_boundary_verified']
            report.update(status='quality_consistent_not_benchmarked' if allrows and verified else 'not_accepted',
                business_passed_count=sum(r['passed'] for r in report['rows']),precision_boundary_verified=verified)
    except Exception as err:report.update(status='failed',error=type(err).__name__);raise
    finally:
        if original is not None:base.run=original
        if post is not None:pp.postprocess_ner_entities=post
        if trial is not None:report['candidate_execution']=trial.stats();trial.close()
        if models is not None:report['original_execution']=models.stats()
        old.close_models(models);q.save(path,report)
    print(json.dumps(dict(status=report['status'],passed=report['business_passed_count'],precision_boundary_verified=report['precision_boundary_verified'])),flush=True)


if __name__=='__main__':
    a=h.arguments(('check','quality'))
    if a.stage!=q.BASE/'staging/ner_mixed_e115_r1':raise ValueError('Unexpected stage')
    {'check':check,'quality':quality}[a.mode](a)
