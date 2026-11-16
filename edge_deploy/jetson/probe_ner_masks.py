"""E115 synthetic padding-mask checks after a build-time half overflow warning."""
import json
import numpy as np
import evaluate_ner_mixed as e

n,q,old,h=e.n,e.q,e.old,e.h


def inputs(length,segment):
    if length not in (8,128,512) or segment not in (0,1):raise ValueError('Unknown probe')
    ids=np.zeros((1,length),np.int64);ids[0,:4]=[101,100,100,102]
    mask=np.zeros_like(ids);mask[0,:4]=1
    types=np.zeros_like(ids);types[0,2:4]=segment
    return dict(input_ids=ids,attention_mask=mask,token_type_ids=types)


def run(a):
    from trt82_backend import Session
    path=a.stage/'mask_probe.json';old.fresh(path)
    meta=e.metadata(a.stage);checked=n.read(a.stage/'check.json')
    if checked['status']!='finite_checked_not_business_accepted' or any(checked[k]!=v for k,v in meta.items()):raise ValueError('Finite check missing')
    result=dict(meta,status='running',probe_sha256=q.digest(__file__),rows=[],check_sha256=q.digest(a.stage/'check.json'),
        note='Synthetic UNK/padding/segment inputs only, not real padded text or GT. Active-token argmax/allclose diagnostic, never a precision-boundary proof. No timing.')
    base=trial=None
    try:
        cfg=n.read(q.RELEASE/'edge_deploy/jetson/deployment.json');spec=cfg['engines']['ner']
        base=Session(q.RELEASE/spec['path'],spec['sha256'],'original_mask_probe');trial=e.candidate(a.stage)
        for length in (8,128,512):
            for segment in (0,1):
                feeds=inputs(length,segment);x=base.run(None,feeds)[0];y=trial.run(None,feeds)[0];z=trial.run(None,feeds)[0]
                if not np.array_equal(y,z):raise ValueError('Repeat drift')
                result['rows'].append(dict(length=length,segment=segment,inputs=n.input_meta(feeds),
                    reference_sha256=h.sha_tensor(x),candidate_sha256=h.sha_tensor(y),repeat_exact=True,metrics=e.compare(x,y,feeds)))
        result['status']='synthetic_masks_finite_not_accuracy_validated'
    except Exception as err:result.update(status='failed',error=type(err).__name__);raise
    finally:
        for name,obj in (('reference',base),('candidate',trial)):
            if obj is not None:result[name+'_execution']=obj.stats();obj.close()
        q.save(path,result)
    print(json.dumps(dict(status=result['status'],cases=len(result['rows']),changed_label_cases=sum(not r['metrics']['active_token_labels_equal'] for r in result['rows']))),flush=True)


if __name__=='__main__':
    a=h.arguments(('check',))
    if a.stage!=q.BASE/'staging/ner_mixed_e115_r1':raise ValueError('Unexpected stage')
    run(a)
