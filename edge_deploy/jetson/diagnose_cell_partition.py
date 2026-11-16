"""Explain E114 cell rank drift separately from actual matched-box displacement.

Diagnostic only: never changes the ordered gate or enables timing/promotion.
"""
import json
import numpy as np
import cell_partition as c

p,q,h=c.p,c.q,c.h


def effective(values,feeds):
    rows=values['fetch_name_0'][:int(values['fetch_name_1'][0])]
    out=rows[rows[:,1]>=.3].astype(np.float64)
    height,width=feeds['im_shape'][0]
    out[:,2:]*=np.array([width,height,width,height])/640
    return out


def match(ref,out):
    scores=np.zeros((len(ref),len(out)))
    for i,x in enumerate(ref):
        wh=np.maximum(0,np.minimum(x[4:6],out[:,4:6])-np.maximum(x[2:4],out[:,2:4]))
        inter=wh.prod(1);union=(x[4:6]-x[2:4]).prod()+(out[:,4:6]-out[:,2:4]).prod(1)-inter
        scores[i]=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)*(out[:,0]==x[0])
    pairs=[]
    if scores.size:
        for i in range(len(ref)):
            j=int(scores[i].argmax());v=scores[i,j]
            if v>=.99 and np.count_nonzero(scores[i]==v)==1 and int(scores[:,j].argmax())==i and np.count_nonzero(scores[:,j]==v)==1:
                pairs.append((i,j))
    return dict(reference_count=len(ref),candidate_count=len(out),matched_count=len(pairs),
        unmatched_reference_indices=[i for i in range(len(ref)) if not any(i==a for a,b in pairs)],
        unmatched_candidate_indices=[j for j in range(len(out)) if not any(j==b for a,b in pairs)],
        reordered_matched_count=sum(i!=j for i,j in pairs),
        min_matched_iou=min((float(scores[i,j]) for i,j in pairs),default=None),
        max_matched_coordinate_delta_pixels=max((float(np.abs(ref[i,2:]-out[j,2:]).max()) for i,j in pairs),default=None),
        note='Same-class unique mutual-best IoU >=0.99, diagnostic only; not a split/merge or GT matcher.')


def run(a):
    target=a.stage/'geometry_diagnostic.json';c.old.fresh(target)
    checked=p.read(a.stage/'check.json');meta=c.metadata(a.stage)
    if any(checked[k]!=v for k,v in meta.items()) or checked['status']!='full_cell_formats_checked_not_business_accepted':raise ValueError('Check drift')
    result=dict(meta,status='running',diagnostic_sha256=q.digest(__file__),check_sha256=q.digest(a.stage/'check.json'),rows=[])
    trial=None
    try:
        rows,_,execution=p.capture(a)
        if [p.describe(r) for r in rows]!=[{k:r[k] for k in p.describe(rows[0])} for r in checked['rows']]:raise ValueError('Capture drift')
        trial=c.Composed(a.stage)
        for row,former in zip(rows,checked['rows']):
            out=trial.run(row['feeds'],'mixed')
            if {k:h.sha_tensor(v) for k,v in out.items()}!=former['variants']['mixed']['output_sha256']:raise ValueError('Candidate drift')
            r=match(effective(row['reference'],row['feeds']),effective(out,row['feeds']))
            result['rows'].append(dict(image_sha256=row['image_sha256'],image_type=row['image_type'],matching=r))
            print('Geometry diagnostic',len(result['rows']),r,flush=True)
        result.update(status='diagnosed_not_acceptance',original_execution=execution,
            note='Exact prior input/output hashes verified. Ordered geometry/business gate remains unchanged; no speed/GT/promotion claim.')
    except Exception as err:result.update(status='failed',error=type(err).__name__);raise
    finally:
        if trial is not None:result['execution']=trial.stats();trial.close()
        q.save(target,result)


if __name__=='__main__':
    a=h.arguments(('diagnose',))
    if a.stage!=q.BASE/'staging/cell_partition_e114_r2':raise ValueError('Unexpected stage')
    run(a)
