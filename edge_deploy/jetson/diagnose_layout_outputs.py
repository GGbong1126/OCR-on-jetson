"""E111 numeric-failure diagnostic; never relabel the failed gate or authorize timing."""
import json
import numpy as np
import evaluate_layout_outputs as e


def match_boxes(ref, out):
    """Unique mutual-best same-class IoU diagnostic, not a GT/acceptance matcher."""
    scores=np.zeros((len(ref),len(out)),np.float64)
    for i,a in enumerate(ref):
        for j,b in enumerate(out):
            if a['label']!=b['label']:continue
            x,y=np.asarray(a['coordinate']),np.asarray(b['coordinate'])
            wh=np.maximum(0,np.minimum(x[2:],y[2:])-np.maximum(x[:2],y[:2]))
            inter=float(np.prod(wh)); union=float(np.prod(x[2:]-x[:2])+np.prod(y[2:]-y[:2])-inter)
            scores[i,j]=inter/union if union>0 else 0
    pairs=[]
    if scores.size:
        for i in range(len(ref)):
            j=int(scores[i].argmax()); score=scores[i,j]
            if (score>=0.99 and np.count_nonzero(scores[i]==score)==1
                    and int(scores[:,j].argmax())==i and np.count_nonzero(scores[:,j]==score)==1):
                pairs.append((i,j))
    return dict(reference_count=len(ref),candidate_count=len(out),unique_matched_count=len(pairs),
        unmatched_reference_count=len(ref)-len(pairs),unmatched_candidate_count=len(out)-len(pairs),
        order_preserved=all(i==j for i,j in pairs),
        matched_min_iou=min((float(scores[i,j]) for i,j in pairs),default=None),
        matched_max_coordinate_delta=max((float(np.max(np.abs(np.asarray(ref[i]['coordinate'])-out[j]['coordinate']))) for i,j in pairs),default=None),
        matched_max_score_delta=max((abs(ref[i]['score']-out[j]['score']) for i,j in pairs),default=None),
        matching='Same-class unique mutual-best IoU >=0.99, diagnostic only; no split/merge or GT.')


def business_boxes(values, feeds, nms):
    rows=values['fetch_name_0']; count=int(values['fetch_name_1'][0]); h,w=feeds['im_shape'][0]
    boxes=[]
    for row in rows[:min(count,len(rows))]:
        if float(row[1])<0.3:continue
        boxes.append(dict(label=int(row[0]),score=float(row[1]),
            coordinate=[float(row[2])*float(w)/800,float(row[3])*float(h)/800,
                        float(row[4])*float(w)/800,float(row[5])*float(h)/800]))
    return boxes,nms(boxes,iou_threshold=0.5)


def run(a):
    checked=e.require(a.stage,'check.json','numeric_not_accepted')
    path=a.stage/'output_diagnostic.json'; e.old.fresh(path)
    report=dict(e.metadata(a.stage),diagnostic_sha256=e.q.digest(__file__),status='running',rows=[])
    trial=None
    try:
        rows,_,_=e.source.capture(a)
        if [e.source.describe(r) for r in rows]!=[{k:r[k] for k in e.source.describe(rows[0])} for r in checked['rows']]:
            raise ValueError('Original capture drift')
        from serve_jetson import deployment
        server,_,_=deployment(); nms=server.OCRPipeline._nms_boxes
        trial=e.candidate(a.stage)
        for row,prior in zip(rows,checked['rows']):
            out=dict(zip(e.b.KEEP,e.retained_outputs(trial,row['feeds'])))
            if {k:e.source.sha_tensor(v) for k,v in out.items()}!=prior['candidate_sha256']:
                raise ValueError('Candidate drift')
            ref=row['reference'];ra,rn=business_boxes(ref,row['feeds'],nms);ca,cn=business_boxes(out,row['feeds'],nms)
            x,y=ref['fetch_name_0'],out['fetch_name_0']
            item=dict(image_sha256=row['image_sha256'],image_type=row['image_type'],
                above_threshold=match_boxes(ra,ca),after_original_nms=match_boxes(rn,cn),
                aligned_rows_class_mismatch_count=int(np.count_nonzero(x[:,0]!=y[:,0])),
                aligned_rows_allclose_failed_count=int(np.count_nonzero(~np.isclose(x,y,rtol=1e-3,atol=1e-4).all(axis=1))),
                aligned_rows_both_below_threshold=int(np.count_nonzero((x[:,1]<0.3)&(y[:,1]<0.3))))
            report['rows'].append(item); print('Diagnosed case',len(report['rows']),item['after_original_nms'],flush=True)
        report.update(status='diagnosed_not_acceptance',original_gate_unchanged='numeric_not_accepted',
            note='Same 6 inputs and exact prior output hashes. Uses original threshold0.3, original source NMS0.5 and coordinate scaling. Match geometry is diagnostic, not full business/GT, and does NOT authorize benchmark or relax original failed gate. Raw matrix errors mix rank/label/score/coordinates; they are not matched-box displacement.')
    except Exception as err:
        report.update(status='failed',error=type(err).__name__);raise
    finally:
        if trial is not None:report['execution']=trial.stats();trial.close()
        e.q.save(path,report)


if __name__=='__main__':
    a=e.source.arguments(('diagnose',));run(a)
