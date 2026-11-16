"""Board-only real-image CPU/TRT regression, no HTTP, activation, or OCR text.

Layout uses unchanged RGB/800 preprocessing and class-wise NMS (0.5).
Cell cases use the CPU layout's actual table crops and unchanged RGB/640 feeds.
Raw numeric metrics are diagnostic; task gate is one-to-one same-class boxes,
IoU>=0.99, coordinate delta<=1 original-image pixel, score delta<=0.001.
This is consistency against CPU, not ground-truth accuracy or a benchmark.
"""
import argparse
import io
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
from trt82_backend import Session, sha256
from check_lowering import metrics


def cpu(path):
    o=ort.SessionOptions(); o.intra_op_num_threads=2
    return ort.InferenceSession(str(path),o,providers=['CPUExecutionProvider'])


def feeds(rgb,size,hw=None):
    height,width=rgb.shape[:2] if hw is None else hw
    return {'image':np.ascontiguousarray((cv2.resize(rgb,(size,size)).astype(np.float32)/255).transpose(2,0,1)[None]),
            'im_shape':np.array([[height,width]],np.float32),
            'scale_factor':np.array([[height/size,width/size]],np.float32)}


def image_pixels(path,mode):
    with Image.open(path) as original:
        image=original.convert('RGB')
        if mode=='business':
            # Exact server._preprocess thumbnail/JPEG encode/decode, in memory.
            image.thumbnail((1500,1500),Image.Resampling.LANCZOS)
            buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=90)
            buffer.seek(0)
            with Image.open(buffer) as decoded:return np.asarray(decoded.convert('RGB'))
        return np.asarray(image)


def iou(a,b):
    l,t=np.maximum(a[:2],b[:2]); r,bot=np.minimum(a[2:],b[2:])
    area=max(0,r-l)*max(0,bot-t)
    return area/(max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-area+1e-6)


def boxes(outputs,hw,size,nms=False,offset=(0,0)):
    detections,count=outputs[0],int(outputs[1].flat[0])
    rows=[]
    for r in detections[:count]:
        if r[1]<0.3:continue
        coord=np.array(r[2:6],np.float64)*np.array([hw[1],hw[0],hw[1],hw[0]])/size
        coord+=np.tile(offset,2)
        rows.append({'class':int(r[0]),'score':float(r[1]),'xyxy':coord})
    if not nms:return rows
    queue=sorted(rows,key=lambda r:r['score'],reverse=True); result=[]
    while queue:
        best=queue.pop(0);result.append(best)
        queue=[r for r in queue if not (r['class']==best['class'] and iou(r['xyxy'],best['xyxy'])>0.5)]
    return result


def compare_boxes(ref,out):
    values=np.array([[iou(a['xyxy'],b['xyxy']) if a['class']==b['class'] else -1
                      for b in out] for a in ref],dtype=float).reshape(len(ref),len(out))
    pairs=[]
    if ref and out:
        for i,a in enumerate(ref):
            j=int(np.argmax(values[i]))
            if values[i,j]>=0.99 and int(np.argmax(values[:,j]))==i:
                b=out[j]
                pairs.append({'reference':i,'candidate':j,'iou':float(values[i,j]),
                              'coordinate_max_abs':float(np.max(np.abs(a['xyxy']-b['xyxy']))),
                              'score_abs':abs(a['score']-b['score'])})
    passed=(len(pairs)==len(ref)==len(out) and
            all(p['coordinate_max_abs']<=1 and p['score_abs']<=.001 for p in pairs))
    return {'passed':passed,'reference_count':len(ref),'candidate_count':len(out),
            'matched_count':len(pairs),'pairs':pairs}


def ordered(session,inputs):
    start=perf_counter(); values=session.run(None,inputs); elapsed=perf_counter()-start
    named={d.name:v for d,v in zip(session.get_outputs(),values)}
    return [named[n] for n in sorted(named)],elapsed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--plugin',type=Path,required=True)
    p.add_argument('--cell-engine',type=Path)
    p.add_argument('--layout-engine',type=Path)
    p.add_argument('--input-mode',choices=('raw','business'),default='business')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Report exists')
    if not a.cell_engine and not a.layout_engine:p.error('At least one candidate engine required')
    cv2.setNumThreads(1)
    paths=sorted(x for x in a.images.iterdir() if x.suffix.lower() in ('.jpg','.jpeg','.png'))
    if not paths:raise ValueError('No images')
    cpus={'layout':cpu(a.models/'layout_det.onnx')}
    if a.cell_engine:cpus['cell']=cpu(a.models/'cell_det.onnx')
    gpus={}
    report={'status':'running','images':[],'cell_cases':[], 'plugin_sha256':sha256(a.plugin),
            'input_mode':a.input_mode,
            'engine_sha256':{r:sha256(e) for r,e in (('layout',a.layout_engine),('cell',a.cell_engine)) if e},
            'source_hashes':{r:sha256(a.models/(r+'_det.onnx')) for r in cpus},
            'gate':{'min_iou':.99,'max_coordinate_delta_pixels':1,'max_score_delta':.001},
            'note':__doc__}
    try:
        for role,engine in (('cell',a.cell_engine),('layout',a.layout_engine)):
            if engine:gpus[role]=Session(engine,sha256(engine),role,[(a.plugin,sha256(a.plugin))])
        for image_index,path in enumerate(paths):
            rgb=image_pixels(path,a.input_mode); hw=rgb.shape[:2]
            x=feeds(rgb,800)
            ref,rt=ordered(cpus['layout'],x)
            rb=boxes(ref,hw,800,nms=True)
            row={'image_sha256':sha256(path),'hw':list(hw),'cpu_layout_wall_s':rt,'cpu_box_count':len(rb)}
            if 'layout' in gpus:
                out,gt=ordered(gpus['layout'],x)
                row.update(gpu_layout_wall_s=gt,raw=[metrics(r,o) for r,o in zip(ref,out)],
                           task=compare_boxes(rb,boxes(out,hw,800,nms=True)))
            report['images'].append(row)
            if 'cell' in gpus:
                tables=[r for r in rb if r['class']==21]
                # A full-page cell diagnostic ensures every image exercises the model;
                # actual table crops are evaluated separately and never substituted.
                cases=[('full_page_diagnostic',(0,0,hw[1],hw[0]))]+[
                    ('cpu_layout_table',tuple(int(v) for v in r['xyxy'])) for r in tables]
                for kind,(x1,y1,x2,y2) in cases:
                    crop=rgb[y1:y2,x1:x2]
                    if crop.size==0:
                        report['cell_cases'].append({'image_index':image_index,'kind':kind,'skipped_empty_crop':True})
                        continue
                    ch,cw=y2-y1,x2-x1
                    inputs=feeds(crop,640,(ch,cw))
                    cr,ct=ordered(cpus['cell'],inputs); co,gt=ordered(gpus['cell'],inputs)
                    report['cell_cases'].append({'image_index':image_index,'kind':kind,'crop_hw':[ch,cw],
                        'cpu_wall_s':ct,'gpu_wall_s':gt,'raw':[metrics(r,o) for r,o in zip(cr,co)],
                        'task':compare_boxes(boxes(cr,(ch,cw),640,offset=(x1,y1)),
                                             boxes(co,(ch,cw),640,offset=(x1,y1)))})
            print('Compared image %d/%d; cell cases=%d'%(image_index+1,len(paths),len(report['cell_cases'])),flush=True)
        rows=[r for r in report['images']+report['cell_cases'] if 'task' in r]
        passed=all(r['task']['passed'] and all(o.get('finite',False) for o in r['raw']) for r in rows)
        report['status']='task_consistent_not_gt_validated' if passed else 'differences_not_accepted'
        report['summary']={'image_count':len(paths),'compared_cases':len(rows),
                           'task_passed':sum(r['task']['passed'] for r in rows),
                           'actual_table_crops':sum(r['kind']=='cpu_layout_table' and 'task' in r for r in report['cell_cases'])}
        report['gpu_execution']={r:s.stats() for r,s in gpus.items()}
    except Exception as e:
        report.update(status='error',error=type(e).__name__+': '+str(e));raise
    finally:
        for s in gpus.values():s.close()
        with a.output.open('x') as f:json.dump(report,f,indent=2)
        print(json.dumps({k:report.get(k) for k in ('status','summary','error')},indent=2))
    if report['status']!='task_consistent_not_gt_validated':raise SystemExit(2)


if __name__=='__main__':main()
