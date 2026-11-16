"""Original model pipeline regression, without HTTP/activation/counter operations.

Hold det/rec/YOLO/NER/enhancement native backends fixed. Compare original CPU
layout/cell to candidate GPU layout/cell via the unchanged OCRPipeline.process.
Only booleans, counts, technical hashes and durations enter the report, never text.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
from time import perf_counter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--image-type',type=int,choices=(1,3),default=3)
    p.add_argument('--limit',type=int)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    release,images,output=a.release.resolve(),a.images.resolve(),a.output.resolve()
    if output.exists():p.error('Report exists')
    if release!=Path('/home/xchq/work/autoocr/releases/jetson-r2'):
        raise ValueError('Only the verified r2 diagnostic release is supported')
    os.environ['AUTOOCR_SOURCE_ROOT']=str(release)
    os.environ['AUTOOCR_RUNTIME_DIR']='/home/xchq/work/autoocr/runtime'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(release/'edge_deploy/jetson'))
    from serve_jetson import deployment,load_models
    from jetson_models import CheckedCPU,cpu_session
    from trt82_backend import sha256
    server,root,config=deployment()
    pipeline,models=load_models(server,root,config)
    reference={r:CheckedCPU(cpu_session(root/('pth/onnx_rec/'+r+'_det.onnx')),r,'regression_reference')
               for r in ('layout','cell')}
    gpu={'layout':pipeline._layout_detector,'cell':pipeline._table_cell_processor}
    paths=sorted(f for f in images.iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png'))
    if a.limit is not None:paths=paths[:a.limit]
    if not paths:raise ValueError('No images')
    report={'status':'running','mode':'offline_model_pipeline_not_http', 'image_type':a.image_type,
            'deployment_sha256':sha256(release/'edge_deploy/jetson/deployment.json'),
            'engine_sha256':{r:v['sha256'] for r,v in config['engines'].items()},
            'warmup':models.stats(),'rows':[],'note':__doc__}
    try:
        for index,path in enumerate(paths):
            results=[];durations=[]
            for sessions in (reference,gpu):
                pipeline._layout_detector=sessions['layout'];pipeline._table_cell_processor=sessions['cell']
                start=perf_counter()
                result=pipeline.process(str(path),image_type=a.image_type,run_ner=True)
                durations.append(perf_counter()-start)
                if not result['success'] or not models.healthy or any(s.errors for s in reference.values()):
                    raise RuntimeError('Pipeline backend error, including a swallowed optional-stage error')
                results.append(result['data'])
            ref,out=results
            keys=('FullText','Text','MarkdownBodyLines','NER','special')
            equality={k:ref[k]==out[k] for k in keys}
            raw_equal=([r['text'] for r in ref['RawLines']]==[r['text'] for r in out['RawLines']])
            row={'image_sha256':sha256(path),'reference_box_count':len(ref['Box']),
                 'candidate_box_count':len(out['Box']),'raw_line_count':len(ref['RawLines']),
                 'raw_line_texts_equal':raw_equal,'strict_equal':equality,
                 'boxes_exactly_equal':ref['Box']==out['Box'],
                 'reference_wall_s':durations[0],'candidate_wall_s':durations[1],
                 'passed':raw_equal and all(equality.values())}
            report['rows'].append(row)
            print('Offline model pipeline %d/%d: strict_text_field_consistent=%s'%(index+1,len(paths),row['passed']),flush=True)
        report['status']='offline_pipeline_consistent_not_http_or_gt_validated' if all(r['passed'] for r in report['rows']) else 'differences_not_accepted'
        report['gpu_execution']=models.stats();report['cpu_reference_execution']={r:s.stats() for r,s in reference.items()}
    except Exception as e:
        report.update(status='error',error=type(e).__name__+': '+str(e));raise
    finally:
        pipeline._layout_detector=gpu['layout'];pipeline._table_cell_processor=gpu['cell']
        with output.open('x') as f:json.dump(report,f,indent=2)
        print(json.dumps({'status':report['status'],'images':len(report['rows']),
                          'passed':sum(r['passed'] for r in report['rows'])}))
    if report['status']!='offline_pipeline_consistent_not_http_or_gt_validated':raise SystemExit(2)


if __name__=='__main__':main()
