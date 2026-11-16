"""Synthetic dynamic-shape TRT/ORT checks; no user text or document transfer."""
import argparse
import json
from pathlib import Path
import numpy as np
from jetson_models import cpu_session
from trt82_backend import Session, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--role',choices=('ner','yolo'),required=True)
    ap.add_argument('--onnx',type=Path,required=True)
    ap.add_argument('--engine',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Report exists')
    cpu = cpu_session(args.onnx)
    gpu = Session(args.engine,sha256(args.engine),args.role)
    rng = np.random.default_rng(37)
    inputs = []
    if args.role == 'yolo':
        for h,w in ((64,64),(448,640),(640,448),(640,640)):
            inputs.append({'images':rng.random((1,3,h,w),dtype=np.float32)})
    else:
        for length in (2,128,512,3):
            ids = rng.integers(0,3000,size=(1,length),dtype=np.int64)
            ids[0,0],ids[0,-1] = 101,102
            mask = np.ones_like(ids)
            if length >= 128:
                mask[:,-length//4:] = 0
            inputs.append({'input_ids':ids,'attention_mask':mask,'token_type_ids':np.zeros_like(ids)})
    report = {'status':'running','role':args.role,'model_sha256':sha256(args.onnx),
              'engine_sha256':sha256(args.engine),'comparisons':[]}
    try:
        for feeds in inputs:
            a,b = cpu.run(None,feeds)[0],gpu.run(None,feeds)[0]
            if a.shape != b.shape or not np.isfinite(a).all():
                raise ValueError('Invalid reference')
            diff = np.abs(a.astype(np.float64)-b.astype(np.float64))
            row = {'input_shapes':{n:list(v.shape) for n,v in feeds.items()},
                'output_shape':list(a.shape),'mae':float(diff.mean()),
                'max_abs_error':float(diff.max()),'allclose':bool(np.allclose(a,b,rtol=1e-3,atol=1e-4)),
                'argmax_last_axis_equal':bool(np.array_equal(a.argmax(-1),b.argmax(-1)))}
            report['comparisons'].append(row)
            print(json.dumps(row),flush=True)
        report['status'] = 'synthetic_compared_not_task_validated'
    except Exception as exc:
        report.update(status='error',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        gpu.close()
        with args.output.open('x') as f:
            json.dump(report,f,indent=2)


if __name__ == '__main__':
    main()
