"""CPU-only synthetic equivalence check for a portable derived ONNX graph."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import onnxruntime as ort
from trt82_backend import sha256


def metrics(a, b):
    if a.shape != b.shape: return {'shape_equal':False, 'reference':list(a.shape), 'candidate':list(b.shape)}
    d = np.abs(a.astype(np.float64)-b.astype(np.float64))
    return {'shape_equal':True, 'shape':list(a.shape),
            'finite':bool(np.isfinite(a).all() and np.isfinite(b).all()),
            'exact':bool(np.array_equal(a,b)), 'max_abs_error':float(d.max(initial=0)),
            'mae':float(d.mean()) if d.size else 0,
            'allclose':bool(np.allclose(a,b,rtol=1e-3,atol=1e-4))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--size',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('Output exists')
    opt = ort.SessionOptions(); opt.intra_op_num_threads=2
    sessions = [ort.InferenceSession(str(x),opt,providers=['CPUExecutionProvider']) for x in (a.source,a.candidate)]
    rng = np.random.default_rng(31)
    rows = []
    for height,width in ((a.size,a.size),(1075,1518)):
        feeds = {'image':rng.random((1,3,a.size,a.size),dtype=np.float32),
                 'im_shape':np.array([[height,width]],np.float32),
                 'scale_factor':np.array([[height/a.size,width/a.size]],np.float32)}
        outs = [s.run(None,feeds) for s in sessions]
        rows.append({'hw':[height,width], 'outputs':[metrics(x,y) for x,y in zip(*outs)]})
        print(rows[-1],flush=True)
    passed = all(v.get('allclose',False) for r in rows for v in r['outputs'])
    report={'status':'synthetic_cpu_compared_not_task_validated','allclose_all':passed,'rows':rows,
            'source_sha256':sha256(a.source),'candidate_sha256':sha256(a.candidate)}
    with a.output.open('x') as f: json.dump(report,f,indent=2)
    if not passed: raise SystemExit(2)


if __name__=='__main__':main()
