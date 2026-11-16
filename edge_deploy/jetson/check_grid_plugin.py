"""Compare serialized/reloaded plugin engines against ORT GridSample, synthetic only."""
import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, TensorProto as T
import onnxruntime as ort
from build_deformable import build
from trt82_backend import Session, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plugin', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('Output exists')
    a.output.mkdir(parents=True)
    handle = ctypes.CDLL(str(a.plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    rows = []
    rng = np.random.default_rng(42)
    cases = [(1,3,4,5,7,9), (2,5,1,1,3,11), (8,32,80,80,300,4), (1,2,5,1,2,9)]
    for idx, (n,c,hi,wi,ho,wo) in enumerate(cases):
        x = rng.normal(size=(n,c,hi,wi)).astype(np.float32)
        grid = rng.uniform(-1.8, 1.8, (n,ho,wo,2)).astype(np.float32)
        # Corners, center, pixel centers, one-pixel fringe, extreme finite values.
        edges = np.array([[-1,-1],[1,1],[0,0],[-1+1/wi,-1+1/hi],
                          [-1-1/wi,0],[1+1/wi,0],[1e20,0],[-1e20,0]], np.float32)
        grid.reshape(-1,2)[:len(edges)] = edges
        ins = [h.make_tensor_value_info('x',T.FLOAT,list(x.shape)), h.make_tensor_value_info('grid',T.FLOAT,list(grid.shape))]
        outs = [h.make_tensor_value_info('y',T.FLOAT,[n,c,ho,wo])]
        node = h.make_node('GridSample',['x','grid'],['y'], align_corners=0, mode='bilinear',padding_mode='zeros')
        model = h.make_model(h.make_graph([node],'grid',ins,outs),opset_imports=[h.make_opsetid('',16)],ir_version=8)
        opts = ort.SessionOptions(); opts.intra_op_num_threads=2
        reference = ort.InferenceSession(model.SerializeToString(),opts,providers=['CPUExecutionProvider']).run(None,{'x':x,'grid':grid})[0]
        node = model.graph.node[0]; node.op_type='AutoOCRGridSample'; node.domain='autoocr'; del node.attribute[:]
        model.opset_import.append(h.make_opsetid('autoocr',1))
        source = a.output / ('grid%d.onnx'%idx); onnx.save(model,str(source))
        engine = a.output / ('grid%d.engine'%idx)
        meta = build(source,a.plugin,engine)
        s = Session(engine,sha256(engine),'grid_test')
        try:
            output = s.run(None,{'x':x,'grid':grid})[0]
            row = {'shape':list(x.shape),'grid_shape':list(grid.shape),
                   'finite':bool(np.isfinite(output).all()),
                   'max_abs_error':float(np.max(np.abs(reference-output))),
                   'allclose':bool(np.allclose(reference,output,rtol=1e-5,atol=2e-6)), 'build':meta}
            rows.append(row)
            with (a.output/('case%d.json'%idx)).open('x') as f: json.dump(row,f,indent=2)
            print(json.dumps(row),flush=True)
            if not row['finite'] or not row['allclose']: raise RuntimeError('Plugin mismatch')
        finally: s.close()
    with (a.output/'comparison.json').open('x') as f:
        json.dump({'status':'passed_synthetic_not_model_validated','cases':rows},f,indent=2)
    print(json.dumps(rows,indent=2))


if __name__ == '__main__': main()
