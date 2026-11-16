"""Prepare a real-image FP32 input/reference on the existing Windows/WSL CPU stack.

No OCR text or image contents are printed. The case directory contains private
image-derived arrays: keep it out of git. Pre/postprocessing uses the existing
project baseline without modifying it.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from ocr_det import preprocess, postprocess, read_image, sha256_file


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--image', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Output exists; choose a new case directory')
    image = read_image(args.image)
    tensor = np.ascontiguousarray(preprocess(image))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(args.model), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    reference = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    if reference.dtype != np.float32 or not np.isfinite(reference).all():
        raise ValueError('Invalid CPU reference')
    boxes = postprocess(reference, image.shape)
    args.output.mkdir(parents=True)
    np.save(str(args.output / 'input.npy'), tensor, allow_pickle=False)
    np.save(str(args.output / 'reference.npy'), reference, allow_pickle=False)
    meta = {'model_sha256': sha256_file(args.model), 'image_sha256': sha256_file(args.image),
            'image_shape': list(image.shape), 'input_shape': list(tensor.shape),
            'input_npy_sha256': sha256_file(args.output / 'input.npy'),
            'reference_npy_sha256': sha256_file(args.output / 'reference.npy'),
            'reference_box_count': len(boxes), 'ort': ort.__version__,
            'reference_provider': session.get_providers(),
            'note': 'CPU comparison reference, not ground truth or HTTP validation.'}
    with (args.output / 'case.json').open('x', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
