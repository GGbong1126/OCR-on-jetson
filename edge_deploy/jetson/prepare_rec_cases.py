"""Generate three actual-crop FP32 CPU references; don't print recognized text."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from ocr_det import read_image, postprocess, sha256_file
from ocr_rec import Recognizer, crop_text_region, preprocess_rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image', type=Path, required=True)
    ap.add_argument('--det-case', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--dict', dest='dictionary', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Output exists')
    image = read_image(args.image)
    boxes = postprocess(np.load(args.det_case / 'reference.npy', allow_pickle=False), image.shape)
    rec = Recognizer(args.model, args.dictionary, threads=2)
    tensors = [preprocess_rec(crop_text_region(image, b)) for b in boxes]
    ordered = sorted(range(len(tensors)), key=lambda i: tensors[i].shape[-1])
    args.output.mkdir(parents=True)
    rows = []
    for i in (ordered[0], ordered[len(ordered)//2], ordered[-1]):
        directory = args.output / ('crop_%02d' % i)
        directory.mkdir()
        output = rec.run_tensor(tensors[i])
        if not np.isfinite(output).all():
            raise ValueError('Nonfinite reference')
        np.save(str(directory / 'x.npy'), tensors[i], allow_pickle=False)
        np.save(str(directory / 'reference.npy'), output, allow_pickle=False)
        row = {'index': i, 'input_shape': list(tensors[i].shape),
            'model_sha256': sha256_file(args.model), 'image_sha256': sha256_file(args.image),
            'arrays_sha256': {p.name: sha256_file(p) for p in directory.glob('*.npy')}}
        (directory / 'case.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
        rows.append(row)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
