"""Export application Torch models for legacy TRT trials; never alter originals.

FP32 CPU + ONNX CPU checks on synthetic inputs only. Not task accuracy acceptance.
Run on the existing development host, not in system Python on the Jetson.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--role', choices=('yolo', 'ner', 'enhance'), required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    source = args.source.resolve()
    if args.output.exists():
        ap.error('Output must be a new directory')
    args.output.mkdir(parents=True)
    os.environ['YOLO_CONFIG_DIR'] = str(args.output.resolve() / 'ultralytics')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(source))
    torch.set_num_threads(2)
    torch.manual_seed(0)
    extra = {}
    if args.role == 'yolo':
        from layout.yolo_compat import ensure_yolo_custom_layers
        ensure_yolo_custom_layers()
        from ultralytics import YOLO
        weights = source / 'pth/rec/Det/Det/det_photo/best1.pt'
        yolo = YOLO(str(weights))
        model = yolo.model.float().cpu().eval()
        model.fuse(verbose=False)
        head = model.model[-1]
        head.export, head.format, head.dynamic = True, 'onnx', True
        names = ['images']
        values = (torch.rand(1, 3, 640, 640),)
        dynamic = {'images': {2: 'height', 3: 'width'}, 'output': {2: 'anchors'}}
        extra = {'names': yolo.names, 'stride': int(model.stride.max())}
    elif args.role == 'enhance':
        from train_mobilenetv3_321 import MobileNetV3_Unet
        files = list((source / 'pth/imgpre').glob('*.pth'))
        if len(files) != 1:
            raise ValueError('Ambiguous enhancement checkpoint')
        weights = files[0]
        checkpoint = torch.load(str(weights), map_location='cpu', weights_only=True)
        model = MobileNetV3_Unet(pretrained=False).float().eval()
        model.load_state_dict(checkpoint.get('G_model', checkpoint), strict=True)
        names, values, dynamic = ['images'], (torch.rand(1, 3, 512, 512),), {}
    else:
        from transformers import AutoModelForTokenClassification
        directory = source / 'pth/ner/ner_model_cadre_docs_continuous_v3/ner_model_cadre_docs_continuous_v3'
        weights = directory / 'model.safetensors'
        original = AutoModelForTokenClassification.from_pretrained(
            str(directory), local_files_only=True, attn_implementation='eager').float().eval()

        class Logits(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, input_ids, attention_mask, token_type_ids):
                return self.inner(input_ids=input_ids, attention_mask=attention_mask,
                                  token_type_ids=token_type_ids, return_dict=False)[0]

        model = Logits(original).eval()
        names = ['input_ids', 'attention_mask', 'token_type_ids']
        values = (torch.randint(0, original.config.vocab_size, (1, 32), dtype=torch.int64),
                  torch.ones((1, 32), dtype=torch.int64), torch.zeros((1, 32), dtype=torch.int64))
        dynamic = {name: {1: 'sequence'} for name in names + ['output']}
        extra = {'model_type': original.config.model_type, 'num_labels': original.config.num_labels}
    model.eval()
    with torch.inference_mode():
        reference = model(*values).detach().numpy()
    path = args.output / (args.role + '.onnx')
    torch.onnx.export(model, values, str(path), input_names=names, output_names=['output'],
                      dynamic_axes=dynamic, opset_version=13, do_constant_folding=True,
                      dynamo=False, external_data=False)
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    feeds = {name: value.numpy() for name, value in zip(names, values)}
    candidate = session.run(None, feeds)[0]
    if candidate.shape != reference.shape or not np.isfinite(candidate).all():
        raise ValueError('Invalid exported output')
    diff = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    for name, value in feeds.items():
        np.save(str(args.output / (name + '.npy')), value, allow_pickle=False)
    np.save(str(args.output / 'reference.npy'), reference, allow_pickle=False)
    result = {'role': args.role, 'status': 'synthetic_cpu_export_compared_not_task_validated',
              'source_sha256': sha256(weights), 'onnx_sha256': sha256(path),
              'torch': torch.__version__, 'onnxruntime': ort.__version__, 'opset': 13,
              'inputs': {n: list(v.shape) for n, v in feeds.items()},
              'output_shape': list(candidate.shape), 'mae': float(diff.mean()),
              'max_abs_error': float(diff.max()),
              'allclose_rtol_1e_3_atol_1e_4': bool(np.allclose(reference, candidate, rtol=1e-3, atol=1e-4)),
              'model_info': extra,
              'array_sha256': {p.name: sha256(p) for p in args.output.glob('*.npy')}}
    with (args.output / 'export_report.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
