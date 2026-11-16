"""Real GPU adapter test using only the native OCR class; no server/license import.

Reads a trusted local server.py AST. No HTTP, activation, counters or private text output.
"""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from release_business import fingerprint, verify
from trt_sessions import attach


def run(root, image_path):
    verified = verify(root)
    for name in ('trt_sessions.py', 'release_business.py'):
        if fingerprint(Path(__file__).with_name(name)) != fingerprint(root / 'edge_deploy/business' / name):
            raise ValueError('Preflight helpers differ from frozen release')
    sys.path.insert(0, str(root))
    tree = ast.parse((root / 'server.py').read_text(encoding='utf-8-sig'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_ONNXOCREngine')
    ns = {'os': os, 'np': np, 'cv2': cv2, 'math': math, 'log': lambda *args: None}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(root / 'server.py'), 'exec'), ns)
    cv2.setNumThreads(1)
    engine = ns['_ONNXOCREngine'](str(root / 'pth/onnx_rec/ocr_det.onnx'), str(root / 'pth/onnx_rec/ocr_rec.onnx'))
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Image decode failed')
    print('Native CPU OCR reference (no service import) ...', flush=True)
    expected = engine.predict(image)
    print('Attach TensorRT and compare same native algorithm ...', flush=True)
    wrapped = attach(engine, root)
    try:
        actual = wrapped.predict(image)
        equal_boxes = len(actual) == len(expected) and all(np.array_equal(a[0], b[0]) for a, b in zip(expected, actual))
        equal_texts = [x[1][0] for x in expected] == [x[1][0] for x in actual]
        real_stats = wrapped.stats()
        fallback_checks = {}
        # Probe the CPU route beyond BOTH profiles with unchanged synthetic tensors.
        for role, shape in (('det', (1, 3, 32, 960)), ('rec', (1, 3, 48, 1537))):
            session = wrapped.sessions[role]
            tensor = np.zeros(shape, np.float32)
            before = session.ort_fallback_calls
            a = session.cpu.run(None, {session.input_name: tensor})[0]
            b = session.run(None, {session.input_name: tensor})[0]
            fallback_checks[role] = bool(np.array_equal(a, b) and session.ort_fallback_calls == before + 1)
        passed = bool(expected and equal_boxes and equal_texts and all(fallback_checks.values())
                      and all(real_stats[r]['trt_calls'] > 0 and real_stats[r]['errors'] == 0 for r in ('det', 'rec')))
        return {'status': 'passed_not_gt_accuracy' if passed else 'failed', 'release': verified,
                'image_sha256': fingerprint(image_path)['sha256'], 'reference_count': len(expected),
                'candidate_count': len(actual), 'ordered_boxes_equal': equal_boxes, 'ordered_texts_equal': equal_texts,
                'real_image_execution': real_stats, 'synthetic_out_of_profile_cpu_exact': fallback_checks,
                'final_execution': wrapped.stats(), 'note': 'No service, enhancement/layout/NER, license or count access. Not benchmark.'}
    finally:
        wrapped.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Reserve a new report before loading models; never overwrite/retry automatically.
    with args.output.open('x', encoding='utf-8') as stream:
        report = {'status': 'interrupted_before_completion'}
        try:
            report = run(args.root.resolve(), args.image)
        except Exception as error:
            report = {'status': 'failed', 'error_type': type(error).__name__}
            raise
        finally:
            json.dump(report, stream, ensure_ascii=True, indent=2)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report['status'] == 'passed_not_gt_accuracy' else 2


if __name__ == '__main__':
    raise SystemExit(main())
