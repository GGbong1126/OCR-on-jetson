"""Add Jetson-only adapters/config/engine hardlinks to a verified source payload.

No systemd change, activation, license writes, or service startup. New files only.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
from trt82_backend import sha256

FILES = ('trt82_backend.py','jetson_models.py','serve_jetson.py',
         'start_jetson.sh','health_jetson.py','autoocr-jetson.service')
ENGINES = {'det':'ocr_det_fp32_hw32_960_r1.engine',
           'rec':'ocr_rec_fp32_w32_1536_r1.engine',
           'enhance':'enhance_fp32_512_r1.engine',
           'yolo':'yolo_fp32_hw64_640_r2.engine',
           'ner':'ner_fp32_s2_512_r4.engine'}
SOURCE = {'det':'pth/onnx_rec/ocr_det.onnx','rec':'pth/onnx_rec/ocr_rec.onnx',
          'yolo':'pth/rec/Det/Det/det_photo/best1.pt',
          'ner':'pth/ner/ner_model_cadre_docs_continuous_v3/ner_model_cadre_docs_continuous_v3/model.safetensors'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--adapter-source',type=Path,required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    if root != Path('/home/xchq/work/autoocr'):
        raise ValueError('Unexpected project root')
    release = root / 'releases/jetson-r1'
    target = release / 'edge_deploy/jetson'
    if target.exists():
        raise ValueError('Adapters already staged; review before changing a release')
    for parent in (target, *target.parents):
        if parent.is_symlink():
            raise ValueError('Symlink destination')
    manifest = json.loads((release/'jetson_source_manifest.json').read_text())
    for name,row in manifest['files'].items():
        if sha256(release/name) != row['sha256']:
            raise ValueError('Source changed')
    checked = {}
    for role,name in ENGINES.items():
        path = root/'staging/jp461-r1/engines'/name
        if not path.is_file() or path.is_symlink():
            raise ValueError('Missing native engine: '+role)
        checked[role] = {'path':'pth/tensorrt/'+name,'sha256':sha256(path)}
        source = release/SOURCE[role] if role in SOURCE else next((release/'pth/imgpre').glob('*.pth'))
        checked[role]['source_sha256'] = sha256(source)
    libraries = [Path('/usr/lib/aarch64-linux-gnu')/n for n in (
        'libnvinfer.so.8.2.1','libnvinfer_plugin.so.8.2.1')]
    libraries += [root/'lib/cuda10.2'/n for n in ('libcublas.so','libcublasLt.so')]
    config = {'schema_version':1,'platform':'L4T32.7.1_TRT8.2.1_CUDA10.2',
        'engines':checked,'cpu_modules':{
            'layout':'Original opset16 model: LayerNormalization/GridSample unsupported by audited TRT8.2 parser',
            'cell':'Original opset16 model: shape Reshape parser failure; GridSample also unsupported'},
        'runtime_libraries':[{'path':str(p),'sha256':sha256(p)} for p in libraries],
        'adapter_sha256':{name:sha256(args.adapter_source/name) for name in FILES},
        'status':'staged_not_activated_or_http_accepted'}
    target.mkdir(mode=0o700)
    for name in FILES:
        shutil.copyfile(str(args.adapter_source/name),str(target/name))
        (target/name).chmod(0o600)
    engines = release/'pth/tensorrt'
    engines.mkdir(mode=0o700)
    for role,name in ENGINES.items():
        os.link(str(root/'staging/jp461-r1/engines'/name),str(engines/name))
    with (target/'deployment.json').open('x') as f:
        json.dump(config,f,indent=2)
    print(json.dumps({'status':config['status'],'engines':list(checked),
                      'cpu_modules':list(config['cpu_modules'])}))


if __name__ == '__main__':
    main()
