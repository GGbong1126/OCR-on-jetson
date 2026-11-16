"""Recheck a frozen JPEG pixel plan without starting models or the old HTTP service."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from compare_jpeg_rgb import invoke, metrics, pixels, resized_pixels, memory_guard


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def run(probe, images, plan_path, output):
    if output.exists(): raise ValueError('Use a new report')
    plan=json.loads(plan_path.read_text()); fallback=[]
    if 'rows' in plan:
        from PIL.JpegImagePlugin import get_sampling
        cases=[]
        for row in plan['rows']:
            path=Path(plan['images'])/row['file']
            if sha(path)!=row['id']: raise ValueError('Frozen image changed')
            with Image.open(path) as im:
                supported=(im.format=='JPEG' and im.mode=='RGB' and not im.info.get('progressive')
                    and im.width%2==0 and im.height%2==0 and im.width*im.height<=16_000_000 and get_sampling(im)==2)
                case=dict(image_sha256=row['id'],size=list(im.size))
            (cases if supported else fallback).append(case)
    else: cases=plan['cases']
    needed={tuple(r['size']):r['image_sha256'] for r in cases}
    found={}
    for path in sorted(images.rglob('*')):
        if path.is_symlink() or not path.is_file() or path.stat().st_size>16*1024**2: continue
        try:
            with Image.open(path) as im: size=im.size
            if size in needed and sha(path)==needed[size]: found[needed[size]]=path
        except (OSError,ValueError): pass
    if set(found)!=set(needed.values()): raise ValueError('Frozen images not found')
    report=dict(status='running',probe_sha256=sha(probe),plan_sha256=sha(plan_path),rows=[],software_scope=fallback)
    try:
        with tempfile.TemporaryDirectory(prefix='jpeg_',dir=output.parent) as temp:
            root=Path(temp)
            for i,color in enumerate(((15,85,180),(240,30,60),(30,220,60))):
                path=root/f'color{i}.jpg'; raw=root/f'color{i}.rgb'
                Image.new('RGB',(512,256),color).save(path,subsampling=2,quality=95)
                meta=invoke(probe,path,raw); diff=metrics(pixels(path),np.frombuffer(raw.read_bytes(),np.uint8).reshape(256,512,3))
                if diff['mae']>3 or diff['max_abs_error']>3: raise ValueError('Frozen color gate failed')
                report['rows'].append(dict(synthetic=True,metadata=meta,raw_rgb=diff))
            for i,case in enumerate(cases):
                memory_guard(); path=found[case['image_sha256']]; raw=root/f'real{i}.rgb'
                meta=invoke(probe,path,raw); w,h=case['size']
                candidate=np.frombuffer(raw.read_bytes(),np.uint8).reshape(h,w,3)
                diff=metrics(pixels(path),candidate)
                image=Image.fromarray(candidate)
                try: processed=resized_pixels(image)
                finally: image.close()
                with Image.open(path) as original: reference=resized_pixels(original)
                report['rows'].append(dict(image_sha256=case['image_sha256'],metadata=meta,raw_rgb=diff,
                    after_thumbnail_jpeg90=metrics(reference,processed)))
                print('Frozen pixel case completed',i+1,flush=True)
        report['status']='pixel_diagnostic_completed_not_ocr_validated'
    except Exception as error:
        report.update(status='failed',error=type(error).__name__); raise
    finally:
        with output.open('x') as file: json.dump(report,file,indent=2,allow_nan=False)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True); p.add_argument('--images',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); r=run(a.probe.resolve(),a.images.resolve(),a.plan.resolve(),a.output.resolve())
    print(json.dumps(r),flush=True)
