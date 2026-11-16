"""Trace only this synthetic child decoder. No attach, global ftrace, models or user images."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw
from compare_jpeg_rgb import device_calls, metrics, pixels


def run(probe, tracer, output):
    trace=output.with_suffix('.strace')
    if output.exists() or trace.exists(): raise FileExistsError('Preserve previous evidence')
    report=dict(status='running',probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),
                tracer_sha256=hashlib.sha256(tracer.read_bytes()).hexdigest(),synthetic_size=[512,258],global_ftrace=False)
    try:
        with tempfile.TemporaryDirectory(prefix='submission_',dir=output.parent) as folder:
            path=Path(folder)/'synthetic.jpg'; raw=Path(folder)/'synthetic.rgb'
            image=Image.new('RGB',(512,258),(40,90,180)); ImageDraw.Draw(image).line((0,257,511,257),fill=(230,40,70))
            image.save(path,quality=95,subsampling=2)
            result=subprocess.run([str(tracer),'-o',str(trace),'-e','trace=open,openat,close,ioctl','-e','raw=ioctl',
                str(probe),str(path),str(raw)],capture_output=True,text=True,timeout=20)
            if result.returncode!=0 or raw.stat().st_size!=512*258*3: raise ValueError('Synthetic probe failed')
            nodes=device_calls(trace.read_text(),include_commands=True)
            count=nodes.get('/dev/nvhost-nvjpg',{}).get('successful_commands',{}).get('0xc080481a',0)
            diff=metrics(pixels(path),np.frombuffer(raw.read_bytes(),np.uint8).reshape(258,512,3))
            report.update(nodes=nodes,nvjpg_submit_success=count,pixel_metrics=diff)
            if count<1 or diff['max_abs_error']!=0: raise ValueError('Actual submission/exact pixel gate failed')
        report.update(status='hardware_submission_and_partial_mcu_pixels_passed',trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
    except Exception as error:
        report.update(status='failed',error=type(error).__name__); raise
    finally:
        with output.open('x') as file: json.dump(report,file,indent=2,allow_nan=False)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True); p.add_argument('--strace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    print(json.dumps(run(a.probe.resolve(),a.strace.resolve(),a.output.resolve())))
