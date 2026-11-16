"""Deduplicate our staged immutable model copies, preserving every path and byte.

Only four known original ONNX models, checked against the release manifest.
An atomic replacement with a verified hardlink reclaims space on the same eMMC.
No recursive removal, user pictures, NVMe, system files, or model content edits.
"""
import json
import os
from pathlib import Path
from trt82_backend import sha256


def main():
    root=Path('/home/xchq/work/autoocr')
    release=root/'releases/jetson-r1'
    manifest=json.loads((release/'jetson_source_manifest.json').read_text())
    report=root/'reports/dedupe_staged_originals_r1.json'
    if report.exists():raise ValueError('Already audited; do not repeat')
    rows=[]
    for name in ('ocr_det.onnx','ocr_rec.onnx','layout_det.onnx','cell_det.onnx'):
        source=release/'pth/onnx_rec'/name
        target=root/'staging/jp461-r1/models'/name
        expected=manifest['files']['pth/onnx_rec/'+name]['sha256']
        for path in (source,target):
            if path.is_symlink() or not path.resolve().is_relative_to(root) or sha256(path)!=expected:
                raise ValueError('Unsafe or changed model')
        if source.stat().st_dev!=target.stat().st_dev:raise ValueError('Different filesystems')
        if source.stat().st_ino==target.stat().st_ino:continue
        temp=target.with_suffix('.dedupe-new')
        if temp.exists() or temp.is_symlink():raise ValueError('Temporary path exists')
        size=target.stat().st_size
        os.link(source,temp)
        if sha256(temp)!=expected:raise ValueError('Hardlink hash mismatch')
        os.replace(temp,target)
        rows.append({'name':name,'sha256':expected,'duplicate_bytes_reclaimed':size,
                     'paths_retained':True,'inode_equal':source.stat().st_ino==target.stat().st_ino})
    with report.open('x') as f:json.dump({'status':'deduplicated_no_content_change','rows':rows},f,indent=2)
    print(json.dumps({'models':len(rows),'bytes_reclaimed':sum(r['duplicate_bytes_reclaimed'] for r in rows)}))


if __name__=='__main__':main()
