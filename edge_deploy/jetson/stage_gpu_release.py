"""Stage jetson-r2 only after both real-image module gates pass. Never start service.

Hardlink verified immutable originals/engines; write NEW adapter/config files.
No r1 modification, license access, counter consumption, or systemd mutation.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
from trt82_backend import sha256
from source_bundle import safe_name
from stage_service import FILES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/xchq/work/autoocr'))
    p.add_argument('--adapters',type=Path,required=True)
    p.add_argument('--plugin',type=Path,required=True)
    p.add_argument('--layout-engine',type=Path,required=True)
    p.add_argument('--cell-engine',type=Path,required=True)
    p.add_argument('--reports',type=Path,nargs='+',required=True)
    a=p.parse_args()
    root=a.root.resolve();old=root/'releases/jetson-r1';new=root/'releases/jetson-r2'
    a.reports=[v.resolve() for v in a.reports]
    a.plugin=a.plugin.resolve();a.cell_engine=a.cell_engine.resolve();a.layout_engine=a.layout_engine.resolve()
    a.adapters=a.adapters.resolve()
    if any(not v.is_relative_to(root) for v in [*a.reports,a.plugin,a.cell_engine,a.layout_engine,a.adapters]):
        raise ValueError('All stage inputs must stay in the project')
    if root!=Path('/home/xchq/work/autoocr') or new.exists() or new.is_symlink():
        raise ValueError('Unexpected/existing target')
    for parent in (root,*root.parents,new.parent):
        if parent.is_symlink():raise ValueError('Linked project root')
    manifest=json.loads((old/'jetson_source_manifest.json').read_text())
    config=json.loads((old/'edge_deploy/jetson/deployment.json').read_text())
    for name,row in manifest['files'].items():
        if not safe_name(name) or sha256(old/name)!=row['sha256']:raise ValueError('Original changed')
    accepted={}
    image_sets=[]
    for path in a.reports:
        r=json.loads(path.read_text())
        if (r['status']!='task_consistent_not_gt_validated' or r['summary']['image_count']!=13
                or r.get('input_mode')!='business'):
            raise ValueError('Requires all 13-image task gates')
        if r['plugin_sha256']!=sha256(a.plugin):raise ValueError('Plugin differs from tested binary')
        image_sets.append(sorted(v['image_sha256'] for v in r['images']))
        for role,digest in r['engine_sha256'].items():
            if role in accepted and accepted[role]!=digest:raise ValueError('Conflicting tested engine')
            accepted[role]=digest
        if not all(row['task']['passed'] for row in r['images']+r['cell_cases'] if 'task' in row):
            raise ValueError('Failed per-case gate')
        if 'cell' in r['engine_sha256'] and r['summary']['actual_table_crops']<1:
            raise ValueError('Actual table-crop coverage missing')
        for role,digest in r['source_hashes'].items():
            if sha256(old/('pth/onnx_rec/'+role+'_det.onnx'))!=digest:
                raise ValueError('Reference model differs')
    if any(v!=image_sets[0] for v in image_sets) or set(accepted)!={'cell','layout'}:
        raise ValueError('Incomplete or different image coverage')
    for role,engine in (('cell',a.cell_engine),('layout',a.layout_engine)):
        if sha256(engine)!=accepted[role]:raise ValueError('Engine differs from tested binary')
    for role,row in config['engines'].items():
        if sha256(old/row['path'])!=row['sha256']:raise ValueError('r1 engine changed')
    new.mkdir(mode=0o700)
    def link(source,relative):
        target=new/relative
        if not safe_name(relative) or source.is_symlink():raise ValueError('Unsafe link source/target')
        target.parent.mkdir(parents=True,exist_ok=True)
        os.link(source,target)
    for name in list(manifest['files'])+['jetson_source_manifest.json']:
        link(old/name,name)
    for row in config['engines'].values():link(old/row['path'],row['path'])
    plug='pth/tensorrt/libautoocr_grid.so';link(a.plugin,plug)
    for role,engine in (('cell',a.cell_engine),('layout',a.layout_engine)):
        relative='pth/tensorrt/'+engine.name;link(engine,relative)
        config['engines'][role]={'path':relative,'sha256':sha256(engine),
            'source_sha256':manifest['files']['pth/onnx_rec/'+role+'_det.onnx']['sha256'],
            'plugins':[{'path':plug,'sha256':sha256(a.plugin)}]}
    config['cpu_modules']={}
    config['status']='seven_gpu_models_staged_not_activated_or_http_accepted'
    config['regression_reports']={str(p.relative_to(root)):sha256(p) for p in a.reports}
    dest=new/'edge_deploy/jetson';dest.mkdir(mode=0o700)
    for name in FILES:
        text=(a.adapters/name).read_text(encoding='utf-8')
        if name=='autoocr-jetson.service':text=text.replace('/releases/jetson-r1/','/releases/jetson-r2/')
        with (dest/name).open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    config['adapter_sha256']={name:sha256(dest/name) for name in FILES}
    with (dest/'deployment.json').open('x') as f:json.dump(config,f,indent=2)
    print(json.dumps({'status':config['status'],'release':str(new),'models':sorted(config['engines'])},indent=2))


if __name__=='__main__':main()
