"""Write offline acceptance evidence for r2; never activate or start HTTP."""
import argparse
import json
from pathlib import Path
from trt82_backend import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/xchq/work/autoocr'))
    p.add_argument('--type3',type=Path,required=True)
    p.add_argument('--type1',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    if root!=Path('/home/xchq/work/autoocr'):raise ValueError('Unexpected root')
    release=root/'releases/jetson-r2'; configpath=release/'edge_deploy/jetson/deployment.json'
    config=json.loads(configpath.read_text());engines=config['engines']
    if set(engines)!={'det','rec','layout','cell','yolo','ner','enhance'} or config['cpu_modules']:
        raise ValueError('Expected seven default GPU backends')
    artifact=release/'edge_deploy/jetson/ACCEPTANCE.json'
    if artifact.exists():raise ValueError('Acceptance exists')
    rows=[]
    for path,image_type,count in ((a.type3.resolve(),3,13),(a.type1.resolve(),1,3)):
        if not path.is_relative_to(root):raise ValueError('Report outside project')
        r=json.loads(path.read_text())
        if (r['status']!='offline_pipeline_consistent_not_http_or_gt_validated'
                or r['image_type']!=image_type or len(r['rows'])!=count
                or not all(v['passed'] for v in r['rows'])
                or r['deployment_sha256']!=sha256(configpath)
                or r['engine_sha256']!={k:v['sha256'] for k,v in engines.items()}):
            raise ValueError('Pipeline acceptance failed')
        for role,v in r['gpu_execution'].items():
            if (v['backend']!='tensorrt82_cuda' or v['errors'] or v['calls']<1
                    or v.get('cpu_fallback',{}).get('errors',0)):
                raise ValueError('Backend unhealthy')
        if image_type==1 and r['gpu_execution']['enhance']['calls']<=r['warmup']['enhance']['calls']:
            raise ValueError('Actual enhancement was not exercised')
        rows.append({'report':str(path.relative_to(root)),'sha256':sha256(path),
                     'image_type':image_type,'images':count,'all_text_fields_equal':True,
                     'gpu_execution':r['gpu_execution']})
    for role,v in engines.items():
        if sha256(release/v['path'])!=v['sha256']:raise ValueError('Engine changed')
        for plugin in v.get('plugins',[]):
            if sha256(release/plugin['path'])!=plugin['sha256']:raise ValueError('Plugin changed')
    result={'status':'seven_default_gpu_models_offline_accepted',
            'deployment_sha256':sha256(configpath),'pipeline_reports':rows,
            'http_started':False,'activation_performed':False,'license_or_counter_rules_changed':False,
            'note':'CPU pre/post and original-tensor out-of-profile OCR fallback remain. No GT accuracy or steady-state benchmark claimed.'}
    with artifact.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'status':result['status'],'reports':len(rows),'http_started':False}))


if __name__=='__main__':main()
