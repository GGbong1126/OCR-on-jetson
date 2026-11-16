"""Local batch OCR client. Prints technical statistics only; no Extract or retries."""
import argparse
import base64
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, ProxyHandler, build_opener

business=Path(__file__).resolve().parents[3]/'business'
if not (business/'smoke_business.py').is_file():
    business=Path('/home/xchq/work/autoocr/releases/jetson-r2/edge_deploy/business')
sys.path.insert(0,str(business))
from smoke_business import validate_url, ocr_statistics


def payload(paths, kind):
    if not 1<=len(paths)<=4 or type(kind) is not int or kind not in (1,2,3): raise ValueError('One to four images and type 1/2/3 required')
    sizes=[p.stat().st_size for p in paths]
    if any(not 1<=n<=16*1024**2 for n in sizes) or sum(4*((n+2)//3) for n in sizes)+4096>24*1024**2:
        raise ValueError('Input exceeds batch HTTP byte limit; use smaller batches')
    images=[]
    for i,(path,size) in enumerate(zip(paths,sizes)):
        with path.open('rb') as file: data=file.read(16*1024**2+1)
        if len(data)!=size: raise ValueError('Input changed while reading')
        images.append(dict(imageId='image'+str(i),imageType=kind,imageBase64=base64.b64encode(data).decode()))
    return json.dumps(dict(images=images)).encode()


def statistics(value, count):
    rows=value.get('results',[])
    if len(rows)!=count or [r['image_id'] for r in rows]!=['image'+str(i) for i in range(count)]:
        raise ValueError('Batch result IDs/order differ')
    summary=[]
    for row in rows:
        item=dict(image_id=row['image_id'],status=row['status'])
        if row['status']=='ok': item.update(ocr_statistics(dict(isSuccess=True,status='200',msg=row['data'])))
        else: item['code']=row['code']
        summary.append(item)
    good=all(r['status']=='ok' for r in rows)
    if value.get('status')!=('ok' if good else 'partial'): raise ValueError('Batch status differs from item statuses')
    return dict(status=value['status'],rows=summary,execution=value.get('execution',{}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:5052'); p.add_argument('--images',type=Path,nargs='+',required=True)
    p.add_argument('--image-type',type=int,choices=(1,2,3),default=3)
    args=p.parse_args(); url=validate_url(args.url); body=payload(args.images,args.image_type)
    request=Request(url+'/OCRBatch',data=body,headers={'Content-Type':'application/json'})
    begin=time.monotonic()
    with build_opener(ProxyHandler({})).open(request,timeout=320) as response:
        data=response.read(64*1024**2+1)
        if len(data)>64*1024**2: raise ValueError('Client response limit exceeded')
        value=json.loads(data)
    summary=statistics(value,len(args.images)); summary['client_request_wall_s']=time.monotonic()-begin
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if summary['status']!='ok': raise SystemExit(1)


if __name__=='__main__': main()
