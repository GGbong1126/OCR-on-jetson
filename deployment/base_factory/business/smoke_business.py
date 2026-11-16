"""Local HTTP smoke; print only technical counts, never recognized document text.

Default: readiness and invalid payload rejection. --image adds real /OCR.
--extract additionally exercises /Extract with real OCR output and may consume one licensed count.
"""
import argparse
import base64
import json
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


def validate_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ('', '/')):
        raise ValueError('Only a local HTTP service base URL is allowed')
    return url.rstrip('/')


def call(base, endpoint, payload=None, timeout=240):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = Request(base+endpoint, data=body, headers={'Content-Type':'application/json'})
    # Never inherit HTTP_PROXY for loopback requests.
    with build_opener(ProxyHandler({})).open(req, timeout=timeout) as response:
        data = response.read(24*1024*1024+1)
        if len(data)>24*1024*1024:
            raise ValueError('Response exceeds smoke limit')
        return response.status, json.loads(data)


def ocr_statistics(response):
    if response.get('isSuccess') is not True or response.get('status') != '200':
        raise ValueError('OCR application returned failure status '+str(response.get('status')))
    pages = response.get('msg',{}).get('pages',[])
    if len(pages)!=1 or not isinstance(pages[0],dict):
        raise ValueError('Expected exactly one OCR page')
    page=pages[0]
    blocks=page.get('blocks',[])
    if not isinstance(blocks,list) or not blocks:
        raise ValueError('OCR smoke produced no blocks; inspect model/output separately')
    width,height=page['width'],page['height']
    for block in blocks:
        if not (0<=block['x1']<=block['x2']<=width and 0<=block['y1']<=block['y2']<=height):
            raise ValueError('OCR block coordinates outside original image')
    return {'page_count':len(pages),'block_count':len(blocks),'width':width,'height':height,
            'text_characters':sum(len(b.get('text','')) for b in blocks),
            'markdown_characters':len(page.get('markdown','')),
            # /Extract reserves the first three lines for header fields.
            'markdown_body_characters':len('\n'.join(page.get('markdown','').splitlines()[3:])),
            'bounds_checked':True}


def extract_statistics(response):
    if response.get('isSuccess') is not True or response.get('status') != '200':
        raise ValueError('Extract application returned failure')
    fields = response.get('msg')
    if not isinstance(fields, dict) or not isinstance(fields.get('NER'), dict):
        raise ValueError('Expected Extract msg and NER objects')
    if not all(isinstance(value, str) for value in fields['NER'].values()):
        raise ValueError('Expected string NER field values')
    def nonempty(value):
        return value is not None and (not isinstance(value, str) or bool(value.strip()))
    return {'returned_field_count':len(fields),  # Legacy name: top-level keys, not entities.
            'returned_top_level_key_count':len(fields),
            'nonempty_top_field_count':sum(nonempty(value) for key, value in fields.items() if key != 'NER'),
            'ner_field_count':len(fields['NER']),
            'nonempty_ner_field_count':sum(nonempty(value) for value in fields['NER'].values()),
            'note':'Counts only, not field accuracy or model-entity counts; one licensed count may have been consumed.'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:5051')
    parser.add_argument('--image',type=Path)
    parser.add_argument('--image-type',choices=('1','2','3'),default='3')
    parser.add_argument('--extract',action='store_true',help='May consume one licensed count; explicit opt-in')
    args=parser.parse_args(argv)
    base=validate_url(args.url)
    if args.extract and not args.image:
        parser.error('--extract requires --image')
    status,health=call(base,'/healthz',timeout=15)
    if status!=200 or health.get('status')!='ready':
        raise RuntimeError('Health check not ready')
    print(json.dumps({'health':health}),flush=True)
    for endpoint in ('/OCR','/Extract'):
        status,result=call(base,endpoint,{},timeout=15)
        if status!=200 or result.get('isSuccess') is not False or result.get('status')!='400':
            raise RuntimeError('Invalid-payload contract failed for '+endpoint)
        print(json.dumps({'endpoint':endpoint,'http_status':status,'application_status':'400','invalid_payload_rejected':True}),flush=True)
    if args.image:
        raw=args.image.read_bytes()
        if len(raw)>16*1024*1024:
            raise ValueError('Image exceeds client smoke limit')
        start=perf_counter()
        status,result=call(base,'/OCR',{'img':base64.b64encode(raw).decode('ascii'),'imageType':args.image_type})
        if status!=200:
            raise RuntimeError('Unexpected OCR HTTP status')
        print(json.dumps({'endpoint':'/OCR','image_type':args.image_type,'single_request_wall_s':perf_counter()-start,
                          **ocr_statistics(result)}),flush=True)
        if args.extract:
            start=perf_counter()
            status,fields=call(base,'/Extract',{'ocrResult':result['msg'],
                'topFields':[{'name':'title','type':'String','description':'title'}],
                'nerFields':[{'name':'姓名','description':'姓名'}]})
            if status!=200:
                raise RuntimeError('Unexpected Extract HTTP status')
            print(json.dumps({'endpoint':'/Extract','single_request_wall_s':perf_counter()-start,
                              **extract_statistics(fields)}),flush=True)
    detail = 'No real OCR/Extract requested.'
    if args.image:
        detail = ('OCR imageType=3 skips enhancement.' if args.image_type == '3' else
                  f'OCR imageType={args.image_type} requests enhancement; execution not independently observed.')
    print('HTTP smoke passed; not GT accuracy, not a benchmark. '+detail)


if __name__=='__main__':
    main()
