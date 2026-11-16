"""Stage metadata validation only: no handlers, models, paths or authorization."""
import json
import math
from pathlib import Path
import re

import ipc_wire as wire

LIMITS_PATH = Path(__file__).resolve().parent.parent / 'scheduler/limits.json'
LIMITS = json.loads(LIMITS_PATH.read_text(encoding='utf-8'))
COMMON = {'schema','op','session','remaining_ms'}
BINDINGS = {'det':{'x'}, 'rec':{'x'}, 'layout':{'image','im_shape','scale_factor'},
            'cell':{'image','im_shape','scale_factor'}, 'yolo':{'images'}, 'enhance':{'images'}}
CONTROL_OPS = {'image.submit','model.run','model.resume','image.collect','image.cancel'}
ERROR_CODES = {'BUSY','LIMIT','STATE','SESSION','TIMEOUT','CANCELLED','GPU_FAILED','DISCONNECTED'}


def need(condition):
    if not condition:
        raise wire.ProtocolError('Invalid OCR stage contract')


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def token(value):
    return type(value) is str and re.fullmatch(r'[A-Za-z0-9_.-]{1,64}',value) is not None


def validate_limits(cfg):
    need(set(cfg)==set(LIMITS) and cfg['schema']=='ocr.runtime/1')
    need(type(cfg['det_batch_enabled']) is bool)
    for key,value in cfg.items():
        if key not in ('schema','det_batch_enabled'): need(integer(value,1,2**31-1))
    need(cfg['gpu_dispatchers']==cfg['backend_gpu_workers']==1)
    need(cfg['input_workers']==cfg['result_workers']==cfg['backend_cpu_workers']==2)
    need(cfg['max_active_images']==2 and cfg['max_inflight_images']==cfg['queue_items']==4)
    need(cfg['active_budget_bytes']==cfg['page_reservation_bytes']*cfg['max_active_images'])
    need(cfg['explicit_budget_bytes']==sum(cfg[k] for k in (
        'active_budget_bytes','ingress_budget_bytes','transport_budget_bytes','result_budget_bytes')))
    need(cfg['explicit_budget_bytes']<=768*1024**2)
    need(cfg['input_queue_bytes']<=cfg['ingress_budget_bytes'])
    need(cfg['result_queue_bytes']<=cfg['result_budget_bytes'])
    need(cfg['gpu_queue_bytes']<=cfg['active_budget_bytes'])
    need(cfg['max_encoded_bytes']<=cfg['input_queue_bytes'] and cfg['max_result_bytes']<=cfg['result_queue_bytes'])
    need(cfg['max_dimension']<=8192 and cfg['max_pixels']<=16_000_000 and cfg['max_tensor_bytes']<=wire.MAX_PAYLOAD)
    need(2304<=cfg['max_rec_cpu_width']<=4096)
    need(cfg['control_timeout_ms']<=cfg['model_timeout_ms']<=cfg['image_timeout_ms'])
    need(cfg['det_batch_wait_ms']<=cfg['enqueue_timeout_ms']<=cfg['image_timeout_ms'])
    return cfg


validate_limits(LIMITS)


def _ids(values, maximum=2):
    need(type(values) is list and 1<=len(values)<=maximum)
    need(all(integer(v,1,wire.MAX_ID) for v in values) and len(set(values))==len(values))


def _ready(meta):
    model=meta['model']; need(type(model) is str and model in BINDINGS)
    need(token(meta['image_id']) and integer(meta['image_type'],1,3) and integer(meta['call_id'],1,wire.MAX_ID))
    inputs=meta['inputs']; need(type(inputs) is list and 1<=len(inputs)<=3)
    names=[]; handles=[]; total=0
    for value in inputs:
        need(type(value) is dict and set(value)=={'name','handle','dtype','shape','bytes'})
        need(type(value['name']) is str and value['name'] in BINDINGS[model])
        need(integer(value['handle'],1,wire.MAX_ID) and value['dtype']=='float32')
        shape=value['shape']; need(type(shape) is list and 1<=len(shape)<=4)
        need(all(integer(n,1,8192) for n in shape))
        size=4
        for n in shape:
            need(size<=LIMITS['max_tensor_bytes']//n); size*=n
        need(integer(value['bytes'],1,LIMITS['max_tensor_bytes']) and value['bytes']==size)
        if value['name'] in ('im_shape','scale_factor'): need(shape==[1,2])
        elif model in ('layout','cell'): need(shape==[1,3,800,800] if model=='layout' else shape==[1,3,640,640])
        elif model=='enhance': need(shape==[1,3,512,512])
        elif model=='rec': need(len(shape)==4 and shape[:3]==[1,3,48] and 32<=shape[3]<=3200)
        else:
            need(len(shape)==4 and shape[:2]==[1,3])
            if model=='det': need(all(32<=d<=960 and d%32==0 for d in shape[2:]))
            else: need(all(64<=d<=640 and d%32==0 for d in shape[2:]))
        names.append(value['name']); handles.append(value['handle']); total+=size
    need(set(names)==BINDINGS[model] and len(set(names))==len(names) and len(set(handles))==len(handles))
    need(total<=LIMITS['max_tensor_bytes'])


def validate_stage(raw, payload_size):
    need(integer(payload_size,0,wire.MAX_PAYLOAD))
    meta=wire.decode_metadata(raw)
    need(meta.get('schema')=='ocr.stage/1' and type(meta.get('op')) is str)
    need(type(meta.get('session')) is str and re.fullmatch(r'[0-9a-f]{32}',meta['session']) is not None)
    need(integer(meta.get('remaining_ms'),1,LIMITS['image_timeout_ms']))
    op=meta['op']
    if op=='session.ready':
        need(set(meta)==COMMON and payload_size==0)
    elif op in ('channel.bind','channel.bound'):
        need(set(meta)==COMMON|{'channel'} and payload_size==0)
        need(type(meta['channel']) is str and meta['channel'] in ('input','gpu','result','events'))
    elif op=='rpc.reply':
        need(set(meta)==COMMON|{'for_op','status','code'} and payload_size==0)
        need(type(meta['for_op']) is str and meta['for_op'] in CONTROL_OPS)
        need(meta['status'] in ('ok','error') and type(meta['code']) is str)
        need(meta['code']=='OK' if meta['status']=='ok' else meta['code'] in ERROR_CODES)
    elif op=='image.submit':
        need(token(meta.get('image_id')) and integer(meta.get('image_type'),1,3))
        if meta.get('format')=='encoded':
            need(set(meta)==COMMON|{'image_id','image_type','format'})
            need(1<=payload_size<=LIMITS['max_encoded_bytes'])
        else:
            need(set(meta)==COMMON|{'image_id','image_type','format','shape','stride'} and meta.get('format')=='bgr8')
            shape=meta['shape']; need(type(shape) is list and len(shape)==3)
            need(all(type(n) is int for n in shape) and shape[2]==3)
            need(all(1<=n<=LIMITS['max_dimension'] for n in shape[:2]) and shape[0]*shape[1]<=LIMITS['max_pixels'])
            need(type(meta['stride']) is int and meta['stride']==shape[1]*3 and payload_size==shape[0]*meta['stride'])
    elif op=='model.ready':
        need(set(meta)==COMMON|{'image_id','image_type','call_id','model','inputs'} and payload_size==0)
        _ready(meta)
    elif op=='model.run':
        need(set(meta)==COMMON|{'call_ids'} and payload_size==0); _ids(meta['call_ids'])
    elif op=='model.resume':
        need(set(meta)==COMMON|{'call_id'} and payload_size==0 and integer(meta['call_id'],1,wire.MAX_ID))
    elif op in ('image.collect','image.cancel'):
        need(set(meta)==COMMON|{'image_id'} and payload_size==0 and token(meta['image_id']))
    elif op=='model.done':
        need(set(meta)==COMMON|{'call_ids','output_bytes'} and payload_size==0); _ids(meta['call_ids'])
        sizes=meta['output_bytes']; need(type(sizes) is list and len(sizes)==len(meta['call_ids']))
        need(all(integer(n,0,LIMITS['max_tensor_bytes']) for n in sizes))
    elif op=='model.failed':
        need(set(meta)==COMMON|{'call_ids','code'} and payload_size==0)
        _ids(meta['call_ids']); need(meta['code']=='GPU_FAILED')
    elif op in ('image.result','image.done'):
        fields=COMMON|{'image_id','status'}
        need(set(meta)==(fields if meta.get('status')=='ok' else fields|{'code'}) and token(meta['image_id']))
        need(meta['status'] in ('ok','error','cancelled'))
        need((1<=payload_size<=LIMITS['max_result_bytes'] if meta['status']=='ok' else payload_size==0)
             if op=='image.result' else payload_size==0)
        if meta['status']!='ok':
            need(type(meta['code']) is str and meta['code'] in ERROR_CODES)
            need((meta['status']=='cancelled')==(meta['code']=='CANCELLED'))
    else:
        raise wire.ProtocolError('Unknown OCR stage operation')
    return meta


def validate_run(raw, ready, *, allow_det_batch=False):
    """Check metadata references; actual registry ownership/completion is a later handler."""
    meta=validate_stage(raw,0); need(meta['op']=='model.run')
    calls=[]
    for call_id in meta['call_ids']:
        need(call_id in ready)
        row=ready[call_id]
        validate_stage(json.dumps(row,separators=(',',':')).encode(),0)
        need(row['op']=='model.ready')
        need(row['session']==meta['session'] and row['call_id']==call_id)
        calls.append(row)
    if len(calls)==2:
        need(allow_det_batch is True and all(row['model']=='det' for row in calls))
        need(calls[0]['image_id']!=calls[1]['image_id'] and calls[0]['image_type']==calls[1]['image_type'])
        signature=lambda row:[(v['name'],v['dtype'],v['shape'],v['bytes']) for v in row['inputs']]
        need(signature(calls[0])==signature(calls[1]))
        need({v['handle'] for v in calls[0]['inputs']}.isdisjoint(v['handle'] for v in calls[1]['inputs']))
    return meta


def validate_result(payload, *, max_bytes=LIMITS['max_result_bytes']):
    """Generic result integrity only, not the original OCR response schema."""
    need(integer(max_bytes,1,wire.MAX_PAYLOAD))
    need(type(payload) is bytes and 1<=len(payload)<=max_bytes)
    def finite_float(text):
        value=float(text); need(math.isfinite(value)); return value
    try:
        value=json.loads(payload.decode('utf-8'),object_pairs_hook=wire._object,
                         parse_constant=wire._constant,parse_float=finite_float)
        need(type(value) is dict)
        stack=[(value,0)]
        while stack:
            item,depth=stack.pop(); need(depth<=64)
            if type(item) is dict:
                for key,child in item.items():
                    key.encode('utf-8'); stack.append((child,depth+1))
            elif type(item) is list: stack.extend((child,depth+1) for child in item)
            elif type(item) is str: item.encode('utf-8')
        return value
    except (ValueError,UnicodeError,RecursionError) as error:
        raise wire.ProtocolError('Invalid result JSON object') from error
