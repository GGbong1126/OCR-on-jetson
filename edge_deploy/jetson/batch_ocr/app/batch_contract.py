"""Local batch front protocol. No paths/pickle/base64 in the IPC envelope."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'ipc'))
import ipc_wire as wire
from stage_contract import integer, token, need, validate_result

def validate(raw, size):
    m=wire.decode_metadata(raw); need(m.get('schema')=='ocr.batch/1')
    if m.get('op')=='batch.reply':
        need(set(m)=={'schema','op','status','count'} and m['status'] in ('ok','error'))
        need(integer(m['count'],0,4) and integer(size,1,32*1024**2)); return m
    need(integer(m.get('remaining_ms'),1,600000))
    if m.get('op')=='batch.health': need(set(m)=={'schema','op','remaining_ms'} and size==0); return m
    need(m.get('op')=='batch.submit' and set(m)=={'schema','op','remaining_ms','images'})
    rows=m['images']; need(type(rows) is list and 1<=len(rows)<=4)
    seen=set(); total=0
    for row in rows:
        need(type(row) is dict and set(row)=={'image_id','image_type','encoded_bytes'})
        need(token(row['image_id']) and row['image_id'] not in seen); seen.add(row['image_id'])
        need(integer(row['image_type'],1,3) and integer(row['encoded_bytes'],1,16*1024**2))
        total+=row['encoded_bytes']
    need(total==size and total<=64*1024**2); return m

def result(payload):
    return validate_result(payload,max_bytes=32*1024**2)
