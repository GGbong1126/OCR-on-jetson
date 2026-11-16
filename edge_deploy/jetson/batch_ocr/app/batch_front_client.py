"""Bounded private-socket batch client shared by validation and the HTTP front."""
import socket
import time

import batch_contract as contract
from stage_session import encoded, require
import ipc_wire as wire


def exchange(path, images=None, *, remaining_ms=300000):
    metadata=dict(schema='ocr.batch/1',op='batch.health',remaining_ms=remaining_ms)
    body=b''
    if images is not None:
        metadata.update(op='batch.submit',images=[dict(image_id=name,image_type=kind,encoded_bytes=len(data)) for name,kind,data in images])
        size=sum(len(data) for _,_,data in images); contract.validate(encoded(metadata),size)
        body=b''.join(data for _,_,data in images)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
        deadline=time.monotonic()+remaining_ms/1000
        sock.connect(str(path)); wire.check_peer(sock)
        wire.send_frame(sock,wire.Frame(1,encoded(metadata),body),deadline,validator=contract.validate)
        response=wire.receive_frame(sock,deadline,validator=contract.validate)
        meta=contract.validate(response.metadata,len(response.payload))
        require(response.request_id==1 and meta['status']=='ok')
        result=contract.result(response.payload)
        if images is not None:
            require(meta['count']==len(images) and len(result['results'])==len(images))
            require([v['image_id'] for v in result['results']]==[row[0] for row in images])
            # Confirm C++ has completed its send/cleanup before HTTP serializes another full copy.
            released=exchange(path,remaining_ms=5000)['execution']
            require(released['result_reserved']==0)
        return result
