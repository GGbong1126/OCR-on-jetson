"""Probe-only IPC v1. Standard library, synthetic data, no OCR/model imports."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import socket
import stat
import struct
import time

HEADER = struct.Struct('!4sHHIIQ')
MAX_META = 4096
MAX_PAYLOAD = 64 * 1024**2
MAX_ID = 2**63 - 1


class ProtocolError(ValueError):
    pass


class PeerClosed(ConnectionError):
    pass


@dataclass(frozen=True)
class Frame:
    request_id: int
    metadata: bytes
    payload: bytes


def header(request_id, meta_size, payload_size):
    if (type(request_id) is not int or type(meta_size) is not int or type(payload_size) is not int
            or not 1 <= request_id <= MAX_ID
            or not 1 <= meta_size <= MAX_META or not 0 <= payload_size <= MAX_PAYLOAD):
        raise ProtocolError('Invalid frame bounds')
    return HEADER.pack(b'AOCR', 1, 0, meta_size, payload_size, request_id)


def parse_header(raw):
    if len(raw) != HEADER.size:
        raise ProtocolError('Invalid header size')
    magic, version, flags, meta_size, payload_size, request_id = HEADER.unpack(raw)
    if (magic, version, flags) != (b'AOCR', 1, 0):
        raise ProtocolError('Unknown protocol/version/flags')
    header(request_id, meta_size, payload_size)
    return request_id, meta_size, payload_size


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError('Duplicate JSON key')
        result[key] = value
    return result


def _constant(value):
    raise ProtocolError('Nonfinite JSON constant')


def decode_metadata(raw):
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= MAX_META:
        raise ProtocolError('Invalid metadata size/type')
    try:
        meta = json.loads(raw.decode('utf-8'), object_pairs_hook=_object, parse_constant=_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ProtocolError('Invalid JSON metadata') from exc
    if type(meta) is not dict:
        raise ProtocolError('Metadata object required')
    return meta


def validate_probe(raw, payload_size):
    meta = decode_metadata(raw)
    if set(meta) != {'op', 'image_id', 'dtype', 'layout', 'shape', 'stride'}:
        raise ProtocolError('Unknown metadata fields')
    if (meta['op'] != 'probe' or meta['dtype'] != 'uint8' or meta['layout'] != 'HWC'
            or type(meta['image_id']) is not str or not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', meta['image_id'])):
        raise ProtocolError('Unknown operation/format/ID')
    shape = meta['shape']
    if (type(shape) is not list or len(shape) != 3 or any(type(n) is not int for n in shape)
            or shape[2] != 3 or not 1 <= shape[0] <= 8192 or not 1 <= shape[1] <= 8192
            or shape[0] * shape[1] > 16_000_000):
        raise ProtocolError('Invalid image shape')
    if (type(meta['stride']) is not int or meta['stride'] != shape[1] * 3
            or payload_size != shape[0] * meta['stride'] or payload_size > MAX_PAYLOAD):
        raise ProtocolError('Invalid stride/payload size')
    return meta


def _remaining(deadline):
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError('IPC deadline exceeded')
    return left


def _exact(sock, count, deadline):
    data = bytearray(count)
    view = memoryview(data)
    offset = 0
    while offset < count:
        sock.settimeout(_remaining(deadline))
        try:
            size = sock.recv_into(view[offset:])
        except InterruptedError:
            continue
        if not size:
            raise PeerClosed('IPC peer closed before frame completed')
        offset += size
    return bytes(data)


def receive_frame(sock, deadline, *, validator=validate_probe):
    original_timeout = sock.gettimeout()
    try:
        request_id, meta_size, payload_size = parse_header(_exact(sock, HEADER.size, deadline))
        metadata = _exact(sock, meta_size, deadline)
        validator(metadata, payload_size)
        payload = _exact(sock, payload_size, deadline)
        _remaining(deadline)
        return Frame(request_id, metadata, payload)
    finally:
        sock.settimeout(original_timeout)


def send_frame(sock, frame, deadline, *, validator=validate_probe):
    if not isinstance(frame.payload, bytes):
        raise ProtocolError('Owned immutable payload bytes required')
    prefix = header(frame.request_id, len(frame.metadata), len(frame.payload))
    validator(frame.metadata, len(frame.payload))
    original_timeout = sock.gettimeout()
    try:
        for data in (prefix, frame.metadata, frame.payload):
            view = memoryview(data)
            while view:
                sock.settimeout(_remaining(deadline))
                try:
                    count = sock.send(view)
                except InterruptedError:
                    continue
                if not count:
                    raise PeerClosed('IPC peer closed during send')
                view = view[count:]
        _remaining(deadline)
    finally:
        sock.settimeout(original_timeout)


def check_peer(sock):
    pid, uid, gid = struct.unpack('iII', sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    if uid != os.geteuid():
        raise PermissionError('IPC peer UID differs')
    return pid, uid, gid


@contextmanager
def private_listener(path):
    path = Path(path)
    parent = path.parent.lstat()
    if (not path.is_absolute() or path.parent.resolve() != path.parent
            or not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.geteuid()
            or stat.S_IMODE(parent.st_mode) & 0o077 or len(os.fsencode(path)) >= 108):
        raise PermissionError('Private owned directory and short absolute socket path required')
    if os.path.lexists(path):
        raise FileExistsError('Existing IPC path must be preserved')
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    owned = None
    try:
        listener.bind(str(path))
        owned = path.lstat()
        path.chmod(0o600)
        listener.listen(1)
        yield listener
    finally:
        try:
            try:
                current = path.lstat()
            except FileNotFoundError:
                current = None
            if (owned is not None and current is not None and stat.S_ISSOCK(current.st_mode)
                    and (owned.st_dev, owned.st_ino) == (current.st_dev, current.st_ino)):
                path.unlink()
        finally:
            listener.close()
