import json
import os
from pathlib import Path
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import ipc_wire as w


def example(request_id=101):
    meta = dict(dtype='uint8', image_id='synthetic_1', layout='HWC', op='probe', shape=[2,4,3], stride=12)
    return w.Frame(request_id, json.dumps(meta, separators=(',', ':'), sort_keys=True).encode(), bytes(range(24)))


def packed(frame):
    return w.header(frame.request_id, len(frame.metadata), len(frame.payload)) + frame.metadata + frame.payload


class WireTests(unittest.TestCase):
    def test_header_network_endian_and_bounds(self):
        raw = w.header(0x0102030405060708, 17, 31)
        self.assertEqual(raw.hex(), '414f435200010000000000110000001f0102030405060708')
        self.assertEqual(w.parse_header(raw), (0x0102030405060708,17,31))
        for values in ((0,1,1), (w.MAX_ID+1,1,1), (True,1,1), (1,True,1), (1,0,1),
                       (1,w.MAX_META+1,1), (1,1,w.MAX_PAYLOAD+1), (1,1,-1)):
            with self.subTest(values=values), self.assertRaises(w.ProtocolError):
                w.header(*values)
        for offset in (0,4,6):
            bad = bytearray(raw); bad[offset] ^= 1
            with self.assertRaises(w.ProtocolError): w.parse_header(bytes(bad))

    def test_strict_metadata(self):
        original = json.loads(example().metadata)
        variants = [dict(original, op='Extract'), dict(original, dtype='float32'),
                    dict(original, shape=[True,4,3]), dict(original, shape=[2,4,4]),
                    dict(original, shape=[8192,8192,3]), dict(original, stride=13),
                    dict(original, image_id='../image'), dict(original, path='/tmp/input')]
        for meta in variants:
            with self.subTest(meta=meta), self.assertRaises(w.ProtocolError):
                w.validate_probe(json.dumps(meta).encode(), 24)
        for raw in (b'\xff', b'[]', b'{}', b'{"op":"probe","op":"probe"}', b'{"shape":NaN}',
                    b'{"stride":Infinity}', b'['*2000 + b']'*2000):
            with self.assertRaises(w.ProtocolError): w.validate_probe(raw, 24)
        self.assertEqual(w.validate_probe(example().metadata,24)['shape'],[2,4,3])

    def test_fragmented_reads(self):
        data = packed(example()) + packed(example(102))
        sock = Mock(); sock.gettimeout.return_value = None
        pending = bytearray(data)
        def recv(view):
            count = min(3,len(view),len(pending))
            view[:count] = pending[:count]; del pending[:count]; return count
        sock.recv_into.side_effect = recv
        self.assertEqual([w.receive_frame(sock,time.monotonic()+1).request_id for _ in range(2)],[101,102])
        self.assertFalse(pending)

    @unittest.skipUnless(sys.platform=='linux','Native Unix socketpair required')
    def test_coalesced_frames(self):
        a, b = socket.socketpair()
        b.settimeout(2)
        received = []
        errors = []
        def receiver():
            try:
                for _ in range(2): received.append(w.receive_frame(a, time.monotonic()+2))
            except BaseException as exc: errors.append(exc)
        worker = threading.Thread(target=receiver)
        try:
            worker.start()
            data = packed(example()) + packed(example(102))
            b.sendall(data)
            worker.join(3)
            self.assertFalse(worker.is_alive()); self.assertEqual(errors,[])
            self.assertEqual([f.request_id for f in received],[101,102])
            self.assertEqual(received[0].payload,bytes(range(24)))
            self.assertIsNone(a.gettimeout())
        finally:
            a.close(); b.close(); worker.join(3)

    def test_short_writes_and_interrupts(self):
        sock = Mock(); sock.gettimeout.return_value = None
        with self.assertRaises(w.ProtocolError):
            w.send_frame(sock,w.Frame(1,example().metadata,bytearray(24)),time.monotonic()+1)
        output = bytearray(); interrupted = [True]
        def send(view):
            if interrupted[0]: interrupted[0] = False; raise InterruptedError()
            count = min(3,len(view)); output.extend(view[:count]); return count
        sock.send.side_effect = send
        w.send_frame(sock,example(),time.monotonic()+1)
        self.assertEqual(bytes(output),packed(example()))
        self.assertEqual(sock.settimeout.call_args.args,(None,))

    def test_deadline_not_reset_by_partial_progress(self):
        sock = Mock(); sock.gettimeout.return_value = None
        def recv(view): view[0] = 1; return 1
        sock.recv_into.side_effect = recv
        with patch.object(w.time,'monotonic',side_effect=[0,0.4,0.8,1.2]):
            with self.assertRaises(TimeoutError): w.receive_frame(sock,1)
        self.assertEqual(sock.recv_into.call_count,3)

    @unittest.skipUnless(sys.platform=='linux','Native Unix socketpair required')
    def test_timeout_and_closed_peer(self):
        a,b = socket.socketpair()
        try:
            start = time.monotonic()
            with self.assertRaises(TimeoutError): w.receive_frame(a,start+0.03)
            self.assertLess(time.monotonic()-start,1)
            b.close()
            with self.assertRaises(w.PeerClosed): w.receive_frame(a,time.monotonic()+1)
            with self.assertRaises((BrokenPipeError,ConnectionError,OSError)):
                w.send_frame(a,example(),time.monotonic()+1)
        finally: a.close(); b.close()

    @unittest.skipUnless(sys.platform=='linux','Native Unix socketpair required')
    def test_truncation_at_each_frame_component(self):
        frame = example(); data = packed(frame)
        for count in (1,23,25,24+len(frame.metadata)+1):
            a,b = socket.socketpair()
            try:
                b.sendall(data[:count]); b.close()
                with self.assertRaises(w.PeerClosed): w.receive_frame(a,time.monotonic()+1)
            finally: a.close(); b.close()

    def test_reject_lengths_and_schema_before_payload_allocation(self):
        sock = Mock(); sock.gettimeout.return_value = None
        raw = bytearray(w.header(1,1,1)); raw[12:16] = struct.pack('!I',w.MAX_PAYLOAD+1)
        with patch.object(w,'_exact',return_value=bytes(raw)) as read:
            with self.assertRaises(w.ProtocolError): w.receive_frame(sock,time.monotonic()+1)
            self.assertEqual(read.call_count,1)
        raw = b'{"op":"Extract"}'
        with patch.object(w,'_exact',side_effect=[w.header(1,len(raw),w.MAX_PAYLOAD),raw]) as read:
            with self.assertRaises(w.ProtocolError): w.receive_frame(sock,time.monotonic()+1)
            self.assertEqual(read.call_count,2)


@unittest.skipUnless(sys.platform=='linux','Linux ownership and SO_PEERCRED')
class LinuxTests(unittest.TestCase):
    def test_socket_permissions_cleanup_and_peer(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder)/'wire.sock'
            with w.private_listener(path) as listener:
                self.assertEqual(stat.S_IMODE(path.stat().st_mode),0o600)
                with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
                    client.connect(str(path))
                    with listener.accept()[0] as server:
                        self.assertEqual(w.check_peer(client)[1],os.geteuid())
                        self.assertEqual(w.check_peer(server)[1],os.geteuid())
            self.assertFalse(path.exists()); self.assertEqual(listener.fileno(),-1)

    def test_existing_and_replacement_paths_are_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder)/'wire.sock'; path.write_bytes(b'synthetic sentinel')
            with self.assertRaises(FileExistsError):
                with w.private_listener(path): pass
            self.assertEqual(path.read_bytes(),b'synthetic sentinel'); path.unlink()
            with w.private_listener(path):
                path.rename(Path(folder)/'old.sock'); path.write_bytes(b'synthetic replacement')
            self.assertEqual(path.read_bytes(),b'synthetic replacement')

    def test_nonprivate_directory_and_foreign_peer_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            os.chmod(folder,0o755)
            with self.assertRaises(PermissionError):
                with w.private_listener(Path(folder)/'wire.sock'): pass
        sock = Mock(); sock.getsockopt.return_value = struct.pack('iII',1,os.geteuid()+1,0)
        with self.assertRaises(PermissionError): w.check_peer(sock)


@unittest.skipUnless(sys.platform=='linux' and os.environ.get('AUTOOCR_IPC_PROBE'), 'Compiled Linux C++ probe required')
class InteropTests(unittest.TestCase):
    def exchange(self, mode):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder)/'wire.sock'
            with w.private_listener(path) as listener:
                listener.settimeout(3)
                process = subprocess.Popen([os.environ['AUTOOCR_IPC_PROBE'],str(path)],
                                           stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                try:
                    with listener.accept()[0] as connection:
                        w.check_peer(connection)
                        frames=[w.receive_frame(connection,time.monotonic()+2) for _ in range(2)]
                        self.assertEqual([f.request_id for f in frames],[101,102])
                        if mode=='normal':
                            for frame in frames: w.send_frame(connection,frame,time.monotonic()+2)
                        elif mode=='wrong_id':
                            frame=frames[0]; w.send_frame(connection,w.Frame(103,frame.metadata,frame.payload),time.monotonic()+2)
                        elif mode=='truncated': connection.sendall(b'AO')
                        elif mode=='timeout':
                            out,err=process.communicate(timeout=4)
                            self.assertEqual(process.returncode,1); self.assertIn('timeout',err)
                            return
                    out,err=process.communicate(timeout=4)
                    if mode=='normal':
                        self.assertEqual(process.returncode,0,err)
                        self.assertEqual(json.loads(out)['payload_bytes'],48)
                    else: self.assertEqual(process.returncode,1)
                finally:
                    if process.poll() is None: process.kill()
                    process.communicate(timeout=4)
            self.assertFalse(path.exists())

    def test_cpp_self_test(self):
        result=subprocess.run([os.environ['AUTOOCR_IPC_PROBE'],'--self-test'],capture_output=True,text=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'],'cpp_transport_self_test_passed')

    def test_real_cpp_python_two_frame_roundtrip(self): self.exchange('normal')
    def test_cpp_rejects_wrong_response_id(self): self.exchange('wrong_id')
    def test_cpp_handles_truncated_response(self): self.exchange('truncated')
    def test_cpp_times_out_without_peer_progress(self): self.exchange('timeout')


if __name__=='__main__': unittest.main()
