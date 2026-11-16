import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from test_multibridge import Runtime
from bridge import MultiBridge
from stage2 import MultiChannel
import ipc_wire as wire
from pathlib import Path


@unittest.skipUnless(sys.platform=='linux','Linux owned Unix channel required')
class StartupTests(unittest.TestCase):
    def reject_start(self,name):
        before=set(threading.enumerate()); captured=[]; runtimes=[]
        original=threading.Thread.start
        def start(thread):
            if thread.name==name:
                captured.append(thread._target.__self__)
                raise RuntimeError('injected auxiliary thread start failure')
            return original(thread)
        def factory():
            runtime=Runtime();runtimes.append(runtime);return runtime
        with tempfile.TemporaryDirectory() as folder,wire.private_listener(Path(folder)/'backend.sock') as listener:
            channel=MultiChannel(listener,4)
            try:
                with patch.object(threading.Thread,'start',start):
                    with self.assertRaisesRegex(RuntimeError,'auxiliary thread start failure'):
                        MultiBridge(channel,lambda *a:b'',factory)
                self.assertEqual(len(captured),1)
                self.assertTrue(captured[0].closed,'Constructor left owned workers/resources running')
                self.assertTrue(runtimes and all(r.closed for r in runtimes))
                self.assertEqual(set(threading.enumerate())-before,set())
            finally:
                # Let the old-code negative fail cleanly; the fake runtime never touches CUDA.
                for bridge in captured:
                    if bridge.closed: continue
                    bridge.stop()
                    for t in (*bridge.cpu,bridge.gpu,bridge.decoder,bridge.cpu_route):
                        if t.ident is not None:t.join(5)
                        self.assertFalse(t.is_alive())
                    channel.selector.unregister(bridge.reader)
                    bridge.reader.close();bridge.writer.close()
                channel.close()
    def test_decoder_start_failure_retires_owned_workers(self):self.reject_start('ocr-ctc')
    def test_cpu_route_start_failure_retires_started_decoder(self):self.reject_start('ocr-rec-cpu-route')


if __name__=='__main__':unittest.main()
