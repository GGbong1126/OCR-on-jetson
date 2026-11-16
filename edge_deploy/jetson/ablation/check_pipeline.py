"""Real C++/Unix channels with fake tensors; no models or GPU loaded."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from experiment import MODES, install_bridge


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--mode', choices=MODES, required=True)
    a = p.parse_args()
    sys.path[:0] = [str(a.source/'ipc'), str(a.source/'app'), str(a.source/'multistream')]
    import bridge
    from test_multibridge import PipelineTests, Runtime, Backend
    from batch_front_client import exchange
    from stage_session import StateError
    install_bridge(bridge, a.mode)
    if a.mode == 'blocking_submit':
        original = Runtime.submit_rec
        def submit(self, *args):
            event = original(self, *args)
            while not event.ready(): time.sleep(.0001)
            return event
        Runtime.submit_rec = submit
    case = PipelineTests()
    result = case.exercise(4)
    if a.mode == 'one_inflight': assert result['worker']['rec_high_inflight'] == 1
    case.exercise(4, zero=True)
    # With per-crop waiting, decode failure may return per-page errors before
    # another inference is accepted, rather than closing the whole transport.
    with tempfile.TemporaryDirectory(prefix='ablation_error_') as folder, Backend(folder,4,fail_decode=True) as backend:
        front=Path(folder)/'front.sock'
        cpp=subprocess.Popen([os.environ['AUTOOCR_BATCH_APP'],str(backend.path),str(front),'--rec-slots=4'],
                             stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            end=time.monotonic()+5
            while not front.exists() and cpp.poll() is None and time.monotonic()<end: time.sleep(.01)
            assert front.exists()
            try:
                result=exchange(front,[('a',3,b'a'),('b',3,b'b')],remaining_ms=10000)
                assert len(result['results'])==2 and all(r['status']=='error' for r in result['results'])
            except StateError:
                cpp.wait(5); assert cpp.returncode==1
        finally:
            if cpp.poll() is None: cpp.terminate()
            cpp.wait(5)
    assert backend.end['worker']['rec_tasks']==backend.end['session']['registered_bytes']==0
    print(a.mode, '80 ordered results, zero crop, decode error and safe retirement passed', flush=True)


if __name__ == '__main__': main()
