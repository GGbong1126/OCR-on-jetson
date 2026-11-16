"""E165 process-local ablations; never modify the installed E162/E164 sources."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import textwrap
import threading
import time

MODES = ('sync', 'full', 'one_inflight', 'blocking_submit', 'no_fanout', 'inline_ctc')


def transform(source, mode):
    """Fail closed if the frozen implementation no longer matches the intervention."""
    edits = {
        'one_inflight': ('for slot_id,r in work:\n',
                         'for slot_id,r in work:\n                        if slot_id!=5 and len(active)>=1: continue\n'),
        'no_fanout': ("j.timings['rec_prepare_and_backpressure_ms']=j.timings.get('rec_prepare_and_backpressure_ms',0)+(time.monotonic()-start)*1000\n",
                      "j.timings['rec_prepare_and_backpressure_ms']=j.timings.get('rec_prepare_and_backpressure_ms',0)+(time.monotonic()-start)*1000\n                    b.cv.wait_for(lambda:request.removed)\n"),
        'inline_ctc': ('self.rec_decodes.append(r); self.notify()',
                       'self._ablation_decode(r); self.notify()'),
    }
    if mode not in edits:
        return source
    old, new = edits[mode]
    assert source.count(old) == 1, (mode, 'frozen source differs')
    return source.replace(old, new)


class Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.local = threading.local()
        self.rows = {}

    def call(self, label, fn, *args, **kwargs):
        wall = time.monotonic_ns(); cpu = time.thread_time_ns()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.monotonic_ns() - wall; used = time.thread_time_ns() - cpu
            with self.lock:
                row = self.rows.setdefault(label, [0, 0, 0])
                row[0] += 1; row[1] += elapsed; row[2] += used

    def snapshot(self):
        with self.lock:
            return {k: v[:] for k, v in self.rows.items()}


def delta(first, last):
    return {k: dict(calls=v[0]-first.get(k, [0, 0, 0])[0],
                    wall_ms=(v[1]-first.get(k, [0, 0, 0])[1])/1e6,
                    thread_cpu_ms=(v[2]-first.get(k, [0, 0, 0])[2])/1e6)
            for k, v in last.items()}


def install_bridge(bridge, mode):
    changed = {}
    targets = {'one_inflight': (bridge.MultiBridge, 'gpu_loop'),
               'no_fanout': (bridge.MultiAPI, 'recognize'),
               'inline_ctc': (bridge.MultiBridge, 'step')}
    if mode in targets:
        cls, method = targets[mode]
        old = textwrap.dedent(inspect.getsource(getattr(cls, method)))
        patched = transform(inspect.getsource(getattr(cls, method)), mode)
        if mode == 'inline_ctc':
            assert cls.__bases__ == (bridge.WorkerBridge,) and patched.count('super().step()') == 1
            # A standalone recompiled function has no lexical __class__ cell.
            patched = patched.replace('super().step()', 'WorkerBridge.step(self)')
        namespace = dict(vars(bridge))
        exec(compile(textwrap.dedent(patched), '<isolated_ablation>', 'exec'), namespace)
        setattr(cls, method, namespace[method])
        changed[method] = dict(before_sha256=hashlib.sha256(old.encode()).hexdigest(),
                               after_sha256=hashlib.sha256(textwrap.dedent(patched).encode()).hexdigest())
    if mode == 'inline_ctc':
        original = inspect.getsource(bridge.MultiBridge.decode_loop)
        start = original.index('                value=None; failure=False; begin=time.monotonic()')
        end = original.index('\n        finally:\n            with self.cv: self.decoder_closed=True')
        body = textwrap.indent(textwrap.dedent(original[start:end]), '    ')
        namespace = dict(vars(bridge))
        exec(compile('def _ablation_decode(self,r):\n'+body, '<original_ctc_body>', 'exec'), namespace)
        bridge.MultiBridge._ablation_decode = namespace['_ablation_decode']
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, extra = parser.parse_known_args()
    assert not extra
    source = args.source.resolve(); output = args.output.resolve()
    base = Path('/home/xchq/work/autoocr')
    assert output.parent.is_relative_to(base/'staging') and not output.exists()
    diagnostic = output.with_suffix('.host.json'); assert not diagnostic.exists()
    sys.path[:0] = [str(source/'ipc'), str(source/'app'), str(source/'multistream'),
                   str(base/'overlays/rec-wide-e124-r1'), str(base/'releases/jetson-r2/edge_deploy/jetson')]
    spec = importlib.util.spec_from_file_location('ablation_verifier', source/'ipc/verify_ocr_workers.py')
    bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
    import trt82_backend as backend
    import trt_async
    import runtime
    import bridge
    recorder = Recorder(); marks = []
    original_init = backend.Session.__init__

    class Context:
        def __init__(self, context, role):
            self.context = context; self.role = role

        def __getattr__(self, name):
            value = getattr(self.context, name)
            if name not in ('set_binding_shape', 'execute_v2', 'execute_async_v2'):
                return value
            return lambda *a, **k: recorder.call(self.role+'.context.'+name, value, *a, **k)

    def role(name):
        return 'rec' if 'rec' in name else ('det' if 'det' in name else name)

    def initialize(session, *a, **k):
        original_init(session, *a, **k)
        session.context = Context(session.context, role(session.role))

    backend.Session.__init__ = initialize
    old_call = backend.Cuda.call
    def cuda(cuda, name, *a):
        kind = getattr(recorder.local, 'role', 'startup_or_close')
        return recorder.call(kind+'.cuda.'+name, old_call, cuda, name, *a)
    backend.Cuda.call = cuda

    def wrap(cls, method, category, after=None):
        original = getattr(cls, method)
        def call(obj, *a, **k):
            previous = getattr(recorder.local, 'role', 'startup_or_close')
            name = role((obj.session if hasattr(obj, 'session') else obj).role)
            recorder.local.role = name
            try:
                result = recorder.call(name+'.'+category, original, obj, *a, **k)
                if after:
                    recorder.call(name+'.injected_stream_wait', after, obj)
                return result
            finally:
                recorder.local.role = previous
        setattr(cls, method, call)

    wrap(backend.Session, 'run', 'sync_run')
    wait = (lambda slot: slot.cuda.call('cudaStreamSynchronize', slot.stream)) if args.mode == 'blocking_submit' else None
    wrap(trt_async.Slot, 'submit', 'slot_submit', wait)
    wrap(trt_async.Slot, 'collect', 'slot_collect')
    old_split = runtime.rec_functions
    def split(engine):
        prepare, decode = old_split(engine)
        return (lambda *a: recorder.call('rec.prepare_cpu', prepare, *a),
                lambda *a: recorder.call('rec.ctc_cpu', decode, *a))
    runtime.rec_functions = split

    changed = install_bridge(bridge, args.mode)

    original_snapshot = bench.resource_snapshot
    def snapshot(pids):
        value = original_snapshot(pids)
        if threading.current_thread().name == 'cpp-validation-front':
            marks.append(recorder.snapshot())
        return value
    bench.resource_snapshot = snapshot
    binary = base/'staging/multistream_e164_r1/cmake_r16/app/autoocr_batch_app'
    sys.argv = [str(source/'ipc/verify_ocr_workers.py'), '--output', str(output), '--batch-app', str(binary),
                '--all-images', '--benchmark', '--repeat', '3', '--public-response',
                '--rec-slots', '0' if args.mode == 'sync' else '4']
    success = False
    try:
        bench.main(); success = True
    finally:
        value = dict(mode=args.mode,completed=success,changed_methods=changed,
                     snapshot_count=len(marks),source=str(source),
                     measured=delta(marks[0], marks[1]) if len(marks) == 2 else {},
                     note='Measured 78-page host intervals only. CUDA API and context methods are nested in run/submit; never sum levels. Blocking duration is not pure kernel time. No profiler privileges.')
        with diagnostic.open('x') as file:
            json.dump(value, file, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
