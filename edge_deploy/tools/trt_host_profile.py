"""Host-side diagnostic of the existing sequential TensorRT runner.

Preserves _prepare, pageable buffers, copies and one final synchronization.
API return times are NOT device/kernel timings. Single-threaded use only.
"""
import ctypes as ct
from time import perf_counter

import numpy as np


METRICS = ("prepare_ms", "pack_ms", "h2d_api_ms", "enqueue_api_ms",
           "d2h_api_ms", "sync_ms", "call_wall_ms")


class AllocationTrace:
    def __init__(self, cuda, record):
        self.cuda, self.record = cuda, record

    def call(self, name, *args):
        if name not in ("cudaMalloc", "cudaFree"):
            return self.cuda.call(name, *args)
        event = {"api": name, "success": False}
        if name == "cudaMalloc":
            event["bytes"] = int(args[1])
        start = perf_counter()
        try:
            result = self.cuda.call(name, *args)
            event["success"] = True
            return result
        finally:
            event["host_ms"] = (perf_counter() - start) * 1000
            self.record["allocation_events"].append(event)


class HostProfiler:
    def __init__(self, runner):
        self.runner = runner
        self.last_record = None

    def run_tensor(self, tensor):
        runner = self.runner
        record = {"input_shape": list(tensor.shape),
                  "shape_before": list(runner.current_shape) if runner.current_shape else None,
                  "shape_changed": tuple(tensor.shape) != runner.current_shape,
                  "allocation_events": [], "status": "started"}
        self.last_record = record
        start = perf_counter()
        original_cuda = runner.cuda

        def timed(key, fn, *args, **kwargs):
            tick = perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                record[key] = (perf_counter() - tick) * 1000

        try:
            if runner.closed:
                raise RuntimeError("Runner is closed")
            if tensor.dtype != np.float32 or tensor.ndim != 4 or tensor.shape[:3] != (1, 3, 48):
                raise ValueError("Expected float32 [1,3,48,W]")
            runner.cuda = AllocationTrace(original_cuda, record)
            timed("prepare_ms", runner._prepare, tuple(tensor.shape))
            host_input = timed("pack_ms", np.ascontiguousarray, tensor)
            record.update(input_bytes=host_input.nbytes, output_bytes=runner.host_output.nbytes,
                          output_shape=list(runner.host_output.shape))
            try:
                timed("h2d_api_ms", runner.cuda.call, "cudaMemcpyAsync",
                      runner.buffers[runner.input_name][0], ct.c_void_p(host_input.ctypes.data),
                      host_input.nbytes, 1, runner.stream)
                if not timed("enqueue_api_ms", runner.context.execute_async_v3,
                             stream_handle=runner.stream.value):
                    raise RuntimeError("TensorRT execution failed")
                timed("d2h_api_ms", runner.cuda.call, "cudaMemcpyAsync",
                      ct.c_void_p(runner.host_output.ctypes.data),
                      runner.buffers[runner.output_name][0], runner.host_output.nbytes, 2, runner.stream)
            finally:
                # Keep host buffers alive and preserve baseline's failure synchronization.
                timed("sync_ms", runner.cuda.call, "cudaStreamSynchronize", runner.stream)
            record["status"] = "completed"
            return runner.host_output, record  # Borrowed: consume before the next call.
        except BaseException as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            runner.cuda = original_cuda
            record["call_wall_ms"] = (perf_counter() - start) * 1000
            events = record["allocation_events"]
            for name, prefix in (("cudaMalloc", "malloc"), ("cudaFree", "free")):
                selected = [e for e in events if e["api"] == name]
                record[prefix + "_count"] = sum(e["success"] for e in selected)
                record[prefix + "_host_ms"] = sum(e["host_ms"] for e in selected)
