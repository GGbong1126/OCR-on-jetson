"""Bounded, serial fixed-width contexts sharing the existing FP32 engine."""
from benchmark_rec import validate_profile
from trt_fixed_contexts import SharedContextRunner
from trt_host_profile import HostProfiler


class BucketContexts:
    def __init__(self, owner, widths, max_contexts=7):
        widths = list(widths)
        if (not widths or widths != sorted(set(widths)) or
                any(type(w) is not int or w < 32 for w in widths) or
                type(max_contexts) is not int or not 1 <= max_contexts <= 7 or
                len(widths) > max_contexts):
            raise ValueError("Expected 1..7 bounded, sorted unique widths")
        if owner.closed or owner.current_shape is not None:
            raise ValueError("Expected fresh live owner")
        validate_profile([{"index": w, "input_shape": [1, 3, 48, w]} for w in widths],
                         owner.engine.get_tensor_profile_shape(owner.input_name, 0))
        self.owner, self.widths = owner, widths
        self.runners = {widths[0]: owner}
        self.profilers = {}
        self.ready = self.closed = False
        self.last_record = None

    def initialize(self):
        # Explicit second phase: caller retains this object on partial allocation failure.
        if self.closed or self.ready or len(self.runners) != 1:
            raise RuntimeError("Pool cannot be initialized again")
        for width in self.widths[1:]:
            self.runners[width] = SharedContextRunner(self.owner)
        if len({id(r.context) for r in self.runners.values()}) != len(self.widths):
            raise RuntimeError("Contexts must be independent")
        self.profilers = {w: HostProfiler(r) for w, r in self.runners.items()}
        self.ready = True

    def run_tensor(self, tensor):
        self.last_record = None
        if self.closed or not self.ready or self.owner.closed:
            raise RuntimeError("Pool is not live")
        shape = tuple(tensor.shape)
        if len(shape) != 4 or shape[:3] != (1, 3, 48) or shape[3] not in self.runners:
            raise ValueError("Unknown fixed bucket shape; no resizing/fallback")
        width = shape[3]
        runner = self.runners[width]
        if runner.current_shape not in (None, shape):
            raise RuntimeError("Fixed bucket shape changed externally")
        profiler = self.profilers[width]
        try:
            return profiler.run_tensor(tensor)
        finally:
            self.last_record = profiler.last_record
            if self.last_record is not None:
                self.last_record["context_width"] = width

    def resources(self):
        return {"engine_instances": 1, "stream_count": 1, "context_count": len(self.runners),
                "widths": self.widths,
                "contexts": {str(w): {"current_shape": list(r.current_shape) if r.current_shape else None,
                                      "device_io_bytes": sum(n for _, n in r.buffers.values()),
                                      "host_output_bytes": getattr(getattr(r, "host_output", None), "nbytes", 0)}
                             for w, r in self.runners.items()},
                "note": "I/O bytes exclude engine weights, context activation/workspace and allocator overhead."}

    def close(self):
        if self.closed:
            return
        errors = []
        for width, runner in reversed(list(self.runners.items())):
            if runner is self.owner:
                continue
            try:
                runner.close()
            except Exception as error:
                errors.append(f"bucket {width}: {type(error).__name__}: {error}")
        if errors:
            # Keep engine/stream owner alive while any child failed synchronization/free.
            raise RuntimeError("; ".join(errors))
        self.closed = True
