"""Bounded two-context diagnostic sharing ONE engine and ONE serial stream.

Not a production cache. Close the secondary context before closing its owner.
Uses original _prepare and host profiling without changing precision or inputs.
"""
from time import perf_counter

from trt_backend import TensorRTDetector
from trt_host_profile import HostProfiler


class SharedContextRunner:
    # Reuse the exact baseline shape/buffer preparation, not detector processing.
    _prepare = TensorRTDetector._prepare

    def __init__(self, owner):
        if owner.closed or owner.engine is None or not owner.stream.value:
            raise ValueError("Expected a live engine/stream owner")
        self.owner = owner  # Keep engine/runtime/logger alive until child cleanup.
        self.engine = owner.engine
        self.cuda = owner.cuda
        self.stream = owner.stream  # Borrowed, never destroyed here.
        self.input_name, self.output_name = owner.input_name, owner.output_name
        self.buffers = {}
        self.current_shape = None
        self.closed = False
        self.context = self.engine.create_execution_context()
        if self.context is None:
            self.closed = True
            raise RuntimeError("Cannot create second execution context")
        if self.context is owner.context:
            self.closed = True
            self.context = None
            raise RuntimeError("Expected an independent execution context")

    def close(self):
        if self.closed:
            return
        # Do not release live buffers/context if synchronization fails.
        self.cuda.call("cudaStreamSynchronize", self.stream)
        self.context = None
        errors = []
        for name in list(self.buffers):
            try:
                self.cuda.call("cudaFree", self.buffers[name][0])
                del self.buffers[name]
            except Exception as error:
                errors.append(f"{name}: {error}")
        if errors:
            raise RuntimeError("Secondary buffer cleanup failed: " + "; ".join(errors))
        self.host_output = None
        self.engine = self.owner = None
        self.closed = True


class FixedContextProfiler:
    def __init__(self, owner, shapes):
        shapes = {label: tuple(shape) for label, shape in shapes.items()}
        if set(shapes) != {"a", "b"} or len(set(shapes.values())) != 2:
            raise ValueError("Exactly two distinct A/B shapes are required")
        low, _, high = owner.engine.get_tensor_profile_shape(owner.input_name, 0)
        for shape in shapes.values():
            if len(shape) != 4 or shape[:3] != (1, 3, 48) or any(
                    not lo <= s <= hi for s, lo, hi in zip(shape, low, high)):
                raise ValueError("Fixed recognition shape outside engine profile")
        if owner.current_shape is not None:
            raise ValueError("Use a fresh owner context, not a previously measured one")
        self.shapes = shapes
        self.shape_to_label = {shape: label for label, shape in shapes.items()}
        self.last_record = None
        self.closed = False
        start = perf_counter()
        self.secondary = SharedContextRunner(owner)
        self.second_context_create_ms = (perf_counter() - start) * 1000
        self.runners = {"a": owner, "b": self.secondary}
        self.profilers = {label: HostProfiler(runner) for label, runner in self.runners.items()}

    def run_tensor(self, tensor):
        start = perf_counter()
        self.last_record = {"status": "started", "input_shape": list(tensor.shape)}
        profiler = None
        try:
            if self.closed:
                raise RuntimeError("Fixed-context profiler is closed")
            label = self.shape_to_label.get(tuple(tensor.shape))
            if label is None:
                raise ValueError("Unknown shape; fixed contexts do not resize or add routes")
            runner = self.runners[label]
            if runner.closed or self.runners["a"].closed:
                raise RuntimeError("Context owner is closed")
            if runner.current_shape not in (None, self.shapes[label]):
                raise RuntimeError("Fixed context shape was changed externally")
            profiler = self.profilers[label]
            raw, details = profiler.run_tensor(tensor)
            return raw, details
        except BaseException as error:
            self.last_record.update(status="error", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if profiler is not None:
                self.last_record = profiler.last_record
                self.last_record["context_label"] = label
            self.last_record["dispatch_wall_ms"] = (perf_counter() - start) * 1000

    def resources(self):
        owner = self.runners["a"]
        result = {
            "engine_instances": 1, "context_count": 2, "stream_count": 1,
            "same_engine_object": self.secondary.engine is owner.engine,
            "same_stream_handle": self.secondary.stream.value == owner.stream.value,
            "distinct_context_objects": self.secondary.context is not owner.context,
            "second_context_create_ms": self.second_context_create_ms,
            "contexts": {},
            "memory_note": "Explicit I/O byte counts exclude weights, TRT internal activation/workspace and allocator overhead. Engine device-memory requirement is not measured process memory or a peak.",
        }
        requirement = getattr(owner.engine, "device_memory_size_v2", None)
        result["engine_reported_context_device_memory_bytes"] = (
            int(requirement) if requirement is not None else None)
        for label, runner in self.runners.items():
            result["contexts"][label] = {
                "assigned_shape": list(self.shapes[label]),
                "current_shape": list(runner.current_shape) if runner.current_shape else None,
                "device_io_bytes": sum(size for _, size in runner.buffers.values()),
                "host_output_bytes": getattr(getattr(runner, "host_output", None), "nbytes", 0),
            }
        return result

    def validate_measured(self, report):
        for case in report["cases"]:
            for record in case["records"]:
                if (record.get("context_label") != record["sample"] or record["shape_changed"]
                        or record["malloc_count"] or record["free_count"]):
                    raise ValueError("Fixed context invariant failed in measured calls")
        report["fixed_context_validation"] = {
            "all_measured_routes_correct": True, "all_measured_shapes_unchanged": True,
            "all_measured_explicit_allocation_counts_zero": True,
        }

    def close(self):
        if not self.closed:
            self.secondary.close()
            self.closed = True


def close_context_pair(fixed, recognizer):
    """Attempt secondary first, then owner; preserve both errors for the report."""
    errors = []
    for label, target in (("secondary_context", fixed), ("owner_recognizer", recognizer)):
        if target is not None:
            try:
                target.close()
            except Exception as error:
                errors.append(f"{label}: {type(error).__name__}: {error}")
    return errors
