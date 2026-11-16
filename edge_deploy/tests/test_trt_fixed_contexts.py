"""CPU-only ownership/routing tests; not CUDA performance validation."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from test_profile_rec_shapes import runner, tensor, FakeContext
from profile_rec_shapes import measure, SUCCESS
from trt_fixed_contexts import SharedContextRunner, FixedContextProfiler, close_context_pair


class FixedContextTests(unittest.TestCase):
    def setUp(self):
        self.owner = runner()
        self.owner.engine.device_memory_size_v2 = 4096
        self.creations = []

        def create():
            context = FakeContext(self.owner.trace)
            self.creations.append(context)
            return context

        self.owner.engine.create_execution_context = create
        self.shapes = {"a": (1, 3, 48, 484), "b": (1, 3, 48, 45)}
        self.fixed = None
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.owner.cuda.fail = None
        errors = close_context_pair(self.fixed, self.owner)
        self.assertEqual(errors, [])

    def make_fixed(self):
        self.fixed = FixedContextProfiler(self.owner, self.shapes)
        return self.fixed

    def test_one_engine_one_stream_distinct_contexts_and_buffers(self):
        fixed = self.make_fixed()
        self.assertEqual(len(self.creations), 1)  # Existing owner + exactly one new context.
        self.assertIs(fixed.secondary.engine, self.owner.engine)
        self.assertIs(fixed.secondary.stream, self.owner.stream)
        self.assertIsNot(fixed.secondary.context, self.owner.context)
        self.assertIsNot(fixed.secondary.buffers, self.owner.buffers)
        a, _ = fixed.run_tensor(tensor(484))
        a_copy = a.copy()
        b, _ = fixed.run_tensor(tensor(45))
        self.assertFalse(np.shares_memory(a, b))
        np.testing.assert_array_equal(a, a_copy)
        pointers = [p.value for r in fixed.runners.values() for p, _ in r.buffers.values()]
        self.assertEqual(len(set(pointers)), 4)
        resources = fixed.resources()
        self.assertTrue(resources["same_engine_object"])
        self.assertEqual(resources["engine_reported_context_device_memory_bytes"], 4096)
        self.assertGreater(resources["contexts"]["b"]["device_io_bytes"], 0)

    def test_alternating_calls_stop_shape_changes_after_first_use(self):
        fixed = self.make_fixed()
        for _ in range(4):
            for label, width in (("a", 484), ("b", 45)):
                raw, record = fixed.run_tensor(tensor(width))
                self.assertEqual(record["context_label"], label)
                self.assertGreaterEqual(record["dispatch_wall_ms"], record["call_wall_ms"])
                if _:
                    self.assertFalse(record["shape_changed"])
                    self.assertEqual(record["malloc_count"], 0)
                    self.assertEqual(record["free_count"], 0)
        self.assertEqual(sum(name == "set_input_shape" for name, _ in self.owner.trace), 2)
        calls = [name for name, _ in self.owner.trace if name in ("enqueue", "cudaStreamSynchronize")]
        self.assertEqual(calls, ["enqueue", "cudaStreamSynchronize"] * 8)

    def test_both_orders_preserve_text_and_fixed_invariants(self):
        fixed = self.make_fixed()
        samples = {label: {"tensor": tensor(shape[3]), "row": {"index": i, "text": "A"}}
                   for i, (label, shape) in enumerate(self.shapes.items())}
        for order in ("same-first", "alternating-first"):
            report = {}
            with redirect_stdout(io.StringIO()):
                measure(fixed, ["", "A", " "], samples, 1, 2, order, report)
            fixed.validate_measured(report)
            self.assertEqual(report["status"], SUCCESS)
            self.assertEqual(sum(len(c["records"]) for c in report["cases"]), 16)
            self.assertTrue(report["fixed_context_validation"]["all_measured_shapes_unchanged"])
            bad = deepcopy(report)
            bad["cases"][0]["records"][0]["shape_changed"] = True
            with self.assertRaisesRegex(ValueError, "invariant"):
                fixed.validate_measured(bad)

    def test_unknown_or_changed_route_rejected_without_enqueue(self):
        fixed = self.make_fixed()
        with self.assertRaisesRegex(ValueError, "Unknown shape"):
            fixed.run_tensor(tensor(32))
        self.owner.current_shape = (1, 3, 48, 32)
        with self.assertRaisesRegex(RuntimeError, "externally"):
            fixed.run_tensor(tensor(484))
        self.assertFalse(any(name == "enqueue" for name, _ in self.owner.trace))

    def test_constructor_rejects_unbounded_invalid_or_used_routes(self):
        for shapes in ({"a": self.shapes["a"]}, {"a": self.shapes["a"], "b": self.shapes["a"]},
                       {**self.shapes, "c": (1, 3, 48, 32)}, {**self.shapes, "b": (1, 3, 48, 1025)}):
            with self.assertRaises(ValueError):
                FixedContextProfiler(self.owner, shapes)
        self.owner.current_shape = self.shapes["a"]
        with self.assertRaisesRegex(ValueError, "fresh"):
            FixedContextProfiler(self.owner, self.shapes)
        self.assertEqual(self.creations, [])

    def test_context_creation_failure_does_not_close_owner(self):
        self.owner.engine.create_execution_context = lambda: None
        with self.assertRaisesRegex(RuntimeError, "second execution"):
            self.make_fixed()
        self.assertFalse(self.owner.closed)
        self.assertIsNotNone(self.owner.context)

    def test_secondary_close_does_not_destroy_shared_stream_or_owner(self):
        fixed = self.make_fixed()
        fixed.run_tensor(tensor(45))
        engine = self.owner.engine
        fixed.close()
        fixed.close()  # Idempotent.
        self.assertIs(self.owner.engine, engine)
        self.assertFalse(self.owner.closed)
        self.assertFalse(any(name == "cudaStreamDestroy" for name, _ in self.owner.trace))
        self.owner.run_tensor(tensor(484))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            fixed.run_tensor(tensor(484))

    def test_failed_secondary_sync_retains_resources_for_retry(self):
        fixed = self.make_fixed()
        fixed.run_tensor(tensor(45))
        self.owner.cuda.fail = "cudaStreamSynchronize"
        with self.assertRaisesRegex(RuntimeError, "injected CUDA"):
            fixed.close()
        self.assertFalse(fixed.closed)
        self.assertIsNotNone(fixed.secondary.context)
        self.assertEqual(len(fixed.secondary.buffers), 2)
        self.owner.cuda.fail = None
        fixed.close()

    def test_execution_failure_has_context_tag_and_final_sync(self):
        fixed = self.make_fixed()
        fixed.secondary.context.fail = True
        with self.assertRaisesRegex(RuntimeError, "execution failed"):
            fixed.run_tensor(tensor(45))
        self.assertEqual(fixed.last_record["context_label"], "b")
        self.assertEqual(fixed.last_record["status"], "error")
        self.assertEqual(self.owner.trace[-1][0], "cudaStreamSynchronize")

    def test_cleanup_attempts_secondary_then_owner_even_on_failure(self):
        calls = []

        class Broken:
            def __init__(self, name):
                self.name = name

            def close(self):
                calls.append(self.name)
                raise RuntimeError(self.name)

        errors = close_context_pair(Broken("child"), Broken("owner"))
        self.assertEqual(calls, ["child", "owner"])
        self.assertEqual(len(errors), 2)


if __name__ == "__main__":
    unittest.main()
