"""CPU-only diagnostics tests; no claims about real CUDA latency."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import ctypes as ct
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from trt_backend import TensorRTDetector
from trt_host_profile import HostProfiler, METRICS
from profile_rec_shapes import main, measure, schedules, select_samples, verify_baseline, SUCCESS


class FakeCuda:
    def __init__(self, trace):
        self.trace, self.pointer = trace, 1000
        self.fail = None
        self.nan = False

    def call(self, name, *args):
        self.trace.append((name, args[3] if name == "cudaMemcpyAsync" else None))
        if self.fail == name:
            raise RuntimeError("injected CUDA failure")
        if name == "cudaMalloc":
            self.pointer += 1000
            args[0]._obj.value = self.pointer
        if name == "cudaMemcpyAsync" and args[3] == 2:
            raw = np.ctypeslib.as_array(ct.cast(args[0], ct.POINTER(ct.c_float)),
                                       shape=(args[2] // 4,))
            raw.fill(0)
            raw.reshape(-1, 3)[:, 1] = np.nan if self.nan else 1


class FakeContext:
    def __init__(self, trace):
        self.trace, self.shape, self.fail = trace, None, False

    def set_input_shape(self, name, shape):
        self.trace.append(("set_input_shape", tuple(shape)))
        self.shape = shape
        return True

    def get_tensor_shape(self, name):
        return (1, max(1, self.shape[-1] // 8), 3)

    def set_tensor_address(self, name, pointer):
        self.trace.append(("bind", name))
        return True

    def execute_async_v3(self, **kwargs):
        self.trace.append(("enqueue", None))
        return not self.fail


class FakeEngine:
    def create_execution_context(self):
        return FakeContext([])

    def get_tensor_profile_shape(self, name, profile):
        return [(1, 3, 48, 32), (1, 3, 48, 160), (1, 3, 48, 1024)]


def runner():
    obj = TensorRTDetector.__new__(TensorRTDetector)
    obj.trace = []
    obj.cuda, obj.context = FakeCuda(obj.trace), FakeContext(obj.trace)
    obj.engine, obj.runtime = FakeEngine(), None
    obj.stream, obj.buffers = ct.c_void_p(1), {}
    obj.current_shape, obj.closed = None, False
    obj.input_name, obj.output_name = "x", "y"
    return obj


def tensor(width):
    # Same non-contiguous layout as the normal HWC -> NCHW preprocessing.
    return np.zeros((48, width, 3), dtype=np.float32).transpose(2, 0, 1)[None]


class ShapeProfileTests(unittest.TestCase):
    def setUp(self):
        self.runner = runner()
        self.addCleanup(self.runner.close)
        self.profiler = HostProfiler(self.runner)
        self.samples = {label: {"tensor": tensor(width), "row": {"index": index, "text": "A"}}
                        for label, width, index in (("a", 484, 50), ("b", 45, 49))}

    def test_baseline_call_sequence_unchanged(self):
        baseline = runner()
        self.addCleanup(baseline.close)
        for width in (484, 484, 45, 484):
            expected = baseline.run_tensor(tensor(width)).copy()
            raw, record = self.profiler.run_tensor(tensor(width))
            np.testing.assert_array_equal(raw, expected)
            self.assertEqual(self.runner.trace, baseline.trace)
            self.assertTrue(all(record[k] >= 0 for k in METRICS))
            self.assertLessEqual(record["malloc_host_ms"] + record["free_host_ms"], record["prepare_ms"])

    def test_allocation_counts_reuse_and_borrowed_output(self):
        cuda = self.runner.cuda
        first, r1 = self.profiler.run_tensor(tensor(484))
        second, r2 = self.profiler.run_tensor(tensor(484))
        _, r3 = self.profiler.run_tensor(tensor(45))
        self.assertIs(first, second)
        self.assertEqual((r1["malloc_count"], r1["free_count"]), (2, 0))
        self.assertEqual((r2["malloc_count"], r2["free_count"]), (0, 0))
        self.assertFalse(r2["shape_changed"])
        self.assertEqual((r3["malloc_count"], r3["free_count"]), (2, 2))
        self.assertIs(self.runner.cuda, cuda)

    def test_enqueue_failure_synchronizes_and_restores(self):
        cuda = self.runner.cuda
        self.runner.context.fail = True
        with self.assertRaisesRegex(RuntimeError, "execution failed"):
            self.profiler.run_tensor(tensor(484))
        self.assertIs(self.runner.cuda, cuda)
        self.assertEqual(self.runner.trace[-1][0], "cudaStreamSynchronize")
        self.assertNotIn("d2h_api_ms", self.profiler.last_record)
        self.assertEqual(self.profiler.last_record["status"], "error")

    def test_copy_failure_synchronizes(self):
        self.runner.cuda.fail = "cudaMemcpyAsync"
        with self.assertRaisesRegex(RuntimeError, "injected CUDA"):
            self.profiler.run_tensor(tensor(32))
        self.assertEqual(self.runner.trace[-1][0], "cudaStreamSynchronize")

    def test_prepare_failure_records_failed_allocation(self):
        cuda = self.runner.cuda
        cuda.fail = "cudaMalloc"
        with self.assertRaises(RuntimeError):
            self.profiler.run_tensor(tensor(32))
        self.assertIs(self.runner.cuda, cuda)
        record = self.profiler.last_record
        self.assertEqual(record["malloc_count"], 0)
        self.assertFalse(record["allocation_events"][0]["success"])
        self.assertNotIn("sync_ms", record)

    def test_invalid_input_and_out_of_profile(self):
        for data in (tensor(32).astype(np.float16), tensor(1025)):
            with self.assertRaises(ValueError):
                self.profiler.run_tensor(data)
        self.assertFalse(any(item[0] == "enqueue" for item in self.runner.trace))

    def test_measure_separates_warmup_and_controls_same_tensor(self):
        report = {}
        with redirect_stdout(io.StringIO()):
            measure(self.profiler, ["", "A", " "], self.samples, 1, 2, "same-first", report)
        self.assertEqual(report["status"], SUCCESS)
        self.assertEqual(len(report["cases"]), 4)
        self.assertEqual(sum(len(c["records"]) for c in report["cases"]), 16)
        for case in report["cases"]:
            self.assertEqual(len(case["warmup_records"]), 2)
            expected_changes = 4 if case["name"].startswith("alternate") else 0
            self.assertEqual(sum(r["shape_changed"] for r in case["records"]), expected_changes)
            self.assertTrue(all(r["text_equal"] for r in case["records"]))
            self.assertTrue(all("text" not in r for r in case["records"]))
        self.assertEqual(schedules("alternating-first")[0][0], "alternate_ab")

    def test_nan_text_mismatch_and_interrupt_keep_partial_records(self):
        for mode in ("nan", "text", "interrupt"):
            report = {}
            self.runner.cuda.nan = mode == "nan"
            self.samples["a"]["row"]["text"] = "B" if mode == "text" else "A"
            with patch.object(self.runner.context, "execute_async_v3",
                              side_effect=KeyboardInterrupt if mode == "interrupt" else None,
                              return_value=True):
                with self.assertRaises((ValueError, KeyboardInterrupt)):
                    measure(self.profiler, ["", "A", " "], self.samples, 1, 2, "same-first", report)
            self.assertEqual(report["cases"][0]["warmup_records"][0]["status"], "error")
            self.assertNotIn("summary", report)

    def test_provenance_and_engine_mismatch(self):
        provenance = {"image_sha256": "image", "source_report_sha256": "source", "source_branch": "reference"}
        baseline = {"status": SUCCESS, "rec_backend": "trt", **provenance,
                    "recognizer": {"engine_sha256": "engine"}}
        verify_baseline(baseline, provenance, "engine")
        for key in provenance:
            bad = deepcopy(baseline)
            bad[key] = "wrong"
            with self.assertRaisesRegex(ValueError, "provenance"):
                verify_baseline(bad, provenance, "engine")
        with self.assertRaisesRegex(ValueError, "Engine differs"):
            verify_baseline(baseline, provenance, "wrong")

    def test_selection_requires_distinct_original_widths(self):
        image = np.zeros((48, 100, 3), dtype=np.uint8)
        rows = [{"index": i, "box": [[0, 0], [w, 0], [w, 48], [0, 48]]}
                for i, w in ((1, 32), (2, 48), (3, 32))]
        samples = select_samples(rows, image, 1, 2)
        self.assertEqual(samples["a"]["tensor"].shape, (1, 3, 48, 32))
        for a, b in ((1, 1), (1, 3), (1, 9)):
            with self.assertRaises(ValueError):
                select_samples(rows, image, a, b)

    def test_cli_failure_report_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            argv = ["--engine", "missing", "--onnx", "missing", "--dict", "missing",
                    "--image", "missing", "--source-report", "missing", "--baseline-report", "missing",
                    "--output", str(output)]
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(argv)
            self.assertEqual(error.exception.code, 2)
            content = output.read_bytes()
            self.assertEqual(json.loads(content)["status"], "invalid_diagnostic")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(argv)
            self.assertEqual(output.read_bytes(), content)

    def test_main_success_and_cleanup_failure_preserves_evidence(self):
        image = np.zeros((48, 80, 3), dtype=np.uint8)
        rows = [{"index": index, "box": [[0, 0], [width, 0], [width, 48], [0, 48]],
                 "input_shape": [1, 3, 48, width], "status": "ok", "text": "A"}
                for index, width in ((50, 32), (49, 48))]
        source = {"status": "compared_not_accuracy_validated", "image_sha256": "image",
                  "recognizer": {"model_sha256": "onnx", "dictionary_sha256": "dict"},
                  "reference": {"records": rows}}
        crop_set = [{k: r[k] for k in ("index", "box", "input_shape")} for r in rows]
        baseline = {"status": SUCCESS, "rec_backend": "trt", "source_branch": "reference",
                    "source_report_sha256": "source", "image_sha256": "image",
                    "reference_rec_onnx_sha256": "onnx", "dictionary_sha256": "dict",
                    "crop_set_sha256": hashlib.sha256(json.dumps(crop_set, sort_keys=True).encode()).hexdigest(),
                    "recognizer": {"engine_sha256": "engine"}}
        for context_mode, fail_cleanup in (("single", False), ("single", True), ("fixed", False), ("fixed", True)):
            backend = runner()
            self.addCleanup(backend.close)

            class FakeRec:
                characters = ["", "A", " "]
                profile = FakeEngine().get_tensor_profile_shape("x", 0)

                def __init__(self, *args):
                    self.runner = backend

                def metadata(self):
                    return {"backend": "fake_test_only"}

                def close(self):
                    backend.close()
                    if fail_cleanup:
                        raise RuntimeError("injected cleanup failure")

            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "report.json"
                argv = ["--engine", "engine", "--onnx", "onnx", "--dict", "dict",
                        "--image", "image", "--source-report", "source", "--baseline-report", "baseline",
                        "--warmup", "1", "--repeat", "1", "--context-mode", context_mode,
                        "--memory", "--output", str(output)]
                with patch("profile_rec_shapes.TensorRTRecognizer", FakeRec), \
                     patch("profile_rec_shapes.read_image", return_value=image), \
                     patch("profile_rec_shapes.sha256_file", side_effect=lambda path: str(path)), \
                     patch("benchmark_ocr.memory_snapshot", side_effect=lambda label: {"label": label, "available": False}) as snapshots, \
                     patch.object(Path, "read_text", side_effect=[json.dumps(source), json.dumps(baseline)]), \
                     redirect_stdout(io.StringIO()):
                    if fail_cleanup:
                        with self.assertRaises(SystemExit):
                            main(argv)
                    else:
                        main(argv)
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertTrue(backend.closed)
                self.assertEqual(len(report["cases"]), 4)
                self.assertEqual("summary" in report, not fail_cleanup)
                self.assertEqual(report["status"], "invalid_diagnostic" if fail_cleanup else SUCCESS)
                self.assertEqual(snapshots.call_count, 5 if context_mode == "fixed" else 4)
                if context_mode == "fixed":
                    resources = report["fixed_context_resources_after_measurement"]
                    self.assertTrue(resources["same_engine_object"])
                    self.assertTrue(resources["distinct_context_objects"])
                    self.assertEqual("fixed_context_validation" in report, not fail_cleanup)
                    self.assertEqual("dispatch_summary_ms" in report, not fail_cleanup)


if __name__ == "__main__":
    unittest.main()
