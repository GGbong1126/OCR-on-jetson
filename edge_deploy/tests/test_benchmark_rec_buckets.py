"""CPU/mock tests; no claimed CUDA performance."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
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
from test_profile_rec_shapes import runner, tensor, FakeContext
from trt_bucket_contexts import BucketContexts
from trt_host_profile import HostProfiler
from benchmark_rec_buckets import measure, verify_acceptance, verify_inputs, main, SUCCESS, POLICY_SUCCESS
from compare_rec_buckets import make_plan


class BucketBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.owner = runner()
        self.owner.engine.create_execution_context = lambda: FakeContext(self.owner.trace)
        self.pool = None
        self.addCleanup(self.cleanup)
        self.image = np.zeros((48, 160, 3), np.uint8)
        self.rows = [{"index": i, "box": [[0, 0], [w, 0], [w, 48], [0, 48]],
                      "input_shape": [1, 3, 48, w], "text": "A", "status": "ok"}
                     for i, w in enumerate((32, 80, 45))]
        self.plan = make_plan(self.rows, [64, 128], [[1, 3, 48, 32], [1, 3, 48, 64], [1, 3, 48, 1024]])

    def cleanup(self):
        self.owner.cuda.fail = None
        if self.pool is not None:
            self.pool.close()
        self.owner.close()

    def make_pool(self, widths=(64, 128)):
        self.pool = BucketContexts(self.owner, widths)
        self.pool.initialize()
        return self.pool

    def test_six_contexts_share_engine_stream_and_stabilize(self):
        pool = self.make_pool((64, 128, 160, 256, 512, 960))
        self.assertEqual(len(pool.runners), 6)
        self.assertEqual(len({id(r.context) for r in pool.runners.values()}), 6)
        self.assertTrue(all(r.engine is self.owner.engine and r.stream is self.owner.stream for r in pool.runners.values()))
        for cycle in range(2):
            for w in reversed(pool.widths):
                _, r = pool.run_tensor(tensor(w))
                self.assertEqual(r["context_width"], w)
                if cycle:
                    self.assertFalse(r["shape_changed"])
                    self.assertEqual(r["malloc_count"] + r["free_count"], 0)
        self.assertEqual(pool.resources()["context_count"], 6)

    def test_pool_limits_routes_and_external_shape_changes(self):
        for widths in ([], [32, 32], [64, 32], [31], [1025], list(range(32, 40))):
            with self.assertRaises(ValueError):
                BucketContexts(self.owner, widths)
        with self.assertRaises(ValueError):
            BucketContexts(self.owner, [64, 128], max_contexts=1)
        pool = self.make_pool()
        with self.assertRaises(ValueError):
            pool.run_tensor(tensor(45))
        self.owner.current_shape = (1, 3, 48, 45)
        with self.assertRaises(RuntimeError):
            pool.run_tensor(tensor(64))

    def test_partial_creation_and_cleanup_keep_owner_alive(self):
        count = []
        def create():
            count.append(1)
            return FakeContext(self.owner.trace) if len(count) == 1 else None
        self.owner.engine.create_execution_context = create
        self.pool = BucketContexts(self.owner, [64, 128, 256])
        with self.assertRaises(RuntimeError):
            self.pool.initialize()
        self.assertEqual(len(self.pool.runners), 2)
        self.pool.close()
        self.assertFalse(self.owner.closed)

    def test_failed_child_sync_retains_resources_then_retry(self):
        pool = self.make_pool()
        pool.run_tensor(tensor(128))
        self.owner.cuda.fail = "cudaStreamSynchronize"
        with self.assertRaises(RuntimeError):
            pool.close()
        self.assertFalse(self.owner.closed)
        self.assertFalse(pool.closed)
        self.assertIsNotNone(pool.runners[128].context)
        self.owner.cuda.fail = None
        pool.close()
        pool.close()

    def run_measure(self, profiler, mode, report, roles=None):
        with redirect_stdout(io.StringIO()):
            measure(profiler, ["", "A", " "], self.image, self.rows, self.plan,
                    roles or {}, mode, 1, 2, report, lambda label: None)

    def test_full_sequence_order_warmup_separate_and_fixed_no_alloc(self):
        report = {}
        self.run_measure(self.make_pool(), "padded-fixed", report)
        self.assertEqual(report["status"], SUCCESS)
        self.assertEqual([r["index"] for r in report["records"]], [0, 1, 2]*2)
        self.assertEqual(len(report["warmup_records"]), 3)
        self.assertEqual(report["cycle_summary_ms"]["cycle_wall_ms"]["count"], 2)
        self.assertEqual(sum(report["measured_host_totals"].values()), 0)
        self.assertTrue(all(c["cycle_wall_ms"] >= c["crop_rec_ms"] for c in report["cycles"]))

    def test_original_and_padded_single_use_expected_shapes(self):
        for mode, widths in (("original-single", [32, 80, 45]), ("padded-single", [64, 128, 64])):
            report = {}
            self.run_measure(HostProfiler(self.owner), mode, report)
            self.assertEqual([r["input_shape"][3] for r in report["records"]], widths*2)
            self.assertGreater(report["measured_host_totals"]["shape_changed"], 0)

    def test_page_policy_preserves_raw_difference_and_original_strict(self):
        self.rows[0]["text"] = "- 1 -"
        for mode in ("padded-single", "original-single"):
            texts = iter(["-1-", "A", "A"]*3)
            report = {}
            with patch("benchmark_rec_buckets.decode_ctc", side_effect=lambda *a: (next(texts), .9)):
                if mode == "original-single":
                    with self.assertRaises(ValueError):
                        self.run_measure(HostProfiler(self.owner), mode, report, {0: "page_number"})
                    self.assertNotIn("cycle_summary_ms", report)
                else:
                    self.run_measure(HostProfiler(self.owner), mode, report, {0: "page_number"})
                    self.assertEqual(report["status"], POLICY_SUCCESS)
                    self.assertEqual(report["validation"]["strict_changed_indices"], [0])

    def test_empty_preserved_and_invalid_output_interrupt_stop(self):
        self.rows[0]["text"] = ""
        texts = iter(["", "A", "A"]*3)
        with patch("benchmark_rec_buckets.decode_ctc", side_effect=lambda *a: (next(texts), .9)):
            report = {}
            self.run_measure(HostProfiler(self.owner), "padded-single", report)
            self.assertEqual(report["cycles"][0]["empty_indices"], [0])
        for failure in (ValueError("nonfinite"), KeyboardInterrupt()):
            with patch("benchmark_rec_buckets.decode_ctc", side_effect=failure):
                report = {}
                with self.assertRaises(type(failure)):
                    self.run_measure(HostProfiler(self.owner), "padded-single", report)
                self.assertEqual(report["warmup_records"][0]["status"], "error")
                self.assertNotIn("cycle_summary_ms", report)

    def test_measured_wrong_fixed_invariant_invalidates(self):
        with self.assertRaisesRegex(ValueError, "invariant"):
            self.run_measure(HostProfiler(self.owner), "padded-fixed", {})

    def test_acceptance_and_tensor_hash_checks(self):
        accepted = {"status": POLICY_SUCCESS, "image_sha256": "image", "engine_sha256": "engine",
                    "field_annotations_sha256": "fields", "buckets": [64, 128], "records": []}
        from ocr_rec import preprocess_rec, crop_text_region
        from compare_rec_buckets import pad_white
        for row, plan in zip(self.rows, self.plan):
            original = preprocess_rec(crop_text_region(self.image, row["box"]))
            padded = pad_white(original, plan["padded_input_shape"][3])
            accepted["records"].append({**plan, "status": "finite_pair_compared", "original_equals_source": True,
                "policy_padded_equals_original": True, "policy_padded_equals_source": True,
                "original": {"all_finite": True}, "padded": {"all_finite": True},
                "original_tensor_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
                "padded_tensor_sha256": hashlib.sha256(padded.tobytes()).hexdigest()})
        def check(a):
            verify_acceptance(a, {"image_sha256": "image"}, "engine", "fields", [64, 128], self.rows)
            verify_inputs(self.image, self.rows, self.plan, a)
        check(accepted)
        for key in ("status", "image_sha256", "engine_sha256", "field_annotations_sha256", "buckets"):
            bad = deepcopy(accepted)
            bad[key] = "wrong"
            with self.assertRaises(ValueError):
                check(bad)
        for key in ("original_equals_source", "policy_padded_equals_original", "padded_tensor_sha256"):
            bad = deepcopy(accepted)
            bad["records"][0][key] = False
            with self.assertRaises(ValueError):
                check(bad)

    def test_main_success_failure_cleanup_and_no_overwrite(self):
        for mode in ("ok", "fixed", "provenance", "cleanup", "interrupt"):
            self.owner.close()
            self.owner = runner()
            self.owner.engine.create_execution_context = lambda: FakeContext(self.owner.trace)
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / "report.json"
                argv = ["--mode", "padded-fixed" if mode == "fixed" else "padded-single", "--engine", "engine", "--onnx", "onnx", "--dict", "dict",
                        "--image", "image", "--source-report", "source", "--baseline-report", "baseline",
                        "--acceptance-report", "acceptance", "--field-annotations", "fields", "--buckets", "64", "128",
                        "--warmup", "1", "--repeat", "1", "--output", str(out)]
                with patch("benchmark_rec_buckets.TensorRTRecognizer") as factory, \
                     patch.object(Path, "read_text", return_value="{}"), \
                     patch("benchmark_rec_buckets.sha256_file", side_effect=str), \
                     patch("benchmark_rec_buckets.read_image", return_value=self.image), \
                     patch("benchmark_rec_buckets.prepare_rows", return_value=self.rows), \
                     patch("benchmark_rec_buckets.verify_baseline"), \
                     patch("benchmark_rec_buckets.validate_annotations", return_value={}), \
                     patch("benchmark_rec_buckets.verify_acceptance", side_effect=ValueError("provenance") if mode == "provenance" else None), \
                     patch("benchmark_rec_buckets.verify_inputs", return_value=[]), redirect_stdout(io.StringIO()):
                    rec = factory.return_value
                    rec.runner = self.owner
                    rec.characters = ["", "A", " "]
                    rec.profile = [[1, 3, 48, 32], [1, 3, 48, 64], [1, 3, 48, 1024]]
                    rec.metadata.return_value = {"backend": "mock"}
                    rec.close.side_effect = self.owner.close
                    if mode == "cleanup":
                        rec.close.side_effect = RuntimeError("cleanup")
                    elif mode == "interrupt":
                        factory.side_effect = KeyboardInterrupt
                    if mode in ("ok", "fixed"):
                        main(argv)
                    else:
                        with self.assertRaises(SystemExit) as error:
                            main(argv)
                        self.assertEqual(error.exception.code, 2)
                    self.assertEqual(factory.call_count, int(mode != "provenance"))
                data = out.read_bytes()
                report = json.loads(data)
                self.assertEqual("cycle_summary_ms" in report, mode in ("ok", "fixed"))
                self.assertEqual("validation" in report, mode in ("ok", "fixed"))
                if mode == "fixed":
                    self.assertEqual(report["context_count"], 2)
                    self.assertEqual(sum(report["measured_host_totals"].values()), 0)
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(argv)
                self.assertEqual(out.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
