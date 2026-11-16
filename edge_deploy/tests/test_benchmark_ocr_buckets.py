"""CPU/mock live-detection and bounded-rec integration tests, not GPU results."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from test_benchmark_ocr import FakeDetector
from test_profile_rec_shapes import runner, FakeContext
from benchmark_ocr_buckets import run_pipeline, measure_pipeline, close_pipeline, main, SMOKE_SUCCESS
from compare_rec_buckets import make_plan, SUCCESS, POLICY_SUCCESS
from trt_host_profile import HostProfiler
from trt_bucket_contexts import BucketContexts


class LiveBucketTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((48, 160, 3), np.uint8)
        self.rows = [{"index": i, "box": [[0, 0], [w, 0], [w, 48], [0, 48]],
                      "input_shape": [1, 3, 48, w], "text": "A", "status": "ok"}
                     for i, w in enumerate((32, 80, 45))]
        self.det = FakeDetector([r["box"] for r in self.rows])
        self.owner = runner()
        self.owner.engine.create_execution_context = lambda: FakeContext(self.owner.trace)
        self.profile = [[1, 3, 48, 32], [1, 3, 48, 64], [1, 3, 48, 1024]]
        self.plan = make_plan(self.rows, [64, 128], self.profile)
        self.pool = None
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.owner.cuda.fail = None
        self.assertEqual(close_pipeline(self.pool, self.owner, self.det), [])

    def profiler(self, fixed=False):
        if not fixed:
            return HostProfiler(self.owner)
        self.pool = BucketContexts(self.owner, [64, 128])
        self.pool.initialize()
        return self.pool

    def measure(self, mode, stage, report, roles=None):
        snap = Mock()
        with redirect_stdout(io.StringIO()):
            measure_pipeline(self.det, self.profiler(mode == "padded-fixed"), ["", "A", " "],
                             self.image, self.rows, self.plan, roles or {}, self.det.shape,
                             mode, stage, 1, 2 if stage == "benchmark" else 1, report, snap)
        return snap

    def test_smoke_runs_live_three_times_without_benchmark_summary(self):
        report = {}
        self.measure("padded-fixed", "smoke", report)
        self.assertEqual(self.det.calls, 3)
        self.assertEqual(report["status"], SMOKE_SUCCESS)
        self.assertNotIn("summary_ms", report)
        self.assertEqual(sum(report["measured_rec_host_totals"].values()), 0)
        self.assertEqual(report["validation"]["measured_rec_calls"], 3)

    def test_benchmark_live_detection_warmup_exclusion_and_time_scope(self):
        report = {}
        snap = self.measure("original-single", "benchmark", report)
        self.assertEqual(self.det.calls, 4)
        self.assertEqual(report["status"], SUCCESS)
        self.assertEqual(report["summary_ms"]["pipeline_wall_ms"]["count"], 2)
        self.assertEqual([a.args[0] for a in snap.call_args_list],
                         ["after_validation", "after_warmup", "after_measured"])
        for c in report["cycles"]:
            self.assertEqual([r["index"] for r in c["records"]], [0, 1, 2])
            self.assertGreaterEqual(c["timings"]["pipeline_wall_ms"], c["timings"]["rec_cycle_wall_ms"])
            self.assertEqual(c["changed_box_indices"], [])

    def test_geometry_order_count_shape_reject_before_rec(self):
        for kind in ("box", "order", "count", "shape", "nan"):
            det = FakeDetector([r["box"] for r in self.rows])
            if kind == "box": det.boxes[0][0][0] += 1
            if kind == "order": det.boxes.reverse()
            if kind == "count": det.boxes.pop()
            if kind == "shape": det.shape = [1, 3, 32, 32]
            if kind == "nan": det.boxes[0][0][0] = float("nan")
            profiler, sample = Mock(), {}
            with self.assertRaises(ValueError):
                run_pipeline(det, profiler, [], self.image, self.rows, self.plan, {0: "page_number"},
                             self.det.shape, "padded-fixed", "validation", 0, sample)
            profiler.run_tensor.assert_not_called()
            self.assertEqual(sample["status"], "error")
            json.dumps(sample, allow_nan=False)

    def test_new_live_box_objects_are_used_for_cropping(self):
        seen = []
        from benchmark_rec_buckets import run_cycle as real_cycle
        def capture(*args):
            live_rows = args[3]
            seen.extend(live_rows)
            self.assertTrue(all(a['box'] is not b['box'] for a, b in zip(live_rows, self.rows)))
            return real_cycle(*args)
        with patch("benchmark_ocr_buckets.run_cycle", side_effect=capture):
            self.measure("original-single", "smoke", {})
        self.assertEqual(len(seen), 9)

    def test_page_policy_raw_difference_and_empty_retained(self):
        self.rows[0]["text"] = "- 1 -"
        self.rows[1]["text"] = ""
        texts = iter(["-1-", "", "A"]*4)
        with patch("benchmark_rec_buckets.decode_ctc", side_effect=lambda *args: (next(texts), .9)):
            report = {}
            self.measure("padded-fixed", "benchmark", report, {0: "page_number"})
        self.assertEqual(report["status"], POLICY_SUCCESS)
        self.assertEqual(report["validation"]["strict_changed_indices"], [0])
        self.assertEqual(report["cycles"][0]["empty_indices"], [1])

    def test_text_nonfinite_execution_interrupt_preserve_partial(self):
        for failure in (ValueError("nonfinite"), RuntimeError("execution"), KeyboardInterrupt()):
            report = {}
            with patch("benchmark_rec_buckets.decode_ctc", side_effect=failure), self.assertRaises(type(failure)):
                self.measure("original-single", "smoke", report)
            self.assertEqual(report["validation_cycles"][0]["status"], "error")
            self.assertEqual(len(report["validation_cycles"][0]["records"]), 1)
            self.assertNotIn("summary_ms", report)
        self.rows[0]["text"] = "B"
        with self.assertRaises(ValueError):
            self.measure("original-single", "smoke", {})

    def test_later_detection_drift_retains_validation_no_success(self):
        original = self.det.predict
        def predict(image):
            if self.det.calls == 1: self.det.boxes[0][0][0] += 1
            return original(image)
        self.det.predict = predict
        report = {}
        with self.assertRaises(ValueError):
            self.measure("original-single", "benchmark", report)
        self.assertEqual(report["validation_cycles"][0]["status"], "consistent")
        self.assertEqual(report["warmup_cycles"][0]["records"], [])
        self.assertNotIn("summary_ms", report)

    def test_child_cleanup_failure_preserves_owner_and_closes_detector(self):
        pool, rec, det = Mock(), Mock(), Mock()
        pool.close.side_effect = RuntimeError("child sync failed")
        errors = close_pipeline(pool, rec, det)
        rec.close.assert_not_called()
        det.close.assert_called_once()
        self.assertEqual(len(errors), 1)
        det.close.side_effect = RuntimeError("det cleanup")
        self.assertEqual(len(close_pipeline(pool, rec, det)), 2)

    def test_main_smoke_benchmark_failure_cleanup_and_overwrite(self):
        for case in ("smoke", "benchmark", "source", "rec_load", "child_cleanup", "det_cleanup"):
            with tempfile.TemporaryDirectory() as d:
                out = Path(d)/"report.json"
                argv = ["--mode", "padded-fixed", "--stage", "benchmark" if case == "benchmark" else "smoke",
                        "--det-engine", "det", "--rec-engine", "rec", "--rec-onnx", "onnx", "--dict", "dict",
                        "--image", "image", "--source-report", "source", "--baseline-report", "baseline",
                        "--acceptance-report", "accepted", "--field-annotations", "fields",
                        "--buckets", "64", "128", "--output", str(out), "--memory"]
                if case == "benchmark": argv += ["--warmup", "1", "--repeat", "2"]
                with patch.object(Path, "read_text", return_value='{"reference":{}}'), \
                     patch("benchmark_ocr_buckets.sha256_file", side_effect=str), \
                     patch("benchmark_ocr_buckets.prepare_rows", return_value=self.rows), \
                     patch("benchmark_ocr_buckets.read_image", return_value=self.image), \
                     patch("benchmark_ocr_buckets.verify_baseline"), \
                     patch("benchmark_ocr_buckets.verify_detector", return_value=self.det.shape), \
                     patch("benchmark_ocr_buckets.validate_annotations", return_value={}), \
                     patch("benchmark_ocr_buckets.verify_acceptance", side_effect=ValueError("bad source") if case=='source' else None), \
                     patch("benchmark_ocr_buckets.verify_inputs", return_value=[]), \
                     patch("benchmark_ocr_buckets.create_detector") as det_factory, \
                     patch("benchmark_ocr_buckets.TensorRTRecognizer") as rec_factory, \
                     patch("benchmark_ocr_buckets.BucketContexts") as pool_factory, \
                     patch("benchmark_ocr_buckets.memory_snapshot", return_value={"available":False,"reason":"mock unavailable"}), \
                     redirect_stdout(io.StringIO()):
                    detector = FakeDetector([r["box"] for r in self.rows])
                    det_factory.return_value = detector
                    rec = rec_factory.return_value
                    rec.profile, rec.characters = self.profile, ["", "A", " "]
                    rec.metadata.return_value = {}
                    pool = pool_factory.return_value
                    pool.run_tensor.side_effect = HostProfiler(self.owner).run_tensor
                    # This isolated entry-point test mocks the already separately tested measurement.
                    def measured(*args):
                        report = args[-2]
                        report.update(status=SUCCESS if case=='benchmark' else SMOKE_SUCCESS,
                                      validation={"all_policy_equal":True}, summary_ms={"mock":True}, measured_rec_host_totals={})
                        if case!='benchmark': report.pop('summary_ms')
                    pool.resources.return_value = {}
                    if case=='rec_load': rec_factory.side_effect=RuntimeError("out of memory")
                    if case=='child_cleanup': pool.close.side_effect=RuntimeError("sync")
                    if case=='det_cleanup': detector.close=Mock(side_effect=RuntimeError("det close"))
                    with patch("benchmark_ocr_buckets.measure_pipeline", side_effect=measured):
                        if case in ('smoke','benchmark'): main(argv)
                        else:
                            with self.assertRaises(SystemExit) as error: main(argv)
                            self.assertEqual(error.exception.code,2)
                    if case=='source': det_factory.assert_not_called()
                    if case=='rec_load': self.assertTrue(detector.closed)
                    if case=='child_cleanup': rec.close.assert_not_called()
                data=out.read_bytes()
                report=json.loads(data)
                self.assertEqual('validation' in report,case in ('smoke','benchmark'))
                self.assertEqual('summary_ms' in report,case=='benchmark')
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): main(argv)
                self.assertEqual(data,out.read_bytes())

    def test_smoke_rejects_benchmark_counts(self):
        argv=[]
        for flag in ('det-engine','rec-engine','rec-onnx','dict','image','source-report','baseline-report','acceptance-report','field-annotations'):
            argv += ['--'+flag, 'unused']
        argv += ['--mode','padded-fixed','--output','unused.json','--warmup','2']
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): main(argv)


if __name__ == "__main__":
    unittest.main()
