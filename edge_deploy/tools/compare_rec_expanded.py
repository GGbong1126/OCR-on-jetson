"""Old recorded crops plus scan-discovered over-profile crops: strict CPU/TRT rec regression.

No detection, added padding, page policy, benchmark or recognized text saved.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from compare_rec import compare_rec_outputs, verify_source
from ocr_det import read_image, sha256_file
from ocr_rec import Recognizer, TensorRTRecognizer, crop_text_region, preprocess_rec


def inside(shape, profile):
    return len(shape) == 4 and all(lo <= n <= hi for lo, n, hi in zip(profile[0], shape, profile[2]))


def validated_scan_rows(scan, baseline, baseline_sha, model_sha, dictionary_sha):
    if (scan.get("status") != "completed_inventory_not_rec_validated"
            or scan.get("rec_baseline_sha256") != baseline_sha
            or baseline.get("status") != "completed_consistent_not_accuracy_validated"
            or baseline.get("reference_rec_onnx_sha256") != model_sha
            or baseline.get("dictionary_sha256") != dictionary_sha):
        raise ValueError("Scan/baseline/model/dictionary provenance mismatch")
    rec = baseline["recognizer"]
    profile = rec["profile"]
    if (profile != [[1,3,48,w] for w in (32,160,1024)]
            or scan.get("declared_rec_profile") != profile
            or scan.get("declared_rec_engine_sha256") != rec["engine_sha256"]):
        raise ValueError("Unexpected original recognition profile/engine")
    images = scan.get("images", [])
    if (not images or len(images) != scan.get("expected_image_count")
            or len({im["path"] for im in images}) != len(images)):
        raise ValueError("Incomplete/duplicated scan images")
    selected = []
    for image in images:
        ort_rows = image["branches"]["ort"]["records"]
        trt_rows = image["branches"]["trt"]["records"]
        if (len(ort_rows) != len(trt_rows) or not image["detection"]["ordered_boxes_exactly_equal"]
                or [r["index"] for r in trt_rows] != list(range(len(trt_rows)))):
            raise ValueError("Need identical ordered detector crop sets for this targeted regression")
        for a, b in zip(ort_rows, trt_rows):
            if (a.get("status") != "inventoried" or b.get("status") != "inventoried"
                    or any(a.get(k) != b.get(k) for k in ("index", "box", "input_shape", "input_sha256"))):
                raise ValueError("Invalid or differing detector crops")
            shape = b["input_shape"]
            if len(shape) != 4 or shape[:3] != [1,3,48] or any(type(n) is not int or n < 1 for n in shape):
                raise ValueError("Invalid recorded recognition shape")
            if b.get("within_declared_rec_profile") != inside(shape, profile):
                raise ValueError("Incorrect source profile flag")
            if a.get("legacy_width_cap_applied") or b.get("legacy_width_cap_applied"):
                raise ValueError("Legacy width cap is not accepted")
            if not inside(shape, profile):
                if shape[3] <= profile[2][3]:
                    raise ValueError("Unsupported crop: not a simple upper-width overflow")
            selected.append({"path": image["path"], "image_sha256": image["image_sha256"], **b})
    return selected


def select_overflow(scan, baseline, baseline_sha, model_sha, dictionary_sha):
    rows = validated_scan_rows(scan, baseline, baseline_sha, model_sha, dictionary_sha)
    selected = [r for r in rows if not inside(r["input_shape"], baseline["recognizer"]["profile"])]
    if not selected or len(selected) > 32:
        raise ValueError("Expected 1..32 targeted overflow crops, not a full-dataset run")
    return selected


def make_sample(image, row, identity, expected_text=None, expected_hash=None):
    crop = crop_text_region(image, row["box"])
    if crop is None:
        raise ValueError("Empty crop")
    tensor = np.ascontiguousarray(preprocess_rec(crop))
    sha = hashlib.sha256(tensor.tobytes(order="C")).hexdigest()
    if list(tensor.shape) != row["input_shape"] or (expected_hash and sha != expected_hash):
        raise ValueError("Crop tensor differs from recorded input")
    return {"id": identity, "index": row["index"], "box": row["box"], "input_sha256": sha,
            "input_shape": list(tensor.shape), "tensor": tensor, "expected_text": expected_text}


def prepare_samples(source, image_path, meta, root, selected):
    sha = sha256_file(image_path)
    rows = verify_source(source, sha, meta, "reference")
    image = read_image(image_path)
    samples = [make_sample(image, row, f"reference:{Path(image_path).name}#{row['index']}", row["text"])
               for row in rows]
    for row in selected:
        relative = Path(row["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("Scan path escapes image directory")
        if sha256_file(path) != row["image_sha256"]:
            raise ValueError("Scan image changed")
        sample = make_sample(read_image(path), row, f"scan:{row['path']}#{row['index']}",
                             expected_hash=row["input_sha256"])
        sample["image_sha256"] = row["image_sha256"]
        samples.append(sample)
    return samples


def compare_samples(cpu, gpu, samples, records):
    if cpu.characters != gpu.characters or any(not inside(s["input_shape"], gpu.profile) for s in samples):
        raise ValueError("Candidate dictionary/profile cannot cover all selected tensors")
    for position, sample in enumerate(samples):
        row = {k: v for k, v in sample.items() if k not in ("tensor", "expected_text")}
        try:
            reference = cpu.run_tensor(sample["tensor"])
            candidate = gpu.run_tensor(sample["tensor"])
            result = compare_rec_outputs(reference, candidate, cpu.characters)
            if result["status"] == "finite_outputs_compared":
                ref_text, cand_text = result.pop("reference_text"), result.pop("candidate_text")
                result.update(reference_text_length=len(ref_text), candidate_text_length=len(cand_text),
                              reference_empty=not ref_text, candidate_empty=not cand_text,
                              reference_matches_recorded=(ref_text == sample["expected_text"]
                                  if sample["expected_text"] is not None else None))
            row.update(result)
            del reference, candidate
        except Exception as error:
            row.update(status="error", error=f"{type(error).__name__}: {error}")
        records.append(row)
        if (position+1) % 10 == 0 or position+1 == len(samples):
            print(f"Compared {position+1}/{len(samples)} original-width crops", flush=True)


def summary(records, expected):
    valid = [r for r in records if r["status"] == "finite_outputs_compared"]
    bad = [r["id"] for r in records if r["status"] != "finite_outputs_compared"]
    changed = [r["id"] for r in valid if not r["text_equal"]]
    drift = [r["id"] for r in valid if r["reference_matches_recorded"] is False]
    status = ("invalid_comparison" if bad or len(records) != expected or not expected else
              "completed_with_text_changes_not_accepted" if changed or drift else
              "completed_strict_consistent_not_accuracy_validated")
    total = sum(r["elements"] for r in valid)
    return {"status": status, "expected_count": expected, "compared_count": len(records),
            "finite_count": len(valid), "invalid_ids": bad, "changed_text_ids": changed,
            "reference_drift_ids": drift, "text_equal_count": sum(r["text_equal"] for r in valid),
            "reference_empty_ids": [r["id"] for r in valid if r["reference_empty"]],
            "candidate_empty_ids": [r["id"] for r in valid if r["candidate_empty"]],
            "allclose_failed_ids": [r["id"] for r in valid if not r["allclose_rtol_1e_3_atol_1e_4"]],
            "element_weighted_mae": sum(r["mae"]*r["elements"] for r in valid)/total if total else None,
            "max_abs_error": max((r["max_abs_error"] for r in valid), default=None),
            "note": "Numeric tolerance is diagnostic. Strict text equality is not GT accuracy, including shared empty text."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("onnx", "engine", "dict", "images", "scan-report", "baseline-report",
                 "reference-report", "reference-image", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.output.exists() or args.output.is_symlink():
        parser.error("Need positive threads and a new output file")
    cv2.setNumThreads(1)
    gpu = None
    report = {"status": "started", "records": [], "expected_count": 0,
              "source_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in
                                ("compare_rec_expanded.py", "compare_rec.py", "ocr_rec.py", "trt_backend.py")},
              "note": "Same original-width tensor CPU then TRT; full strict CTC, no page policy or added padding. "
                      "No recognized text saved. Not a benchmark, not all 1569 crops or GT accuracy."}
    try:
        for name in ("onnx", "engine", "dict", "scan_report", "baseline_report", "reference_report", "reference_image"):
            report[name+"_sha256"] = sha256_file(getattr(args, name))
        load = lambda path: json.loads(path.read_text(encoding="utf-8"))
        baseline, scan, source = load(args.baseline_report), load(args.scan_report), load(args.reference_report)
        selected = select_overflow(scan, baseline, report["baseline_report_sha256"],
                                   report["onnx_sha256"], report["dict_sha256"])
        meta = {"model_sha256": report["onnx_sha256"], "dictionary_sha256": report["dict_sha256"]}
        samples = prepare_samples(source, args.reference_image, meta, args.images.resolve(), selected)
        report.update(expected_count=len(samples), overflow_count=len(selected),
                      reference_count=len(samples)-len(selected))
        cpu = Recognizer(args.onnx, args.dict, args.threads)
        report["reference_metadata"] = cpu.metadata()
        gpu = TensorRTRecognizer(args.engine, args.dict)
        report["candidate_metadata"] = gpu.metadata()
        compare_samples(cpu, gpu, samples, report["records"])
        report["status"] = "comparison_finished"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="invalid_comparison", error=f"{type(error).__name__}: {error}")
    finally:
        if gpu is not None:
            try:
                gpu.close()
            except Exception as error:
                report.update(status="invalid_comparison", cleanup_error=f"{type(error).__name__}: {error}")
    report["summary"] = summary(report["records"], report["expected_count"])
    if report["status"] == "comparison_finished":
        report["status"] = report["summary"]["status"]
    else:
        report["summary"]["status"] = "invalid_comparison"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "summary": report["summary"],
                      "error": report.get("error"), "cleanup_error": report.get("cleanup_error")}, indent=2))
    print(f"Saved: {args.output.resolve()}")
    return 0 if report["status"] == "completed_strict_consistent_not_accuracy_validated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
