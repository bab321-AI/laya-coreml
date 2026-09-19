"""Compare Core ML against pinned upstream FP32 outputs and exercise repeat calls."""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import psutil

from laya_coreml import Agent
from laya_coreml.inputs import collate_items

from .common import calibrated, environment, save, softmax


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_dir", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--compute-units", default="cpu_gpu")
    p.add_argument("--reference", type=Path, default=Path("benchmarks/results/reference.json"))
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--allow-unvalidated-gpu", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    reference = json.loads(a.reference.read_text())["models"][a.name]
    started = time.perf_counter()
    agent = Agent(
        a.model_dir, compute_units=a.compute_units, allow_unvalidated_gpu=a.allow_unvalidated_gpu
    )
    load_seconds = time.perf_counter() - started
    if agent.manifest["source_weights_sha256"] != reference["source_weights_sha256"]:
        raise ValueError("Export and golden reference use different weights")
    report = {
        "environment": environment(),
        "model": a.name,
        "export": agent.manifest,
        "reference_sha256": hashlib.sha256(a.reference.read_bytes()).hexdigest(),
        "compute_units": a.compute_units,
        "load_seconds": load_seconds,
        "cases": [],
    }
    total = agreements = 0
    max_error = max_action_error = 0.0
    expected = []
    for entry in reference["cases"]:
        items, _ = agent.prepare(entry["state"], entry["questions"])
        if items != entry["items"]:
            raise AssertionError(f"Input tokens differ: {entry['name']}")
        if max(len(i["ids"]) for i in items) > agent.shape["max_length"]:
            report["cases"].append(
                {"name": entry["name"], "skipped": "outside exported length limit"}
            )
            continue
        case = {
            "name": entry["name"],
            "questions": len(items),
            "tokens_equal": True,
            "argmax_agreements": 0,
            "probability_max_abs_error": 0.0,
            "action_probability_max_abs_error": 0.0,
            "logits_max_abs_error": 0.0,
        }
        for start in range(0, len(items), agent.batch_size):
            chunk = items[start : start + agent.batch_size]
            batch = collate_items(chunk, agent.tok.pad_token_id, shape=agent.shape)
            logits, action = agent.forward(batch)
            if not np.isfinite(logits).all() or not np.isfinite(action).all():
                raise FloatingPointError(f"Non-finite outputs in {entry['name']}")
            for row, item in enumerate(chunk):
                ref_z, ref_action = (
                    entry["logits"][start + row],
                    entry["action_logits"][start + row],
                )
                prob, ref_prob = (
                    calibrated(agent, logits[row], item),
                    calibrated(agent, ref_z, item),
                )
                error = float(np.max(np.abs(prob - ref_prob)))
                act_error = float(np.max(np.abs(softmax(action[row]) - softmax(ref_action))))
                agree = int(prob.argmax() == ref_prob.argmax())
                case["argmax_agreements"] += agree
                case["probability_max_abs_error"] = max(case["probability_max_abs_error"], error)
                case["action_probability_max_abs_error"] = max(
                    case["action_probability_max_abs_error"], act_error
                )
                k = len(item["markers"])
                case["logits_max_abs_error"] = max(
                    case["logits_max_abs_error"],
                    float(np.max(np.abs(logits[row, :k] - np.asarray(ref_z[:k])))),
                )
                max_error, max_action_error = (
                    max(max_error, error),
                    max(max_action_error, act_error),
                )
                total, agreements = total + 1, agreements + agree
        result = agent.predict(entry["state"], entry["questions"])
        if result["usage"] != entry["result"]["usage"]:
            raise AssertionError("Token accounting mismatch")
        case["public_result_equal"] = result == entry["result"]
        report["cases"].append(case)
        expected.append((entry, result))
        print(
            f"{a.name}/{a.compute_units}/{entry['name']}: {case['argmax_agreements']}/{len(items)}, drift {case['probability_max_abs_error']:.6f}",
            flush=True,
        )
        save(a.output, report)
    gc.collect()
    process = psutil.Process()
    rss_before = process.memory_info().rss
    memory_samples = [rss_before]
    stable = True
    started = time.perf_counter()
    for index in range(a.repeats):
        entry, expected_result = expected[index % len(expected)]
        result = agent.predict(entry["state"], entry["questions"])
        stable &= result == expected_result
        if (index + 1) % 10 == 0:
            memory_samples.append(process.memory_info().rss)
    elapsed = time.perf_counter() - started
    gc.collect()
    tolerance = 1e-4 if agent.manifest["precision"] == "float32" else 0.02
    report.update(
        questions=total,
        argmax_agreements=agreements,
        probability_max_abs_error=max_error,
        action_probability_max_abs_error=max_action_error,
        probability_tolerance=tolerance,
        passed=agreements == total
        and max_error <= tolerance
        and max_action_error <= tolerance
        and stable,
        stability={
            "calls": a.repeats,
            "seconds": elapsed,
            "rounded_public_results_identical": stable,
            "rss_before": rss_before,
            "rss_after": process.memory_info().rss,
            "rss_samples": memory_samples,
            "memory_note": "Process RSS includes Core ML/framework caches; not an active-tensor allocator measurement",
        },
    )
    save(a.output, report)
    print(
        f"FINAL: {agreements}/{total}, probability drift {max_error:.6g}, stable={stable}",
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
