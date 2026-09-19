"""Synchronized end-to-end benchmark; model loading and warmup are reported separately."""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import psutil

from .cases import workload
from .common import compute_plan, environment, save, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_dir", type=Path)
    p.add_argument("--backend", choices=["coreml", "mlx"], default="coreml")
    p.add_argument("--compute-units", choices=["all", "cpu", "cpu_gpu", "cpu_ne"], default="all")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--questions", type=int, nargs="+", default=[1, 3, 10])
    p.add_argument("--long", action="store_true")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.iterations < 1 or a.warmup < 1 or any(q < 1 for q in a.questions):
        p.error("iterations, warmup, and questions must be positive")
    start = time.perf_counter()
    if a.backend == "coreml":
        from laya_coreml import Agent

        agent = Agent(a.model_dir, compute_units=a.compute_units)
    else:
        from laya_mlx import Agent

        agent = Agent(a.model_dir, dtype="float16", batch_size=16)
    loading = time.perf_counter() - start
    report = {
        "environment": environment(),
        "backend": a.backend,
        "compute_units": a.compute_units if a.backend == "coreml" else "gpu",
        "model_dir": str(a.model_dir),
        "load_seconds": loading,
        "measurement": "predict wall time: prompts, tokenization, arrays, synchronous inference, calibration and formatting; excludes load/warmup",
        "export": agent.manifest if a.backend == "coreml" else None,
        "workloads": [],
    }
    for count in a.questions:
        state, questions = workload(count, long=a.long)
        start = time.perf_counter()
        expected = agent.predict(state, questions)
        first_ms = (time.perf_counter() - start) * 1000
        for _ in range(a.warmup - 1):
            agent.predict(state, questions)
        gc.collect()
        rss_before = psutil.Process().memory_info().rss
        samples, stable = [], True
        for _ in range(a.iterations):
            start = time.perf_counter()
            result = agent.predict(state, questions)
            samples.append((time.perf_counter() - start) * 1000)
            stable &= result == expected
            # This call is synchronous: Core ML returns arrays; MLX Agent.forward calls mx.eval.
            if not all(
                np.isfinite(v)
                for ans in result["answers"].values()
                for v in ans.get("probabilities", {}).values()
            ):
                raise FloatingPointError("Non-finite output")
        items, _ = agent.prepare(state, questions)
        entry = {
            "questions": count,
            "long": a.long,
            "unrounded_token_lengths": [len(i["ids"]) for i in items],
            "first_prediction_ms": first_ms,
            "warmup_calls": a.warmup,
            **stats(samples),
            "questions_per_second": count * 1000 / float(np.mean(samples)),
            "identical_rounded_results": stable,
            "rss_before": rss_before,
            "rss_after": psutil.Process().memory_info().rss,
            "latency_ms": samples,
            "example_result": expected,
        }
        report["workloads"].append(entry)
        save(a.output, report)
        print(
            f"{a.backend}/{a.compute_units}/{count}: P50 {entry['p50_ms']:.3f} ms, P95 {entry['p95_ms']:.3f} ms, stable={stable}",
            flush=True,
        )
    if a.plan and a.backend == "coreml":
        try:
            report["compute_plan"] = compute_plan(agent)
        except Exception as error:
            report["compute_plan_error"] = f"{type(error).__name__}: {error}"
    save(a.output, report)


if __name__ == "__main__":
    main()
