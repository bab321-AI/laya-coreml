"""Alternating, sustained inference with unprivileged SMC power telemetry.

System power is the SMC PSTR estimate, not an external electrical measurement.
CPU/GPU/ANE counters are retained separately; missing CPU counters are never
silently interpreted as a complete chip-power measurement.
"""

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from .cases import workload
from .common import environment, save, stats

POWER_FIELDS = ("sys_power", "cpu_power", "gpu_power", "ane_power", "ram_power")


def integrate(
    samples, start, end, field="sys_power", max_system_watts=500, max_sample_gap_seconds=2
):
    """Trapezoidal estimate with interpolated boundaries and no extrapolation."""
    if any(field not in s["metrics"] for s in samples):
        raise ValueError(f"Missing {field} reading; do not delete or interpolate missing samples")
    rows = [(s["monotonic_seconds"], s["metrics"][field]) for s in samples]
    if end <= start or len(rows) < 2:
        raise ValueError("A positive interval and at least two samples are required")
    times, watts = np.asarray(rows, dtype=np.float64).T
    if not np.isfinite(times).all() or not np.isfinite(watts).all():
        raise ValueError("Non-finite power sample")
    if field == "sys_power" and np.any(watts <= 0):
        raise ValueError("System power must be positive; zero can indicate a missing SMC reading")
    if field == "sys_power" and np.any(watts > max_system_watts):
        raise ValueError(
            f"System power exceeds the configured {max_system_watts:g} W sanity ceiling; "
            "reject this run rather than clipping or deleting samples"
        )
    if np.any(np.diff(times) <= 0) or times[0] > start or times[-1] < end:
        raise ValueError("Power samples must strictly increase and cover the interval")
    if np.any(np.diff(times) > max_sample_gap_seconds):
        raise ValueError(f"Power sample gap exceeds {max_sample_gap_seconds:g} seconds")
    interior = (times > start) & (times < end)
    t = np.r_[start, times[interior], end]
    p = np.r_[np.interp(start, times, watts), watts[interior], np.interp(end, times, watts)]
    joules = float(np.trapezoid(p, t))
    return {"joules": joules, "mean_watts": joules / (end - start)}


class Sampler:
    def __init__(self, executable, interval_ms, max_system_watts=500):
        self.samples = []
        self.errors = []
        self.command = [str(executable), "pipe", "-i", str(interval_ms)]
        self.process = None
        self.max_system_watts = max_system_watts
        self.max_sample_gap_seconds = max(2, 3 * interval_ms / 1000)

    def start(self):
        self.process = subprocess.Popen(
            self.command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 15
        while not self.samples:
            if time.monotonic() > deadline or self.process.poll() is not None:
                self.stop()
                raise RuntimeError(f"macmon returned no samples: {self.errors}")
            time.sleep(0.05)
        return self

    def _read(self):
        for line in self.process.stdout:
            received = time.perf_counter()
            wall = time.time()
            try:
                metric = json.loads(line)
                self.samples.append(
                    {"monotonic_seconds": received, "wall_seconds": wall, "metrics": metric}
                )
            except (ValueError, TypeError) as error:
                self.errors.append(str(error))

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.thread.join(timeout=5)

    def covered(self, end):
        deadline = time.monotonic() + 5
        while self.samples[-1]["monotonic_seconds"] < end:
            if time.monotonic() > deadline:
                raise RuntimeError("Power sampler stopped covering the benchmark")
            time.sleep(0.05)


def summarize_power(sampler, start, end):
    sampler.covered(end)
    return {
        field: integrate(
            sampler.samples,
            start,
            end,
            field,
            sampler.max_system_watts,
            sampler.max_sample_gap_seconds,
        )
        for field in POWER_FIELDS
        if field == "sys_power" or all(field in s["metrics"] for s in sampler.samples)
    }


def idle(sampler, seconds):
    start = time.perf_counter()
    time.sleep(seconds)
    end = time.perf_counter()
    # Let previous accelerator activity and slow SMC updates settle first.
    measured_start = start + min(3, seconds / 2)
    return {
        "start": start,
        "end": end,
        "measured_start": measured_start,
        "power": summarize_power(sampler, measured_start, end),
    }


def endpoint_block(agent, cases, seconds, rate, expected=None):
    calls, misses, latencies, unstable = 0, 0, [], 0
    process = psutil.Process()
    rss_before = process.memory_info().rss
    start = time.perf_counter()
    deadline = start + seconds
    while time.perf_counter() < deadline:
        if rate:
            target = start + calls / rate
            if target >= deadline:
                break
            time.sleep(max(0, target - time.perf_counter()))
        t0 = time.perf_counter()
        case_index = calls % len(cases)
        result = agent.predict(*cases[case_index])
        t1 = time.perf_counter()
        if not result.get("answers"):
            raise ValueError("The endpoint returned no decisions")
        if expected is not None and result != expected[case_index]:
            unstable += 1
        latencies.append((t1 - t0) * 1000)
        calls += 1
        if rate and t1 > start + calls / rate:
            misses += 1
    if rate:
        time.sleep(max(0, deadline - time.perf_counter()))
    end = time.perf_counter()
    return {
        "start": start,
        "end": end,
        "duration_seconds": end - start,
        "completed_decisions": calls,
        "deadline_misses": misses,
        "unstable_rounded_results": unstable,
        "decisions_per_second": calls / (end - start),
        "interval_mean_ms_per_decision": (end - start) * 1000 / calls,
        "latency": stats(latencies),
        "latency_ms": latencies,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": process.memory_info().rss,
    }


def comparison(blocks):
    totals = {}
    for backend in sorted({b["backend"] for b in blocks}):
        group = [b for b in blocks if b["backend"] == backend]
        calls = sum(b["completed_decisions"] for b in group)
        duration = sum(b["duration_seconds"] for b in group)
        gross = sum(b["power"]["sys_power"]["joules"] for b in group)
        incremental = sum(b["incremental_system_joules"] for b in group)
        totals[backend] = {
            "blocks": len(group),
            "completed_decisions": calls,
            "duration_seconds": duration,
            "mean_ms_per_decision": duration * 1000 / calls,
            "system_mean_watts": gross / duration,
            "system_joules_per_decision": gross / calls,
            "idle_subtracted_mean_watts": incremental / duration,
            "idle_subtracted_joules_per_decision": incremental / calls,
        }
    ratios = {}
    for backend in [key for key in totals if key != "mlx"]:
        if "mlx" not in totals:
            break
        a, b = totals["mlx"], totals[backend]
        ratios[backend] = {
            "speed": a["mean_ms_per_decision"] / b["mean_ms_per_decision"],
            "system_power": a["system_mean_watts"] / b["system_mean_watts"],
            "system_energy_per_decision": a["system_joules_per_decision"]
            / b["system_joules_per_decision"],
            "idle_subtracted_energy_per_decision": (
                a["idle_subtracted_joules_per_decision"] / b["idle_subtracted_joules_per_decision"]
                if b["idle_subtracted_joules_per_decision"] > 0
                and a["idle_subtracted_joules_per_decision"] > 0
                else None
            ),
        }
    if ratios:
        totals["ratios_by_candidate"] = ratios
        if set(ratios) == {"ane"}:
            totals["ratios"] = ratios["ane"]
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--fp16-candidate", type=Path, help="Optional third arm using the same factory"
    )
    parser.add_argument(
        "--candidate-factory", required=True, help="module:function(source, package)"
    )
    parser.add_argument("--sampler", "--macmon", dest="macmon", type=Path, required=True)
    parser.add_argument(
        "--pstr-only",
        action="store_true",
        help="The supplied sampler reads SMC PSTR directly without macmon's component-sum floor",
    )
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--idle-seconds", type=float, default=10)
    parser.add_argument(
        "--cycles", type=int, default=3, help="Balanced cycles; 3 gives 6 blocks/backend"
    )
    parser.add_argument("--sample-ms", type=int, default=500)
    parser.add_argument(
        "--max-system-watts",
        type=float,
        default=500,
        help="Reject implausible telemetry; 500 W is a loose ceiling for the tested M3 Max",
    )
    parser.add_argument(
        "--rate", type=float, default=0, help="0=saturated; otherwise decisions/sec"
    )
    parser.add_argument("--mlx-eager", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds < 10 or args.idle_seconds < 6 or args.cycles < 1 or args.rate < 0:
        parser.error("Require seconds>=10, idle>=6, cycles>=1 and rate>=0")
    if args.sample_ms < 1:
        parser.error("sample-ms must be positive")
    if not np.isfinite(args.max_system_watts) or args.max_system_watts <= 0:
        parser.error("max-system-watts must be finite and positive")
    from laya_mlx import Agent

    module, function = args.candidate_factory.split(":")
    factory = getattr(importlib.import_module(module), function)
    agents = {
        "mlx": Agent(
            args.source,
            dtype="float16",
            batch_size=1,
            compile=not args.mlx_eager,
            cache_prompts=True,
            pad_to_multiple=32,
        ),
        "ane": factory(args.source, args.candidate),
    }
    if args.fp16_candidate:
        agents["ane_fp16"] = factory(args.source, args.fp16_candidate)
    state, questions = workload(1)
    cases = []
    for suffix in range(8):
        changed = copy.deepcopy(state)
        changed["subject"] = f"Duplicate charge on invoice #441{suffix}"
        cases.append((changed, questions))
    warmup = {}
    for backend, agent in agents.items():
        lengths = [len(agent.prepare(*case)[0][0]["ids"]) for case in cases]
        results, latency = [], []
        for i in range(32):
            start = time.perf_counter()
            result = agent.predict(*cases[i % len(cases)])
            latency.append((time.perf_counter() - start) * 1000)
            if i < len(cases):
                results.append(result)
        warmup[backend] = {"lengths": lengths, "latency_ms": latency, "results": results}
    agreements = {
        backend: [
            all(
                lhs["answers"][key]["choice"] == rhs["answers"][key]["choice"]
                for key in lhs["answers"]
            )
            for lhs, rhs in zip(warmup["mlx"]["results"], warmup[backend]["results"])
        ]
        for backend in agents
        if backend != "mlx"
    }
    if not all(all(agreement) for agreement in agreements.values()):
        raise ValueError("Selected answers differ on the energy workload")
    report = {
        "environment": environment(),
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "measurement": {
            "scope": "synchronous complete predict; all compared models resident; load and warmup excluded",
            "system_power": (
                "SMC PSTR only, without a component-sum floor; includes baseline machine power; not a calibrated external meter"
                if args.pstr_only
                else "macmon max(SMC PSTR, component sum) estimate; includes baseline machine power; not a calibrated external meter"
            ),
            "integration": "trapezoidal, monotonic receive timestamps; interpolated interval boundaries",
            "component_integration": "diagnostic estimates only: IOReport reports preceding-interval averages, so treating them as point samples can shift boundaries by about half a sample",
            "idle_subtraction": "mean of preceding and following settled idle; negative values retained",
            "counter_caveat": (
                "Direct PSTR-only sampling; component counters are not read or emitted, and no component-sum floor is applied."
                if args.pstr_only
                else "CPU and ANE counters usually return zero on this OS and can spike implausibly; sys_power has a component-sum floor, so these faults can contaminate it. Reject invalid runs; never sum missing counters as complete chip power."
            ),
            "telemetry_sanity": "All samples must be positive, finite and below the configured system-power ceiling; any failure invalidates the energy run without deleting or clipping samples.",
            "max_sample_gap_seconds": max(2, 3 * args.sample_ms / 1000),
            "ratio_identity": "speed * average power reduction = energy per decision improvement (do not multiply speed again)",
        },
        "macmon_version": subprocess.check_output(
            [str(args.macmon), "--version"], text=True
        ).strip(),
        "sampler_executable_sha256": hashlib.sha256(args.macmon.read_bytes()).hexdigest(),
        "candidate_manifests": {
            name: getattr(agent, "manifest", None)
            for name, agent in agents.items()
            if name != "mlx"
        },
        "candidate_source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(inspect.getfile(factory)).parent.glob("*.py"))
        },
        "baseline_options": {
            "dtype": "float16",
            "batch_size": 1,
            "compile": not args.mlx_eager,
            "cache_prompts": True,
            "pad_to_multiple": 32,
        },
        "workload": cases,
        "warmup": warmup,
        "answer_agreement": agreements,
        "blocks": [],
    }
    order = (
        ("mlx", "ane_fp16", "ane", "ane", "ane_fp16", "mlx")
        if args.fp16_candidate
        else ("mlx", "ane", "ane", "mlx")
    )
    report["block_order"] = order
    sampler = Sampler(args.macmon, args.sample_ms, args.max_system_watts).start()
    try:
        before = idle(sampler, args.idle_seconds)
        for cycle in range(args.cycles):
            for backend in order:
                block = endpoint_block(
                    agents[backend], cases, args.seconds, args.rate, warmup[backend]["results"]
                )
                block.update(backend=backend, cycle=cycle, idle_before=before)
                block["power"] = summarize_power(sampler, block["start"], block["end"])
                after = idle(sampler, args.idle_seconds)
                block["idle_after"] = after
                idle_watts = (
                    before["power"]["sys_power"]["mean_watts"]
                    + after["power"]["sys_power"]["mean_watts"]
                ) / 2
                block["idle_mean_watts"] = idle_watts
                block["incremental_system_joules"] = (
                    block["power"]["sys_power"]["joules"] - idle_watts * block["duration_seconds"]
                )
                report["blocks"].append(block)
                report["summary"] = comparison(report["blocks"])
                report["power_samples"] = sampler.samples.copy()
                save(args.output, report)
                print(
                    f"cycle={cycle} {backend}: P50={block['latency']['p50_ms']:.3f} ms, "
                    f"system={block['power']['sys_power']['mean_watts']:.2f} W, "
                    f"idle={idle_watts:.2f} W, calls={block['completed_decisions']}",
                    flush=True,
                )
                before = after
    finally:
        sampler.stop()
        report["power_samples"] = sampler.samples.copy()
        report["sampler_errors"] = sampler.errors
        save(args.output, report)


if __name__ == "__main__":
    main()
