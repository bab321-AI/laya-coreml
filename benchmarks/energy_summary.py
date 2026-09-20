"""Audit raw energy runs and summarize ratios without counting speed twice."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .common import save, stats
from .energy import comparison, integrate


def audit_structure(report):
    settings = report["settings"]
    order = (
        ["mlx", "ane_fp16", "ane", "ane", "ane_fp16", "mlx"]
        if settings.get("fp16_candidate")
        else ["mlx", "ane", "ane", "mlx"]
    )
    if report.get("block_order", order) != order:
        raise ValueError("Unexpected balanced block order")
    expected = [(cycle, backend) for cycle in range(settings["cycles"]) for backend in order]
    observed = [(block["cycle"], block["backend"]) for block in report["blocks"]]
    if not expected or observed != expected:
        raise ValueError("Incomplete or unbalanced measurement cycles")


def summarize(path):
    report = json.loads(path.read_text())
    audit_structure(report)
    if report.get("sampler_errors"):
        raise ValueError("Power sampler reported malformed readings")
    blocks, samples = report["blocks"], report["power_samples"]
    times = np.array([s["monotonic_seconds"] for s in samples])
    # Re-audit original runs, including those recorded before strict SMC checks.
    for block in blocks:
        if block.get("unstable_rounded_results", 0):
            raise ValueError("Rounded predictions changed during the sustained test")
        actual = integrate(
            samples,
            block["start"],
            block["end"],
            max_system_watts=report["settings"].get("max_system_watts", 500),
            max_sample_gap_seconds=max(2, 3 * report["settings"].get("sample_ms", 500) / 1000),
        )
        if not np.isclose(actual["joules"], block["power"]["sys_power"]["joules"]):
            raise ValueError("Stored and recomputed system energy disagree")
        if not np.isclose(block["end"] - block["start"], block["duration_seconds"]):
            raise ValueError("Stored duration disagrees with interval boundaries")
        if block["completed_decisions"] != len(block["latency_ms"]):
            raise ValueError("Decision count and latency sample count disagree")
        idle_watts = []
        for key in ("idle_before", "idle_after"):
            idle = block[key]
            measured = integrate(
                samples,
                idle["measured_start"],
                idle["end"],
                max_system_watts=report["settings"].get("max_system_watts", 500),
                max_sample_gap_seconds=max(2, 3 * report["settings"].get("sample_ms", 500) / 1000),
            )
            if not np.isclose(measured["joules"], idle["power"]["sys_power"]["joules"]):
                raise ValueError("Stored and recomputed idle energy disagree")
            idle_watts.append(measured["mean_watts"])
        idle_mean = float(np.mean(idle_watts))
        incremental = actual["joules"] - idle_mean * block["duration_seconds"]
        if not np.isclose(idle_mean, block["idle_mean_watts"]) or not np.isclose(
            incremental, block["incremental_system_joules"]
        ):
            raise ValueError("Stored and recomputed idle-subtracted energy disagree")
    result = {
        "input": str(path),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "totals": comparison(blocks),
        "telemetry_audit": {
            "complete_balanced_cycles": report["settings"]["cycles"],
            "gross_and_adjacent_idle_energy_recomputed": True,
            "system_power_positive_and_finite": True,
            "system_power_below_sanity_ceiling_watts": report["settings"].get(
                "max_system_watts", 500
            ),
            "maximum_system_power_watts": max(s["metrics"]["sys_power"] for s in samples),
            "sample_count": len(samples),
            "maximum_sample_gap_seconds": float(np.diff(times).max()),
            "all_cpu_power_zero": (
                all(s["metrics"]["cpu_power"] == 0 for s in samples)
                if all("cpu_power" in s["metrics"] for s in samples)
                else None
            ),
            "all_ane_power_zero": (
                all(s["metrics"]["ane_power"] == 0 for s in samples)
                if all("ane_power" in s["metrics"] for s in samples)
                else None
            ),
            "all_blocks_check_rounded_output_stability": all(
                "unstable_rounded_results" in block for block in blocks
            ),
        },
        "latency": {},
        "block_ranges": {},
        "uncertainty_note": "95% percentile bootstrap of whole balanced cycles (ABBA or ABCCBA); few cycles and one desktop session do not capture all meter error, background load, or future-run variation",
    }
    for backend in sorted({b["backend"] for b in blocks}):
        group = [b for b in blocks if b["backend"] == backend]
        result["latency"][backend] = stats([t for b in group for t in b["latency_ms"]])
        result["block_ranges"][backend] = {
            "system_mean_watts": [
                min(b["power"]["sys_power"]["mean_watts"] for b in group),
                max(b["power"]["sys_power"]["mean_watts"] for b in group),
            ],
            "idle_mean_watts": [
                min(b["idle_mean_watts"] for b in group),
                max(b["idle_mean_watts"] for b in group),
            ],
        }
    cycles = sorted({b["cycle"] for b in blocks})
    if len(cycles) >= 3:
        rng = np.random.default_rng(20260920)
        ratios = {}
        # Resample complete balanced cycles as clusters, retaining internal ordering.
        groups = [[b for b in blocks if b["cycle"] == c] for c in cycles]
        for indices in rng.integers(0, len(groups), size=(5000, len(groups))):
            sample = [block for index in indices for block in groups[index]]
            summary = comparison(sample)
            values = summary.get("ratios")
            if values is None:
                values = {
                    f"{backend}.{name}": value
                    for backend, row in summary["ratios_by_candidate"].items()
                    for name, value in row.items()
                }
            for name, value in values.items():
                if value is not None:
                    ratios.setdefault(name, []).append(value)
        result["cycle_bootstrap_95_percent"] = {
            name: np.percentile(values, [2.5, 97.5]).tolist() for name, values in ratios.items()
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [summarize(path) for path in args.inputs]
    save(args.output, reports)
    for report in reports:
        print(report["input"], report["totals"]["ratios_by_candidate"], flush=True)


if __name__ == "__main__":
    main()
