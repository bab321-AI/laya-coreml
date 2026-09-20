import pytest

from benchmarks.energy import comparison, integrate
from benchmarks.energy_summary import audit_structure


def test_integration_interpolates_edges_and_rejects_missing_coverage():
    samples = [{"monotonic_seconds": t, "metrics": {"sys_power": 10 + 2 * t}} for t in (0, 1, 2, 3)]
    assert integrate(samples, 0.5, 2.5) == {"joules": 26.0, "mean_watts": 13.0}
    with pytest.raises(ValueError, match="cover"):
        integrate(samples, 0, 4)


@pytest.mark.parametrize("invalid", [0, float("nan"), float("inf"), 40089.359375])
def test_invalid_system_power_is_not_silently_counted_as_savings(invalid):
    samples = [
        {"monotonic_seconds": 0, "metrics": {"sys_power": 10}},
        {"monotonic_seconds": 1, "metrics": {"sys_power": invalid}},
        {"monotonic_seconds": 2, "metrics": {"sys_power": 10}},
    ]
    with pytest.raises(ValueError):
        integrate(samples, 0, 2)


def test_comparison_uses_interval_means_and_does_not_double_count_speed():
    blocks = []
    for backend, duration, watts in (("mlx", 20, 50), ("ane", 10, 10)):
        blocks.append(
            {
                "backend": backend,
                "completed_decisions": 100,
                "duration_seconds": duration,
                "power": {"sys_power": {"joules": duration * watts}},
                "incremental_system_joules": duration * (watts - 5),
            }
        )
    ratios = comparison(blocks)["ratios"]
    assert ratios["speed"] == 2
    assert ratios["system_power"] == 5
    assert ratios["system_energy_per_decision"] == 10
    assert ratios["idle_subtracted_energy_per_decision"] == 18
    blocks.append(
        {
            "backend": "ane_fp16",
            "completed_decisions": 100,
            "duration_seconds": 10,
            "power": {"sys_power": {"joules": 200}},
            "incremental_system_joules": 150,
        }
    )
    combined = comparison(blocks)["ratios_by_candidate"]
    assert combined["ane"]["system_energy_per_decision"] == 10
    assert combined["ane_fp16"]["system_energy_per_decision"] == 5


def test_missing_system_power_and_long_sampling_gaps_are_rejected():
    missing = [
        {"monotonic_seconds": 0, "metrics": {"sys_power": 10}},
        {"monotonic_seconds": 1, "metrics": {}},
        {"monotonic_seconds": 2, "metrics": {"sys_power": 10}},
    ]
    with pytest.raises(ValueError, match="Missing sys_power"):
        integrate(missing, 0, 2)
    sparse = [{"monotonic_seconds": t, "metrics": {"sys_power": 10}} for t in (0, 1, 4, 5)]
    with pytest.raises(ValueError, match="sample gap"):
        integrate(sparse, 0, 5)


def test_partial_or_reordered_runs_cannot_be_bootstrapped_as_balanced_cycles():
    order = ["mlx", "ane_fp16", "ane", "ane", "ane_fp16", "mlx"]
    report = {
        "settings": {"cycles": 3, "fp16_candidate": "fp16.mlpackage"},
        "block_order": order,
        "blocks": [{"cycle": cycle, "backend": name} for cycle in range(3) for name in order],
    }
    audit_structure(report)
    last = report["blocks"].pop()
    with pytest.raises(ValueError, match="Incomplete or unbalanced"):
        audit_structure(report)
    report["blocks"].append(last)
    report["blocks"][0], report["blocks"][1] = report["blocks"][1], report["blocks"][0]
    with pytest.raises(ValueError, match="Incomplete or unbalanced"):
        audit_structure(report)
