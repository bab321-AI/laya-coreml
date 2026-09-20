"""Run from a fresh wheel environment, without training/runtime comparison extras."""

import argparse
import importlib.util
import json
import socket
import sys
from pathlib import Path

import laya_coreml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub-check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not Path(laya_coreml.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise AssertionError("Test must import the installed wheel, not the source checkout")
    missing = {
        name: importlib.util.find_spec(name) is None for name in ("torch", "mlx", "transformers")
    }
    if not all(missing.values()):
        raise AssertionError(f"Use a clean inference-only environment: {missing}")
    report = {
        "installed_version": laya_coreml.__version__,
        "module_path": laya_coreml.__file__,
        "without_dependencies": missing,
        "models": [],
    }
    original = socket.socket.connect
    attempts = []

    def blocked(*_args, **_kwargs):
        attempts.append(True)
        raise RuntimeError("Unexpected network request during local inference")

    question = {"refund": {"type": "noul", "instructions": "Does the customer request a refund?"}}
    state = "The customer requests a refund of a duplicate payment."
    receipts = json.loads((root / "benchmarks/results/hub-release.json").read_text())
    for receipt in receipts:
        folder = root / "models/hub" / receipt["repository"].split("/")[-1]
        socket.socket.connect = blocked
        try:
            agent = laya_coreml.load(folder, local_files_only=True)
            result = agent.predict(state, question)
            assert 0 <= result["answers"]["refund"]["noul"] <= 1
            assert result["usage"]["output_tokens"] == 0
            assert all(agent.predict(state, question) == result for _ in range(3))
        finally:
            socket.socket.connect = original
        report["models"].append(
            {
                "repository": receipt["repository"],
                "revision": receipt["revision"],
                "compute_units": agent.compute_units,
                "result": result,
                "repeat_calls": 3,
                "stable": True,
            }
        )
        print(receipt["repository"], agent.compute_units, "passed offline", flush=True)
    if args.hub_check:
        receipt = next(row for row in receipts if row["repository"].endswith("-ane"))
        print("Checking public pinned Hub download", flush=True)
        agent = laya_coreml.load(receipt["repository"], revision=receipt["revision"])
        expected = agent.predict(state, question)
        socket.socket.connect = blocked
        try:
            cached = laya_coreml.load(
                receipt["repository"], revision=receipt["revision"], local_files_only=True
            )
            assert cached.predict(state, question) == expected
        finally:
            socket.socket.connect = original
        report["public_hub_download_and_offline_cache"] = {
            "repository": receipt["repository"],
            "revision": receipt["revision"],
            "passed": True,
        }
    report["network_attempts_during_offline_sections"] = len(attempts)
    (root / "benchmarks/results/wheel-runtime-smoke.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    assert not attempts


if __name__ == "__main__":
    main()
