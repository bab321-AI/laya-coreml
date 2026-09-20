"""Validate a fixed-shape experimental ANE runtime against stored FP32 goldens."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchmarks.common import calibrated, compute_plan, environment, save, softmax, stats
from laya_coreml.convert import resolve_source
from laya_coreml.inputs import collate_items

from .artifact import file_digest
from .runtime import ANEAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        help="Original local checkpoint or pinned Hub model; defaults to the artifact source directory",
    )
    parser.add_argument("--package", default="experiments/ane_engineering/body96/model.mlpackage")
    parser.add_argument("--name", default="laya-multilingual")
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/ane_engineering/validation96.json")
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--max-probability-error", type=float, default=0.02)
    args = parser.parse_args()
    if args.repeats < 1 or not 0 < args.max_probability_error < 1:
        parser.error("repeats must be positive and probability error must be between 0 and 1")
    manifest = json.loads((Path(args.package).parent / "manifest.json").read_text())
    source_path = resolve_source(args.source or manifest["source_directory"])
    agent = ANEAgent(source_path, args.package, length=args.length)
    reference = json.loads(Path("benchmarks/results/reference.json").read_text())["models"][
        args.name
    ]
    if (
        reference["source_weights_sha256"]
        != agent.manifest["source_files_sha256"]["model.safetensors"]
    ):
        raise ValueError("Golden reference and artifact use different original checkpoint weights")
    report = {
        "environment": environment(),
        "name": args.name,
        "length": args.length,
        "package": args.package,
        "cases": [],
        "artifact_manifest": agent.manifest,
        "experiment_python_sha256": {
            path.name: file_digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "passed": False,
        "gate": {
            "argmax": "all evaluated questions agree",
            "max_probability_error": args.max_probability_error,
            "max_action_probability_error": args.max_probability_error,
            "finite_outputs": True,
            "unchanged_token_usage": True,
            "repeated_results_identical": True,
        },
    }
    report["compute_plan"] = compute_plan(agent)
    repeat_cases = []
    for entry in reference["cases"]:
        items, _ = agent.prepare(entry["state"], entry["questions"])
        assert items == entry["items"]
        if max(len(item["ids"]) for item in items) > args.length:
            report["cases"].append({"case": entry["name"], "skipped": True})
            continue
        row = {
            "case": entry["name"],
            "questions": len(items),
            "argmax_agree": 0,
            "max_probability_error": 0,
            "max_action_probability_error": 0,
            "max_action_logit_error": 0,
            "max_logit_error": 0,
        }
        for index, item in enumerate(items):
            batch = collate_items([item], agent.tok.pad_token_id, shape=agent.shape)
            logits, action = agent.forward(batch)
            if not np.isfinite(logits).all() or not np.isfinite(action).all():
                raise FloatingPointError("Nonfinite ANE result")
            p, ref = (
                calibrated(agent, logits[0], item),
                calibrated(agent, entry["logits"][index], item),
            )
            row["argmax_agree"] += int(p.argmax() == ref.argmax())
            row["max_probability_error"] = max(
                row["max_probability_error"], float(np.abs(p - ref).max())
            )
            row["max_action_probability_error"] = max(
                row["max_action_probability_error"],
                float(np.abs(softmax(action[0]) - softmax(entry["action_logits"][index])).max()),
            )
            row["max_action_logit_error"] = max(
                row["max_action_logit_error"],
                float(np.abs(action[0] - np.asarray(entry["action_logits"][index])).max()),
            )
            count = len(item["markers"])
            row["max_logit_error"] = max(
                row["max_logit_error"],
                float(np.abs(logits[0, :count] - np.array(entry["logits"][index][:count])).max()),
            )
        result = agent.predict(entry["state"], entry["questions"])
        assert result["usage"] == entry["result"]["usage"]
        repeat_cases.append((entry["state"], entry["questions"], result))
        report["cases"].append(row)
        save(args.output, report)
        print(row, flush=True)
    stable = True
    for index in range(args.repeats):
        state, questions, expected = repeat_cases[index % len(repeat_cases)]
        stable &= agent.predict(state, questions) == expected
    report["repeat"] = {"calls": args.repeats, "identical_rounded_results": stable}
    from benchmarks.cases import workload

    state, questions = workload(1)
    for _ in range(10):
        agent.predict(state, questions)
    values = []
    for _ in range(50):
        started = time.perf_counter()
        agent.predict(state, questions)
        values.append((time.perf_counter() - started) * 1000)
    report["short1_end_to_end"] = {**stats(values), "samples_ms": values}
    completed = [row for row in report["cases"] if not row.get("skipped")]
    report["total_questions"] = sum(row["questions"] for row in completed)
    report["argmax_agreements"] = sum(row["argmax_agree"] for row in completed)
    report["passed"] = bool(
        completed
        and report["argmax_agreements"] == report["total_questions"]
        and max(row["max_probability_error"] for row in completed) <= args.max_probability_error
        and max(row["max_action_probability_error"] for row in completed)
        <= args.max_probability_error
        and stable
    )
    report["coverage"] = "fixed-shape subset; skipped cases remain unvalidated"
    save(args.output, report)
    print(
        "SUMMARY",
        sum(r.get("argmax_agree", 0) for r in report["cases"]),
        sum(r.get("questions", 0) for r in report["cases"]),
        "repeat",
        stable,
        "p50",
        report["short1_end_to_end"]["p50_ms"],
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
