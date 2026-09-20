"""Measure complete fixed-shape predictions for a real short or long fixture."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from benchmarks.cases import workload
from benchmarks.common import environment, save, stats
from laya_coreml.convert import resolve_source

from .artifact import file_digest
from .runtime import ANEAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--source", help="Defaults to the source directory in the manifest")
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be nonnegative and iterations must be positive")
    manifest = json.loads((args.package.parent / "manifest.json").read_text())
    source = resolve_source(args.source or manifest["source_directory"])
    agent = ANEAgent(source, args.package, length=args.length)
    state, questions = workload(1, long=args.long)
    items, _ = agent.prepare(state, questions)
    expected = agent.predict(state, questions)
    for _ in range(args.warmup):
        if agent.predict(state, questions) != expected:
            raise ValueError("Unstable warmup output")
    values = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        result = agent.predict(state, questions)
        values.append((time.perf_counter() - start) * 1000)
        if result != expected:
            raise ValueError("Unstable measured output")
    report = {
        "environment": environment(),
        "artifact_manifest": agent.manifest,
        "experiment_python_sha256": {
            path.name: file_digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "input": {
            "fixture": "workload(1, long=True)" if args.long else "workload(1)",
            "sha256": hashlib.sha256(
                json.dumps([state, questions], ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest(),
            "questions": len(items),
            "actual_token_lengths": [len(item["ids"]) for item in items],
            "exported_length": args.length,
        },
        "warmup": args.warmup,
        "stable_rounded_results": True,
        "result": expected,
        "end_to_end": {**stats(values), "samples_ms": values},
        "note": "Serial screening run, not a paired backend comparison; includes complete predict, excludes model loading and warmup.",
    }
    save(args.output, report)
    print(json.dumps({"input": report["input"], "timing": stats(values)}, indent=2))


if __name__ == "__main__":
    main()
