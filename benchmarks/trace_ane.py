"""Short Core ML workload for a separate Instruments trace (not a latency result)."""

import argparse
import os
import time
from pathlib import Path

from .cases import workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--ready", type=Path, required=True)
    args = parser.parse_args()
    from experiments.ane_engineering.runtime import ANEAgent

    agent = ANEAgent(args.source, args.package)
    state, questions = workload(1)
    for _ in range(10):
        agent.predict(state, questions)
    args.ready.parent.mkdir(parents=True, exist_ok=True)
    args.ready.write_text(str(os.getpid()))
    print("Ready for Core ML tracing", os.getpid(), flush=True)
    end, calls = time.perf_counter() + args.seconds, 0
    while time.perf_counter() < end:
        agent.predict(state, questions)
        calls += 1
    print("Completed", calls, "predictions during diagnostic workload", flush=True)


if __name__ == "__main__":
    main()
