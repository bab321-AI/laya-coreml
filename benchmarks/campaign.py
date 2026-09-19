"""Run performance jobs sequentially to avoid competing model inference."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--model-root", type=Path, default=Path("models"))
    p.add_argument("--iterations", type=int, default=100)
    a = p.parse_args()
    for name in ("laya-multilingual", "laya", "laya-typed-decisions"):
        for backend in ("coreml", "mlx"):
            model = (a.model_root if backend == "coreml" else a.source_root) / name
            command = [
                sys.executable,
                "-m",
                "benchmarks.run",
                str(model),
                "--backend",
                backend,
                "--compute-units",
                "cpu_gpu",
                "--iterations",
                str(a.iterations),
                "--output",
                f"benchmarks/results/perf-{name}-{backend}.json",
            ]
            if backend == "coreml":
                command.append("--plan")
            subprocess.run(command, check=True)
    for unit in ("all", "cpu_ne", "cpu"):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.run",
                str(a.model_root / "laya-multilingual"),
                "--compute-units",
                unit,
                "--questions",
                "1",
                "--iterations",
                "50",
                "--plan",
                "--output",
                f"benchmarks/results/perf-multilingual-{unit}.json",
            ],
            check=True,
        )
    for backend in ("coreml", "mlx"):
        model = (a.model_root if backend == "coreml" else a.source_root) / "laya-multilingual"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.run",
                str(model),
                "--backend",
                backend,
                "--compute-units",
                "cpu_gpu",
                "--questions",
                "1",
                "--long",
                "--iterations",
                "50",
                "--output",
                f"benchmarks/results/perf-multilingual-long-{backend}.json",
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.snake",
            str(a.model_root / "laya-multilingual-snake"),
            str(a.source_root / "laya-multilingual"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
