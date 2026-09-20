"""Generate the benchmark summary from the committed raw measurements."""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    results = root / "benchmarks/results"

    def read(name):
        return json.loads((results / name).read_text())

    lines = [
        "# Local benchmark results",
        "",
        "M3 Max (40 GPU cores, 128 GB unified memory), macOS 27.2, Python 3.12.13, "
        "coremltools 9.0, MLX 0.32.2, NumPy 2.1.3. Measured 2026-09-20.",
        "",
        "**This page measures the ordinary SDPA export**, which does not outperform MLX "
        "on these workloads. Core ML CPU+GPU is its recommended/default configuration. "
        "Automatic and CPU+Neural Engine selection preferred CPU operations in that plan.",
        "",
        "The subsequent **[ANE graph rewrite](docs/ANE_BENCHMARKS.md)** improves short-decision "
        "speed and measured system energy against compiled MLX FP16. Its FP16/8-bit comparison, "
        "shape limits, fidelity tests and hardware evidence are reported separately; "
        "the requested 10× improvement was not achieved.",
        "",
        "## Short typed decisions",
        "",
        "FP16. End-to-end wall time includes prompt construction, tokenization, input arrays, "
        "synchronous model execution, calibration and result formatting. Loading and 10 warmup "
        "calls are excluded. 100 measured calls per cell; timings and the first prediction are "
        "retained in the linked JSON. Jobs ran sequentially with no competing model jobs.",
        "",
        "| Model | Backend | 1 question P50 / P95 | 3 questions P50 | 10 questions P50 |",
        "|---|---|---:|---:|---:|",
    ]
    names = {
        "laya-multilingual": "Multilingual 322M",
        "laya": "Laya 421M",
        "laya-typed-decisions": "Typed Decisions 421M",
    }
    for name, label in names.items():
        for backend in ("coreml", "mlx"):
            filename = f"perf-{name}-{backend}.json"
            report = read(filename)
            w = report["workloads"]
            title = "Core ML CPU+GPU" if backend == "coreml" else "MLX GPU"
            lines.append(
                f"| {label} | [{title}](benchmarks/results/{filename}) | {w[0]['p50_ms']:.2f} / {w[0]['p95_ms']:.2f} ms | {w[1]['p50_ms']:.2f} ms | {w[2]['p50_ms']:.2f} ms |"
            )
    lines += [
        "",
        "Core ML exports use **B=1**, with short sequences padded to 96 tokens; MLX uses "
        "batch_size=16 and unpadded lengths of 91 (multilingual) / 93 (the other models) for "
        "one question. Both preserve the same prompt tokens and mask added padding. "
        "Multi-question rows compare the shipped APIs: Core ML executes questions sequentially, "
        "while MLX batches them. They are not a comparison of equal batch tensor shapes. "
        "MLX uses its eager FP16 path; compile/prefix-cache optimizations are not enabled.",
        "",
        "These are measurements from one desktop run, not a guarantee of future p95 latency. "
        "Clock scaling and background system activity were not controlled. No energy or battery "
        "measurement was performed in this ordinary-export campaign. In particular the multilingual run showed a broader latency "
        "distribution; all individual samples remain available.",
        "",
        "## Compute-unit selection",
        "",
        "Multilingual, one short question. GPU row uses 100 samples; other rows use 50. "
        "All predictions completed and repeated public results were identical.",
        "",
        "| Allowed compute units | P50 | P95 | Plan preference |",
        "|---|---:|---:|---|",
    ]
    for unit, filename in [
        ("CPU+GPU", "perf-laya-multilingual-coreml.json"),
        ("ALL", "perf-multilingual-all.json"),
        ("CPU+NE", "perf-multilingual-cpu_ne.json"),
        ("CPU only", "perf-multilingual-cpu.json"),
    ]:
        r = read(filename)
        w = r["workloads"][0]
        counts = r["compute_plan"]["preferred_operation_counts"]
        preference = ", ".join(
            f"{count} {kind.removeprefix('ML').removesuffix('ComputeDevice')} operations"
            for kind, count in counts.items()
            if kind != "unknown"
        )
        lines.append(
            f"| [{unit}](benchmarks/results/{filename}) | {w['p50_ms']:.2f} ms | {w['p95_ms']:.2f} ms | {preference} |"
        )
    lines += [
        "",
        "No operation in these inspected plans preferred the Neural Engine. In the "
        "CPU+GPU plan, 1,345 nonconstant operations preferred GPU; 1,787 constants had no "
        "device attribution. The ALL / CPU+NE plans attributed 1,318 operations to CPU, "
        "with constants and 27 other operations unattributed. These are anticipated "
        "`MLComputePlan` assignments, not a runtime hardware trace. Allowing Neural Engine "
        "does not establish that it accelerated this model.",
        "",
        "## Long input and startup",
        "",
        "Multilingual, one 1024-token question, 50 samples:",
        "",
        "| Backend | P50 | P95 |",
        "|---|---:|---:|",
    ]
    for backend in ("coreml", "mlx"):
        w = read(f"perf-multilingual-long-{backend}.json")["workloads"][0]
        lines.append(f"| {backend} | {w['p50_ms']:.2f} ms | {w['p95_ms']:.2f} ms |")
    lines += [
        "",
        "Core ML model construction took 3.4–4.2 seconds in the GPU runs; the first "
        "short prediction took 313–475 ms before warmup. These are observed load/first-call "
        "times with existing OS caches, not a controlled cold-launch experiment. Load once "
        "and warm the model before an interactive loop.",
        "",
        "## Actual Snake workload",
        "",
        "[Raw paired traces](benchmarks/results/snake.json). Two seeds × 300 live steps. "
        "Both backends process each identical state; execution order alternates every tick. "
        "The Core ML action advances the game. Core ML uses fixed **B=3, L=64, K=4**; "
        "MLX uses batch 3 with eager FP16. Timings include planner feature construction "
        "and all three questions; exclude the other backend, game update and UI rendering.",
        "",
        "| Seed | Steps | Score / length | Core ML decision P50 / P95 | MLX decision P50 / P95 | Actions match |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    snake = read("snake.json")
    for r in snake["runs"]:
        c, m = r["latency"]["coreml"], r["latency"]["mlx"]
        lines.append(
            f"| {r['seed']} | {r['steps']} | {r['score']} / {r['length']} | {c['p50_ms']:.2f} / {c['p95_ms']:.2f} ms | {m['p50_ms']:.2f} / {m['p95_ms']:.2f} ms | {r['executed_agreement']}/{r['steps']} |"
        )
    lines += [
        "",
        "**600/600 actions matched; both runs survived, with zero safety interventions.** "
        "Maximum difference in displayed move probability was 0.002. The deterministic "
        "planner and cycle safety mechanism are present in both policies. This is neither "
        "proof of unrestricted Snake intelligence nor a terminal FPS or unlimited-survival test.",
        "",
        "## Fidelity and stability",
        "",
        "| Export | Selected answers | Max calibrated probability error | Repeated API calls |",
        "|---|---:|---:|---:|",
    ]
    for name in ("laya", "multilingual", "typed-decisions", "multilingual-fp32"):
        filename = f"validation-{name}.json"
        r = read(filename)
        lines.append(
            f"| [{name}](benchmarks/results/{filename}) | {r['argmax_agreements']}/{r['questions']} | {r['probability_max_abs_error']:.9f} | {r['stability']['calls']} |"
        )
    lines += [
        "",
        "All input token sequences matched upstream exactly. All repeated rounded "
        "public results were identical and finite. Action-probability error was zero on "
        "these fixtures, where the action distributions are strongly saturated; this is "
        "not a broad calibration evaluation. Process RSS is recorded, including framework "
        "caches; it is not interchangeable with MLX active-memory statistics.",
        "",
        "The 63-question suite per model includes eight languages, 512/1024-token limits, "
        "empty input, literal mask tokens, structured criteria, 20 options and 20-question "
        "calls. The committed [reference](benchmarks/results/reference.json) contains "
        "original FP32 logits from pinned upstream Laya. Original weights and their SHA256 "
        "are recorded in every export manifest and validation report.",
        "",
        "Failed RangeDim GPU experiments are also committed, explicitly marked `passed: false`. "
        "See [conversion findings](docs/CONVERSION.md). They are not used in the performance tables.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "pip install -e '.[convert,dev,compare]'",
        "# Export the three models and fixed Snake package as shown in README.md.",
        "python -m benchmarks.campaign --source-root /path/to/original/checkpoints",
        "python -m benchmarks.report",
        "```",
        "",
        "Portable CI covers prompt/shape/trace behavior and packaging. The native Core ML "
        "conversion/prediction test and the real-checkpoint reports were executed locally "
        "on the Mac; CI does not download large model weights.",
        "",
    ]
    (root / "BENCHMARKS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
