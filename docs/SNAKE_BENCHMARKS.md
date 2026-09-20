# Core ML Snake release benchmark

**The ANE FP16 demo sustained 49.1–50.0 fresh decisions/s in three uncapped
600-step episodes, with zero deaths.** Its pooled throughput was **49.66/s**.
Each move includes three sequential model questions, planner work and truecolor
terminal serialization. This is the complete active game loop, not the separate
single-question ~5 ms microbenchmark.

For paced operation, **20 requested decisions/s** was the highest tested setting
to pass the active-computation deadline criterion. The observed wall-clock rate
was **18.39–18.42/s**, including real sleep and scheduling overhead. This result
does not establish a perfectly timed 20 FPS display. The README recording uses
12 requested decisions/s for legibility, observing 11.39/s in its real terminal.

## Environment and method

- Apple M3 Max, 40-core GPU, 128 GiB, macOS 27.2, Python 3.12.13.
- Installed `laya-coreml==0.1.0` wheel in a fresh environment; Core ML Tools 9.0,
  NumPy 2.1.3, Rich 15.0.0; no Torch, MLX or Transformers installed.
- Public ANE FP16 bundle at revision
  `39d6a9b3d0f67f06da74fbade6121ea134cbdb21`, downloaded before execution.
- B1/L96/K32, three sequential questions per game decision, compact prompt,
  24×16 board, six initial snake cells. Twenty warmup decisions excluded.
- Default visible cycle safety layer; raw top-1 is also measured separately.
- Synchronous `predict`, game features, Rich composition, truecolor ANSI
  serialization and game update included. Loading, warmup and terminal-emulator
  painting excluded. Paced runs include actual `sleep` calls.

Passing requires finite outputs, zero deaths, preserved cycle order and food
progress, plus **at most 1% of active ticks exceeding the requested time budget**
in every paced episode. Uncapped mode waits for a fresh prediction on each move
and has no fixed deadline. No other model benchmark ran concurrently.

The runtime, checkpoint, package hash, prompt hash and per-tick measurements are
recorded in [coreml-snake.json](../benchmarks/results/coreml-snake.json).

## Uncapped stability

| Seed | Moves | Observed decisions/s | Score / final length | Deaths | Safety interventions |
|---:|---:|---:|---:|---:|---:|
| 101 | 600 | 49.10 | 20 / 26 | 0 | 1 |
| 102 | 600 | 49.89 | 24 / 30 | 0 | 0 |
| 103 | 600 | 49.99 | 23 / 29 | 0 | 1 |

Across these 1,800 decisions, the three-question API latency was **16.32 ms P50 /
21.33 ms P95**. Full active-tick latency was **19.07 ms P50 / 25.96 ms P95**.
This is bounded survival evidence with explicit assistance, not a claim that the
model can play indefinitely without a safety layer.

## Paced sweep and confirmation

Every sweep uses seed 7 for 120 moves. Failed rates are retained:

| Requested decisions/s | Observed decisions/s | Active deadline misses | Result |
|---:|---:|---:|---|
| 20 | 18.55 | 0.83% | Pass; confirmed below |
| 30 | 28.67 | 10.83% | Fail |
| 40 | 37.22 | 37.50% | Fail |
| 50 | 44.21 | 53.33% | Fail |
| 60 | 50.68 | 100.00% | Fail |

Longer confirmation at the 20/s setting:

| Seed | Moves | Observed decisions/s | Active deadline misses | Score | Result |
|---:|---:|---:|---:|---:|---|
| 101 | 600 | 18.39 | 4 / 600, 0.67% | 20 | Pass |
| 102 | 600 | 18.42 | 4 / 600, 0.67% | 24 | Pass |
| 103 | 600 | 18.42 | 3 / 600, 0.50% | 23 | Pass |

The three-question API was approximately 26.3 ms P50 in these paced episodes,
compared with 16.3 ms during the uncapped episodes. The experiment establishes a
pacing-dependent difference, but does not isolate its cause. Scheduling and
device power-state transitions are possible explanations that need separate
profiling; they are not proven by this report. A sustained throughput result
therefore must not be advertised as a fixed-frame deadline guarantee.

## Raw top-1 and the shareable recording

With the safety override disabled, seeds 101/102/103 each completed 200 moves,
scoring 7/6/7 with zero deaths. The model still receives the same exact planner
features, so these runs do not test reasoning from an unprocessed board. They
also do not establish long-game survival.

Across all benchmark modes there were **4,920 decisions, zero deaths and four
safety interventions**. Separately, the real-terminal showcase recorded 855
decisions over 75.034 seconds, final score 26, length 32, zero deaths and zero
interventions. Its original timestamps, states and actions are tested by exact
deterministic replay. See [LAUNCH.md](LAUNCH.md) for the GIF, MP4 and provenance.

## Reproduce

```bash
pip install 'laya-coreml[demo]==0.1.0'
hf download aac6fef/laya-multilingual-coreml-ane \
  --revision 39d6a9b3d0f67f06da74fbade6121ea134cbdb21 \
  --local-dir models/ane
laya-coreml-snake benchmark --model ./models/ane \
  --rates 20,30,40,50,60 --sweep-steps 120 --soak-steps 600 \
  --seeds 101,102,103 --raw-steps 200 --output snake-benchmark.json
```

To play at the observed unconstrained rate, use
`laya-coreml-snake --model ./models/ane --max-speed`. Actual terminal painting can
reduce the rate relative to the serialization benchmark. The separate Snake GPU
bundle batches all three questions; this release report measures the ANE FP16
bundle only. Historical paired model comparisons remain in
[ANE_BENCHMARKS.md](ANE_BENCHMARKS.md) and [BENCHMARKS.md](../BENCHMARKS.md).
