# Neural Engine speed and energy measurements

**The FP16 ANE rewrite improves complete-predict speed by 1.39× and estimated
whole-system energy per decision by 2.78× against compiled MLX FP16. The separately
validated W8 K-means candidate reaches 1.42× and 3.19× respectively. Neither meets
the requested 10× target.** These are local M3 Max results, with the precision and
power-measurement limits below.

The implementation and conversion experiments are in
[ANE_ENGINEERING.md](ANE_ENGINEERING.md); mathematical bounds and fidelity
contracts are in [ANE_MATH.md](ANE_MATH.md).

## Final sustained short-decision comparison

M3 Max, 40 GPU cores, 128 GiB, macOS 27.2. Same multilingual source checkpoint,
same eight invoice-state variants, one four-option question per request, 91 real
tokens padded to 96. No output cache or generated tokens. All three models stay
resident; loading, compilation and warmup are outside the measured intervals.

The baseline enables **`mx.compile`, prompt-prefix caching, and 32-token shape
buckets**. It is faster than the historical eager MLX baseline. The FP16 ANE
candidate preserves the original weights, with a complete Core ML transformer
body, host embedding lookup and FP32 CPU action tail. W8 uses grouped K-means
**weight-only palette compression**, not W8A8 arithmetic; its activations still
use FP16 and its host action tail remains FP32.

| Metric | MLX GPU FP16, compiled | ANE FP16 | ANE W8 K-means |
|---|---:|---:|---:|
| Completed decisions | 17,184 | 23,961 | 24,453 |
| Measured active duration | 120.02 s | 120.02 s | 120.02 s |
| End-to-end P50 | 6.937 ms | **4.976 ms** | **4.879 ms** |
| End-to-end P95 | 7.393 ms | **5.307 ms** | **5.227 ms** |
| Mean elapsed time per decision | 6.984 ms | 5.009 ms | 4.908 ms |
| Mean whole-system power estimate | 61.39 W | 30.75 W | 27.39 W |
| Whole-system energy per decision | 0.4288 J | 0.1540 J | 0.1344 J |
| Idle-subtracted energy per decision | 0.3393 J | 0.0888 J | 0.0715 J |
| Speed gain vs compiled MLX | 1× | **1.394×** | **1.423×** |
| Whole-system energy gain vs compiled MLX | 1× | **2.784×** | **3.189×** |
| Idle-subtracted energy gain vs compiled MLX | 1× | 3.823× | 4.749× |

Ratios use interval means, including all completed work:

```text
FP16: speed 1.394 × average system-power ratio 1.997 = energy gain 2.784
W8:   speed 1.423 × average system-power ratio 2.241 = energy gain 3.189
```

Multiplying an energy ratio by speed again double-counts elapsed time. The
idle-subtracted row uses a different measurement boundary; it does not mean
whole-laptop power falls by 3.8–4.7×. These are saturated-throughput intervals.
An equal-request-rate experiment would be needed for a fixed-FPS power claim.
W8 improves mean speed only about **2.1%** over FP16 in this session, while its
body package shrinks from 251.91 to 129.29 decimal MB. Package size excludes the
original checkpoint/host embedding table and is not total runtime memory.

Each implementation ran six 20-second blocks in three balanced cycles of
`MLX, ANE FP16, ANE W8, ANE W8, ANE FP16, MLX`. Ten seconds of idle sampling separate
blocks; the first three idle seconds are discarded, and adjacent settled idle
power is averaged. Backend block power ranges were 60.32–62.38 W for MLX,
29.28–35.14 W for FP16 ANE, and 26.68–27.95 W for W8 ANE. Other desktop applications
were open; alternation and adjacent idle sampling reduce but do not eliminate
background-load and sensor uncertainty.

All **65,598 predictions** match their backend's rounded warmup output for the
corresponding state. All three backends select the same answer on these eight
states. The separate fidelity suite is broader: FP16 L96 passes 59/59 fitting
questions with maximum probability drift 0.002925; W8 passes those same 59 with
drift 0.014393 under the unchanged 0.02 gate. The complete 63/63 result belongs
to the separately exported FP16 L1024 model. These are regression fixtures, not
a claim of general task accuracy or preserved calibration on arbitrary inputs.

The run retained **1,101** power samples; maximum gap was **0.510 seconds** and
maximum recorded system power was **74.47 W**. No sample was discarded. Per-block
RSS is recorded for the shared process with all models and recording buffers
resident; those values cannot be assigned to an individual backend's model memory.
The current runtime and package fingerprints are stored with the report.

[Final raw calls and PSTR samples](../benchmarks/results/ane-energy-finalists.json) ·
[Audited summary and within-session bootstrap intervals](../benchmarks/results/ane-energy-summary.json)

The earlier [FP16-only pilot](../benchmarks/results/ane-energy-fp16.json) measured
1.410× speed and 2.939× gross system-energy gain. It predates the last input checks
and per-call stability instrumentation; the fresh three-arm run above is the
published comparison for the final runtime. One metadata caveat in the final raw
report initially described the historical macmon component-sum floor unconditionally;
an additive `metadata_corrections` entry clarifies that this run used PSTR only.
The original fields and all measurements are preserved.

## Snake compatibility is a separate workload

The same published Snake planner, compact prompts and safety policy ran on both
the FP16 ANE adapter and compiled MLX, alternating evaluation order on identical
live states. Seeds 101 and 102 each completed 300 steps: **600/600 proposed and
executed actions agree, zero deaths and zero shield interventions**. Final scores
were 9 and 10, with snake lengths 15 and 16. Maximum move-probability difference
was 0.0036. This validates this FP16 trajectory comparison, not compressed-model
Snake behavior or unlimited survival.

| Seed | ANE complete decision P50 / P95 | Compiled MLX complete decision P50 / P95 |
|---|---:|---:|
| 101 | 23.45 / 27.10 ms | 17.14 / 23.58 ms |
| 102 | 17.68 / 27.30 ms | 19.18 / 33.27 ms |

The ANE adapter runs three questions sequentially at B1/L96; MLX batches the
three questions at up to L64. These timings include planner features and complete
predictions, exclude the other backend's work, game step and terminal rendering,
and show substantial variation. They do **not** establish a consistent Snake
speedup or a maximum stable rendering rate. The single-question 4.98 ms result
must not be advertised as the full Snake frame time. A dedicated B3/L64 ANE export
would be a separate optimization and validation task.

[Raw Snake states, outputs and timings](../benchmarks/results/ane-snake.json)

```bash
python -m benchmarks.snake artifacts/ane-repro/body96/model.mlpackage \
  /path/to/original/laya-multilingual --ane --mlx-compiled \
  --steps 300 --seeds 101 102 --output artifacts/ane-snake.json
```

## Measurement boundary and telemetry limitations

Every timed call includes prompt construction, tokenization, array construction,
host embedding lookup where applicable, synchronous model execution, action
features, calibration and output formatting. MLX evaluates its lazy outputs;
Core ML returns completed NumPy arrays. This compares uncached predictions,
not isolated model kernels.

The final comparison uses the small, unprivileged
[PSTR-only sampler](../benchmarks/pstr_sampler/README.md). It pins the low-level
SMC API from macmon 0.8.2, opens one read-only connection, and samples the original
`PSTR` value every 500 ms. It does not read IOReport component counters. This is
a whole-system sensor estimate, not an external wall-power or battery measurement.
The executable hash and source provenance are recorded with the raw results.

This separate sampler was necessary because the official macmon CLI computes
`sys_power = max(PSTR, component_sum)`, as shown in its
[source](https://github.com/vladkens/macmon/blob/v0.8.2/src_lib/metrics.rs).
CPU and ANE counters usually returned zero on this OS, then one sample jumped to
approximately 38,021 W CPU and 2,068 W ANE. The component floor propagated that
fault into a 40,089 W system reading. The precise cause of the IOReport anomaly
is unresolved; the original PSTR value cannot be recovered from that sample.
**The entire affected energy run was rejected**, without removing or clipping
the bad sample. Its [raw record](../benchmarks/results/ane-energy-rejected-telemetry.json)
remains available, but its stored energy aggregates must not be used as results.

The earlier FP16-only run used the official macmon v0.8.2 release, whose archive
SHA256 was verified as
`588d5bde79885ba36f693e5150911c10c3ad208a2e418a3f2aa827ac84a2d973`.
It passed the later sanity checks and remains historical evidence. The final
PSTR-only run measures every backend again with one consistent sampler.

The harness integrates watts over monotonic receive timestamps with interpolated
boundaries. Missing, nonpositive, non-finite or above-500 W system readings reject
the run. The 500 W ceiling is a deliberately loose sanity check for this M3 Max,
not a calibrated accuracy bound. Gaps above `max(2 seconds, 3 × sample interval)`
also reject the run. The summary verifies complete balanced cycles, per-call
stability, call counts, active energy, both adjacent idle intervals and the
resulting idle-subtracted energy against the raw samples. Incremental energy
retains its sign; it is never clamped to manufacture a large ratio.

The direct sampler omits CPU/GPU/ANE power fields because it does not measure
them. Historical missing or zero component counters cannot establish zero energy
or the absence of ANE execution. The bootstrap resamples complete balanced cycles;
three cycles from one desktop session do not characterize all background load,
future runs, or sensor accuracy. None of the measured ratios is close to 10×.

## Does the Neural Engine actually execute work?

The rewritten B1/L96 graph's Core ML compute plan places all **6,390 nonconstant
operations** on `MLNeuralEngineComputeDevice`; the remaining 3,809 unknown entries
are constants. This is anticipated placement, not sufficient hardware evidence
by itself. The ordinary SDPA export had no NE-preferred operations on this system.

A separate 15.97-second Instruments **Core ML** trace, taken while the candidate
ran with `CPU_AND_NE`, captured **3,124 active “Neural Engine Prediction” intervals**
and one unrelated cached system-model load. The exported hardware table therefore
provides positive runtime ANE activity evidence in addition to the compute plan.
The table is global and does not attach a PID or model identity to every prediction;
the Core ML model-signpost table was empty in this recording. We do not attribute
every hardware interval exclusively to Laya or claim that host work runs on ANE.

[Allowlisted hardware events](../benchmarks/results/ane-hardware-events.json)
include only timestamps, duration, device, label and state. Full Instruments
archives and process environment metadata stay in the ignored `artifacts/`
directory. Tracing was separate from the energy test, and instrumented hardware
intervals are not used as the end-to-end latency result.

## Reproduce

First create and validate the fixed L96 FP16 and W8 K-means packages with the
commands in [the engineering report](ANE_ENGINEERING.md). Those commands write
to fresh directories under `artifacts/ane-repro/`. Build the direct sensor sampler
locally; no system daemon or privileged service is installed.

```bash
pip install -e '.[convert,dev,compare,research]'
cargo build --release --locked --manifest-path benchmarks/pstr_sampler/Cargo.toml
python -m benchmarks.energy \
  --source /path/to/original/laya-multilingual \
  --candidate artifacts/ane-repro/body96-w8km/model.mlpackage \
  --fp16-candidate artifacts/ane-repro/body96/model.mlpackage \
  --candidate-factory experiments.ane_engineering.runtime:ANEAgent \
  --sampler benchmarks/pstr_sampler/target/release/pstr-sampler --pstr-only \
  --cycles 3 --seconds 20 --idle-seconds 10 \
  --output artifacts/energy.json

python -m benchmarks.energy_summary artifacts/energy.json \
  --output artifacts/energy-summary.json
```

For independent runtime diagnostics, start `benchmarks.trace_ane`, wait for its
ready PID file, then attach Instruments without another model workload running:

```bash
python -m benchmarks.trace_ane \
  --source /path/to/original/laya-multilingual \
  --package artifacts/ane-repro/body96/model.mlpackage \
  --seconds 90 --ready artifacts/trace.pid
# From another terminal; use the PID written to that file.
xcrun xctrace record --template 'Core ML' --attach <PID> \
  --time-limit 15s --output artifacts/ane.trace
xcrun xctrace export --input artifacts/ane.trace --toc \
  --output artifacts/toc.xml
xcrun xctrace export --input artifacts/ane.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="ane-hw-intervals"]' \
  --output artifacts/hardware.xml
python -m benchmarks.trace_summary --hardware-xml artifacts/hardware.xml \
  --toc-xml artifacts/toc.xml --output artifacts/hardware-summary.json
```

The original checkpoint and shorter export have separate context limits. The L96
runtime rejects inputs that do not fit; a short-workload speed result does not
establish performance at 512/1024 tokens or across all three Laya checkpoints.
