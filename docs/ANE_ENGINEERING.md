# Engineering investigation: moving Laya's transformer onto ANE

Research environment: Apple M3 Max (40 GPU cores, 128 GiB unified memory), macOS
27.2, Core ML Tools 9.0, PyTorch 2.7.0 and NumPy 2.1.3. This is an independent
prototype under `experiments/ane_engineering/`; the released runtime is unchanged.

## Current result

A fixed **B=1, L=96** multilingual prototype successfully assigns the complete
encoder, decision head and scorer to the Neural Engine in Core ML's anticipated
compute plan: **6,390 nonconstant operations prefer ANE**, with estimated cost
weights summing to approximately 1. The remaining 3,809 entries are constants.
The original enumerated-shape/SDPA export preferred CPU for all 1,318 assigned
operations under `CPU_AND_NE`, despite 988 individual operations listing ANE as a
supported device.

The prototype's complete prediction path measured **5.167 ms p50** over 50
screening calls, including tokenization, embedding lookup, attention masks, ANE
inference, CPU action-head computation, calibration and formatting. The isolated
Core ML body measured 4.403 ms p50 over 30 calls with synthetic embeddings. The
latter is a component measurement and is not an end-to-end speed claim. Model
loading and compilation are excluded from both warm timings.

On the fixed-length subset of the original FP32 golden reference, **59/59 answer
comparisons agree**, including eight languages and choice/score/noul questions.
The largest calibrated probability change is **0.002925**, and 100 repeated
public calls are finite and return identical rounded results. The original
63-question fixture contains three 1,024-token long inputs and one 147-token
20-option input; these four evaluations are explicitly skipped by the L96 export.
Repeated rubrics occur in this fixture. These are regression comparisons, not 59
independent labeled examples or proof of unchanged general task accuracy.

Separate **L192 and L1024** exports also put all 6,390 assigned body operations on
ANE. L192 passes **60/60** comparisons; L1024 passes the complete **63/63** golden
fixture. Both pass 100 repeated public calls and the maximum calibrated-probability
error remains 0.002925. The long-input subset itself has maximum error 0.001128.
All evaluated token-usage counts match the reference.

| Fixed sequence capacity | Evaluated / total fixture questions | Body p50, synthetic inputs | Complete short-question p50 at this capacity | Initial compile/load |
|---|---:|---:|---:|---:|
| 96 | 59 / 63 | 4.403 ms | 5.167 ms | 18.77 s |
| 192 | 60 / 63 | 7.136 ms | 8.178 ms | 19.59 s |
| 1024 | 63 / 63 | 78.405 ms | 88.433 ms | 22.55 s |

These are serial screening runs, not paired comparisons across backends. Body
timings use 5 warmup and 30 measured calls; complete short-question timings use
10 warmup and 50 measured calls. The complete path includes host work and input
checks. The L96 screen predates the final extra input-validation checks; the
final controlled comparison uses the current adapter and records its source
fingerprint. Compile/load times are measured in each process after conversion,
not a promise about first-ever system startup with empty framework caches.
Large fixed graphs perform padded work even for short requests. A practical
adapter would select separate length buckets; routing every request through
L1024 would discard the short-input advantage.

A separate run of the actual `workload(1, long=True)` fixture confirms one
**1,024-token** request, rather than a short request padded to that size. It
measures **91.703 ms p50 / 94.776 ms p95** over 50 complete predictions after ten
warmup calls; all rounded outputs remain stable. The
[historical MLX long-input benchmark](../benchmarks/results/perf-multilingual-long-mlx.json)
is 51.98 ms p50. Those are not same-round paired measurements, but this
screen provides no evidence that the present ANE graph accelerates long inputs.
The input hash, actual token length, current experiment fingerprints and raw
timings are retained in
[long1024-performance.json](../experiments/ane_engineering/long1024-performance.json).

Raw evidence:

- [Complete-body compute plan and component timings](../experiments/ane_engineering/body96/report.json)
- [Real-input FP32-reference validation and complete-predict timings](../experiments/ane_engineering/validation96.json)
- [L192 validation](../experiments/ane_engineering/validation192.json)
- [L1024 complete-fixture validation](../experiments/ane_engineering/validation1024.json)
- [Single-layer plan and numeric checks](../experiments/ane_engineering/layer96/report.json)
- [Single-MLP plan and numeric checks](../experiments/ane_engineering/mlp96/report.json)

`MLComputePlan` describes anticipated placement, not a hardware execution trace.
`CPU_AND_NE` permits CPU and ANE; it is not an ANE-only switch. In this experiment
all assigned heavy-body operations prefer ANE, but runtime hardware telemetry is
evaluated separately. The subsequent Instruments diagnostic recorded Neural
Engine hardware activity; its table is global and cannot attribute every event
to this model. The paired MLX comparison, power-integration results and trace
limitations are reported in [ANE_BENCHMARKS.md](ANE_BENCHMARKS.md). A zero-valued
ANE counter from a monitoring tool cannot establish an absence of ANE activity
without validating that counter on this OS/device.

## Why the original graph was a poor ANE target

The baseline preserves a conventional B×L×C transformer layout, dynamic shape
operations, whole-head batched attention, and Core ML's SDPA operator. Under
CPU/ANE selection, its plan contains 24 SDPA operations without a reported device
assignment, plus many casts, slices, shape queries, gathers and transposes. Some
individual operations support ANE, but the graph as a whole is not partitioned
onto it. Device support for individual operators is therefore insufficient
evidence of a useful ANE execution path.

The successful prototype changes several things together. It is evidence that
the combination enables ANE placement, **not a completed ablation identifying one
sole offending operator**. Fixed-shape original-layout SDPA and explicit-attention
controls are useful next discriminating experiments.

Apple's published Transformer guidance recommends channel-first 4D tensors,
1×1 convolutions for projections, per-head attention and fewer layout copies.
These principles motivated the implementation; Apple's historical DistilBERT
speedups do not establish a 10× improvement against this project's already-fast
MLX FP16 baseline. [Apple's ANE Transformer article](https://machinelearning.apple.com/research/neural-engine-transformers),
[Apple's reference implementation](https://github.com/apple/ml-ane-transformers).

## Prototype architecture and numerical contract

[model.py](../experiments/ane_engineering/model.py) is a separate export model
built from the original checkpoint parameters:

- Hidden activations use **B,C,1,L**. Each dense weight `W[out,in]` becomes a 1×1
  convolution kernel `K[out,in,0,0]` without retraining or weight approximation.
- Attention is split into individual 64-channel heads. The key tensor is
  transposed once, and two explicit einsums compute QK and AV while retaining the
  4D layout. Softmax runs over the key axis, dimension 1.
- RoPE splits each head into its two 32-channel halves. Its cosines, sines and
  bases come from the original model, including multilingual local theta 160000.
- Channel normalization preserves the original `normalized * weight + bias`
  ordering and epsilon. It does not copy the differently ordered affine
  expression or optional clipping in Apple's reference LayerNorm.
- The first encoder attention norm remains the identity; the encoder uses exact
  erf GELU, while the two decision-head FFNs retain ReLU.
- Full attention and sliding-window masks preserve valid-key masking and the
  padded-query rule. The local radius is read from `local_attention // 2`.
- The final marker gather is expressed using an externally prepared one-hot
  selector and a 4D einsum. The graph keeps 32 marker slots, including inactive
  slots; the host replaces inactive logits with the original -1e4 value.

An entire original FP32 encoder layer and its BC1S counterpart differed by at
most 2.29e-5 in the PyTorch layout check. FP16 Core ML layer output error was
larger, as expected for this precision/backend change. The body probe originally
contained a self-comparison; that invalid evidence was removed and its whole-body
PyTorch layout check is explicitly `not_measured`. Real full-model validation is
instead against the stored original FP32 logits and decisions.

The independent CPU regressions in
[test_ane_layout.py](../tests/test_ane_layout.py) additionally compare a complete
tiny `ConvBody` with the original `DecisionModel`, using explicit and SDPA
attention oracles, three question types, changed padding values, nonzero norm
biases, multiple RoPE bases and a nondefault norm epsilon. These tests cover
layout and masking semantics without conflating them with full-checkpoint FP16
hardware precision. All five layout tests passed locally.

The attention prototype adds a finite -1e4 mask bias. This has the intended mask
behavior on the finite validated activations, but it is not a bitwise identity to
replacing masked scores with -1e4 or -infinity for arbitrary extreme inputs.
Likewise, Core ML FP16 execution is not claimed to be bitwise identical to the
original FP32 model.

[runtime.py](../experiments/ane_engineering/runtime.py) supplies one CPU→ANE→CPU
boundary for the whole transformer, rather than a device transition per layer:

1. The CPU tokenizes each question, gathers only requested embedding rows and
   constructs fixed-shape additive masks, type vectors and marker selectors.
   It does not reuse contextual hidden states or K/V between questions.
2. One Core ML call performs embedding normalization, all 22 encoder layers,
   both decision-head layers and scoring convolutions.
3. The CPU derives action features from **uncalibrated raw-logit softmax** and
   runs the small original action head in FP32 with erf GELU. Public calibration
   and output formatting then use the existing implementation.

The exported body has FP16 compute, while the small host action head is FP32.
Embedding lookup uses the original stored weights. The source safetensors file
contains 169 FP16 tensors and one FP32 tensor: "FP32 reference" describes the
original PyTorch execution, not a claim that the original checkpoint is stored
entirely in FP32. This mixed-precision boundary is part of the prototype's
numerical contract. Action probability differences of zero on saturated fixtures
do not prove action-logit identity.

The adapter rejects incompatible package dimensions, out-of-range IDs/markers,
invalid mask values, empty attention-key rows and input lengths exceeding its
fixed capacity. Its default input limit is 96 tokens and its batch size is one;
multiple questions execute sequentially. It never silently truncates a request
to fit the shorter export.

## Reproduction, provenance and quality gates

Run from the repository root in the pinned `.venv`. Generated packages are
ignored by Git; no duplicate embedding NPZ or large weight artifact is required.
`probe.py` refuses an existing output directory. New exports record original
weight/config SHA256 values, package content hashes, shape, tool versions and
experiment-source fingerprints in a manifest.

The reproduction commands write under `artifacts/ane-repro/`, because the
committed experiment directories already contain reports and manifests. Choose
another new directory when repeating an export; neither exporter silently
reuses an existing package.

Install the conversion, development and compression-research dependencies with
`pip install -e '.[convert,dev,research]'` (or the corresponding `uv sync`
extras). The research extra pins `kmeans1d==0.4.0`; this optional dependency is
used for the FP16 grouped K-means experiments and recorded in new manifests.

By default, conversion resolves `laya-multilingual` to the repository's pinned
Hugging Face checkpoint and downloads it when necessary. Pass `--source /path/to/checkpoint`
to use existing local files. Validation defaults to the source directory recorded
in the package manifest. The `ANEAgent` runtime itself only accepts local files;
it verifies the original weight/config hashes, fixed shape and package content
hash before loading. Validation also rejects a golden reference with a different
source-weight hash. The measured multilingual source-weight SHA256 is
`9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204`.

```bash
# Small placement probes, then the complete model.
.venv/bin/python -m experiments.ane_engineering.probe \
  --kind mlp --length 96 --output artifacts/ane-repro/mlp96
.venv/bin/python -m experiments.ane_engineering.probe \
  --kind layer --length 96 --output artifacts/ane-repro/layer96
.venv/bin/python -m experiments.ane_engineering.probe \
  --kind body --length 96 --output artifacts/ane-repro/body96
.venv/bin/python -m experiments.ane_engineering.validate \
  --package artifacts/ane-repro/body96/model.mlpackage \
  --length 96 --repeats 100 \
  --output artifacts/ane-repro/validation96.json
```

The validator requires all evaluated argmax decisions to match, calibrated and
action probability errors <=0.02, finite outputs, unchanged token usage, and
identical repeated public outputs. A failed candidate writes `passed: false` and
exits unsuccessfully. Skipped cases remain unvalidated even if the fixed-shape
subset passes. The initial L96 report had these gates applied explicitly after
measurement; it is marked accordingly, without changing its recorded timings.

Longer fixed exports and their complete golden-reference validation commands:

```bash
.venv/bin/python -m experiments.ane_engineering.probe \
  --kind body --length 192 --output artifacts/ane-repro/body192
.venv/bin/python -m experiments.ane_engineering.validate \
  --package artifacts/ane-repro/body192/model.mlpackage --length 192 \
  --output artifacts/ane-repro/validation192.json
.venv/bin/python -m experiments.ane_engineering.probe \
  --kind body --length 1024 --output artifacts/ane-repro/body1024
.venv/bin/python -m experiments.ane_engineering.validate \
  --package artifacts/ane-repro/body1024/model.mlpackage --length 1024 \
  --output artifacts/ane-repro/validation1024.json
.venv/bin/python -m experiments.ane_engineering.benchmark \
  --package artifacts/ane-repro/body1024/model.mlpackage --length 1024 --long \
  --output artifacts/ane-repro/long1024-performance.json
```

These commands reproduce the independently checked longer exports listed above.
Their placement and numeric fidelity were checked separately from the L96 graph;
padding short requests to 1024 tokens is not the proposed production policy.

## Compression screening and the 10× objective

[palettize.py](../experiments/ane_engineering/palettize.py) prepares independent
weight-only palette variants, with 8-bit uniform lookup tables as the inexpensive
first screening candidate. Only convolution weights larger than 2048 elements
are selected; RoPE constants, normalization, activation math and host action
weights remain unchanged. Grouped output channels use separate lookup tables.
K-means is a more expensive mode. It uses the installed `kmeans1d` implementation
for these FP16 groups. Core ML Tools 9.0 parallelizes independent groups via a
process-pool `starmap` when `num_kmeans_workers > 1`; the experiments use eight
workers for offline K-means and one math-library thread per worker. Worker count
changes export throughput, not the intended codebook objective. Lower-bit
6/4-bit candidates are separate approximate artifacts, not exact implementations.

Compression sizes refer to the **exported transformer-body package**. The
original 196.608-million-entry embedding table remains on the host, with only
requested rows looked up for each request, and the small host action weights
remain unchanged. A body package shrinking by roughly twofold is not a twofold
reduction in the entire checkpoint, runtime memory, or per-request energy.

```bash
.venv/bin/python -m experiments.ane_engineering.palettize \
  --package artifacts/ane-repro/body96/model.mlpackage \
  --bits 8 --mode uniform --group-size 32 \
  --output artifacts/ane-repro/body96-w8
.venv/bin/python -m experiments.ane_engineering.validate \
  --package artifacts/ane-repro/body96-w8/model.mlpackage --length 96 \
  --output artifacts/ane-repro/validation96-w8.json

# Independent K-means candidates; inspect each validation exit status.
for bits in 8 6 4; do
  VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    .venv/bin/python -m experiments.ane_engineering.palettize \
    --package artifacts/ane-repro/body96/model.mlpackage \
    --bits "$bits" --mode kmeans --group-size 32 --workers 8 \
    --output "artifacts/ane-repro/body96-w${bits}km"
  .venv/bin/python -m experiments.ane_engineering.validate \
    --package "artifacts/ane-repro/body96-w${bits}km/model.mlpackage" --length 96 \
    --output "artifacts/ane-repro/validation96-w${bits}km.json"
done
```

The two uniform W8 screens both retain all 59 fixture argmax decisions and pass
100 repeated calls, but **fail the probability gate**: group size 32 reaches
0.023612 error and group size 4 reaches 0.033858. The W8 K-means/group32 screen
passes with maximum error **0.014393** and 59/59 decisions. Saturated action
probabilities hide action-logit differences up to 16.24 for that K-means candidate;
passing this small regression fixture does not establish preserved general
calibration or task accuracy. Compressed candidates remain separately identified
approximate models.

The fixed W6 K-means/group32 candidate also keeps 59/59 argmax decisions and
stable repeated outputs, but **fails** with maximum probability error **0.052243**.
Its maximum action-logit error is 76.62. Reducing weights to six bits therefore
does not satisfy the unchanged acceptance gate, even though its short-request
screening latency remains close to FP16.

W4 K-means/group32 also keeps 59/59 decisions, but maximum probability error
rises to **0.200221** and maximum action-logit error to **605.30**. It fails the
same gate. The absence of argmax changes in all five compressed screens shows
why this fixture's saturated decisions alone are an inadequate acceptance test.

| L96 variant | Body package, decimal MB | Maximum calibrated-probability error | Complete-predict screening p50 | Quality gate |
|---|---:|---:|---:|---|
| [FP16](../experiments/ane_engineering/validation96.json) | 251.91 | 0.002925 | 5.167 ms | Pass |
| [W8 uniform, group 32](../experiments/ane_engineering/validation96-w8.json) | 129.29 | 0.023612 | 4.923 ms | Fail |
| [W8 uniform, group 4](../experiments/ane_engineering/validation96-w8g4.json) | 146.06 | 0.033858 | 5.420 ms | Fail |
| [W8 K-means, group 32](../experiments/ane_engineering/validation96-w8km.json) | 129.29 | 0.014393 | 4.792 ms | Pass |
| [W6 K-means, group 32](../experiments/ane_engineering/validation96-w6km.json) | 96.23 | 0.052243 | 4.841 ms | Fail |
| [W4 K-means, group 32](../experiments/ane_engineering/validation96-w4km.json) | 64.52 | 0.200221 | 5.001 ms | Fail |

All compressed variants retain 6,390 NE-preferred device-assigned operations,
59/59 fixture argmax agreements and 100 stable repeated calls. The remaining
plan entries include constants and weight-LUT reconstruction expressions;
placement metadata alone does not prove how much compressed data travels from
DRAM during a request. The three K-means exports take 186.50, 56.13 and 23.90
seconds respectively with eight offline workers. Setup and worker processes
finish before every inference measurement.

These 50-call serial screens do **not** establish speedup ratios. The FP16
screen predates the last input-validation checks, and the screens are not
interleaved. W8 K-means is the sole compressed finalist for the stronger
same-session comparison in [ANE_BENCHMARKS.md](ANE_BENCHMARKS.md). The rejected
variants are retained as evidence of the precision boundary, not recommended
deployments. Compression has only been validated for this multilingual L96
subset; the 63-question L1024 result above concerns the separate FP16 export.

Core ML palette compression reconstructs floating weights from indexed lookup
tables; smaller stored tensors do not alone establish faster inference or lower
energy. Every variant needs the same precision gates, a fresh compute plan and a
paired end-to-end speed/energy comparison. [Core ML palettization documentation](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-api.html).

The initial successful ANE placement establishes a credible optimization path,
not a 10× result. The comparison must use MLX FP16, including its compiled variant
where faster, and report equal task scope. Power must be integrated over complete
requests. Both gross system energy and any idle-subtracted estimate should be
reported with their measurement limitations. The independent mathematical review
is [ANE_MATH.md](ANE_MATH.md).

## What the manual implementation establishes

The useful manual work here is a complete, independently validated rewrite of
the computation graph into a layout that Core ML can map to ANE. This changes
execution placement while retaining the trained parameters. It is substantially
more effective than changing `compute_units` on the original graph. It does not
remove the 24 sequential attention/MLP blocks or their dense projection work.

The next exact candidate for long requests is attention that actually visits
only local windows, implemented using fixed query/key tiles and the original
padding and RoPE rules. The current graph still computes a dense score matrix
and applies a local mask. A tiled graph could reduce that work, but more slices,
boundaries and small contractions may undermine ANE scheduling; its placement,
precision and end-to-end benefit remain unmeasured. Further weight compression
requires calibration or quality recovery after the failures above. Distillation
or fewer layers would introduce a new model and require a broader task-quality
evaluation. None of those unimplemented directions supplies evidence of a 10×
gain today.
