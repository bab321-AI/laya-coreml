# ANE feasibility: equivalent transformations and a measurable 10× target

Research date: 2026-09-20. Hardware: M3 Max, 40-core GPU, 128 GiB. This document describes hypotheses and mathematical bounds, not a claim of a new measured speedup. The starting point is the original Laya weights and the faster MLX FP16 runtime. The engineering experiments and measurements may supersede the starting observations below.

The best first experiment is a fixed-shape, channel-first rewrite of the entire transformer, followed by an execution-plan inspection. The most plausible order-of-magnitude target is **energy per completed decision**, provided the model keeps useful latency and accuracy. Changing a Core ML compute-unit setting alone is not sufficient evidence that the Neural Engine executed the model.

The subsequent [engineering experiments](ANE_ENGINEERING.md) implemented that rewrite,
and [the measured speed/energy report](ANE_BENCHMARKS.md) now records the results.
The uncompressed candidate improves efficiency but has not met the 10× target.
The hypotheses and original denominator below are retained as the research record;
use the later compiled-MLX comparison for measured performance claims.

## Define the target before optimizing

For the same inputs, checkpoint, precision policy and number of completed decisions, define:

```text
t = elapsed time / completed decisions
P = average measured power over that same interval
E = integrated energy / completed decisions = P × t

S = t_MLX / t_candidate                    speed gain
R = P_MLX / P_candidate                    power reduction factor
S × R = E_MLX / E_candidate                 energy efficiency gain
```

Use block mean latency, not a latency percentile, in this identity. Report P50 and P95 separately. A result that is 2× faster at one fifth of the power is a 10× energy improvement. A result that is half as fast needs a 20× power reduction to deliver the same 10× energy improvement. Multiplying speed by an already-computed energy improvement double-counts elapsed time.

There are two distinct power tests:

1. Saturated sequential inference: measure actual throughput, latency and joules per decision. A lower-power but slower candidate is not automatically more efficient.
2. Equal offered load, such as the same Snake tick rate: both candidates must finish the same work within the deadline. Report average power, total interval energy, deadline misses and completed decisions. Sleeping longer or dropping work is not an optimization.

Record the measured power domain. CPU + GPU + ANE telemetry is not necessarily whole-machine or battery power, and must not be labelled as such. Report raw energy and, if usable, paired idle-subtracted energy. When load-minus-idle is similar to noise, preserve the uncertainty instead of silently clamping it and reporting an enormous ratio. Keep tokenization, input copies, CPU fallback and postprocessing inside the endpoint's accounting boundary.

## Starting evidence and the denominator

The current [multilingual MLX benchmark](../benchmarks/results/perf-laya-multilingual-mlx.json) measures a single 91-token question at **7.870 ms P50 / 9.870 ms P95**. The [English MLX benchmark](../benchmarks/results/perf-laya-mlx.json) measures a 93-token question at **13.334 / 13.734 ms**. These are end-to-end `predict` measurements, excluding model loading and warmup. A new comparison must rerun the strongest applicable MLX path, including its opt-in compile and prompt-cache settings where the workload permits them; a historical eager result is not a permanent denominator.

The existing Core ML multilingual export measures [11.277 ms P50 with CPU + GPU](../benchmarks/results/perf-laya-multilingual-coreml.json), [78.037 ms with ALL](../benchmarks/results/perf-multilingual-all.json), and [81.336 ms with CPU + NE](../benchmarks/results/perf-multilingual-cpu_ne.json). The latter plan records 1,318 CPU-preferred operations and no NE-preferred operations; 24 SDPA operations have unknown device metadata. This establishes no NE execution claim. The plan lists many operations as NE-*supported*, which is different from NE-*preferred*. Apple describes [compute-plan device usage](https://developer.apple.com/documentation/coreml/mlcomputeplandeviceusage) as anticipated device use, so even a favourable plan should be corroborated by runtime profiling or observable NE activity.

The existing FP16 regression gate is 100% fixture argmax agreement, finite deterministic outputs, and at most 0.02 absolute drift in calibrated and action probabilities. That is a conversion-fidelity check on a small corpus. It does not establish general task accuracy, Snake competence or the quality of a compressed model.

## Equivalent graph transformations

Apple's [Transformer deployment study](https://machinelearning.apple.com/research/neural-engine-transformers) motivates four-dimensional `BC1L` activations, 1×1 convolutions, splitting attention into heads, and reducing layout copies. Its published 10× example is a different model, device and baseline; it cannot be transferred to the MLX comparison here. Treat those layout recommendations as candidates to test on this OS and chip, not a complete current hardware support contract.

### Linear projections and the gated MLP

Let `X[b,l,i]` be the existing activation and define `U[b,i,0,l] = X[b,l,i]`. For a linear layer,

```text
Y[b,l,o] = sum_i W[o,i] X[b,l,i] + bias[o]
K[o,i,0,0] = W[o,i]
Conv2D(U,K)[b,o,0,l] = Y[b,l,o]
```

This changes the layout and operator representation without changing the real-valued function. Keep that layout across the entire encoder and both decision-head transformer layers. Converting to and from it around every linear layer can erase the benefit. QKV can remain a single `D → 3D` convolution; split its channel output into Q, K and V. Likewise, preserve the existing fused `D → 2I` encoder projection, split channels into value and gate, apply the original exact GELU to value, multiply by gate, and project `I → D`.

For FP16, a sequence width divisible by 32 also aligns to the 64-byte last-axis alignment described in Apple's study. The initial shapes are `B=1,L=96` for the short API fixture and `B=3,L=64` for compact Snake. A multiple-of-32 recommendation here follows that buffer model; it is not permission to pad every workload to a large arbitrary length. Snake does not need 96 tokens.

### Attention and RoPE

For each head, keep Q and V as `(B,d,1,L)` and transpose K to `(B,L,1,d)`. Compute:

```text
score[b,k,0,q] = sum_c Q[b,c,0,q] K[b,k,0,c] / sqrt(d)
weight = softmax(score + additive_mask, axis=key)
out[b,c,0,q] = sum_k weight[b,k,0,q] V[b,c,0,k]
```

The key axis is axis 1 in this representation. Concatenate head outputs on the channel axis. This is the same attention function; a wrong softmax axis silently changes it. Apple's [reference attention implementation](https://github.com/apple-aiml-research/ml-ane-transformers/blob/main/ane_transformers/reference/multihead_attention.py) demonstrates the corresponding two four-dimensional contractions. Inspect the converted MIL operators: writing an `einsum` does not guarantee the intended device or lowering.

Apply RoPE to each head's channel pairs before QK. Preserve the checkpoint's split-half convention, original positions and per-layer theta. In this multilingual checkpoint, both full and local RoPE use theta 160000. Splitting the first and second halves of *all heads concatenated together* is incorrect; split within each 64-channel head. A position-dependent rotation cannot generally be folded into a single position-independent weight matrix.

The local rule is bidirectional `abs(q-k) <= 64`, inclusive. Preserve key padding and the existing padded-query rule. Constant position-dependent parts can be computed before tracing for fixed shapes; changing sample padding must still affect the key mask. Replacing mathematical exclusion with a finite large negative mask is a numerical approximation unless it matches the original implementation's finite-precision behavior; verify adversarial and padded inputs.

### LayerNorm is not interchangeable with other normalization

For each `(b,l)`, reduce across channels only:

```text
mu = mean_c U
v = mean_c (U-mu)^2
normalized = (U-mu) / sqrt(v + epsilon)
output = normalized * gamma + beta
```

Keep the original epsilon, affine order, population variance, first-layer identity normalization, exact GELU and residual order. Apple's [reference LayerNorm](https://github.com/apple-aiml-research/ml-ane-transformers/blob/main/ane_transformers/reference/layer_norm.py) uses a different affine order; its [DistilBERT adapter](https://github.com/apple-aiml-research/ml-ane-transformers/blob/main/ane_transformers/huggingface/distilbert.py) compensates by transforming the bias. Directly copying that class and loading Laya's state dict would be incorrect for nonzero biases. An explicit original-order affine expression also avoids dividing by a possibly zero gamma.

Clamping activations, switching GELU to tanh, or replacing LayerNorm with RMSNorm changes the function. If squared values overflow, a positive rescaling is a mathematically equivalent option:

```text
normalize(x/a, epsilon/a^2) = normalize(x, epsilon), for a > 0
```

Finite-precision accumulation still needs parity tests. FP32 reductions may cost copies or CPU fallback, so inspect the plan rather than silently relaxing numerical behavior.

### Move unsupported work to the model boundaries

If embedding lookup, dynamic marker gather or the action tail prevents a contiguous NE region, make a separate candidate with this partition:

```text
CPU: tokenizer → selected embedding rows → embedding LayerNorm
NE candidate: all encoder layers → type embedding addition → both heavy head layers
CPU: marker/CLS selection → small scorer → raw-probability features → action head
```

Only the endpoints cross engines. Do not offload every layer's attention or LayerNorm to the CPU. At multilingual `B=1,L=96`, an FP16 embedding tensor is 147,456 bytes; at `B=3,L=64` it is 294,912 bytes. Include these copies and any full-hidden-state output copy in end-to-end measurement.

The multilingual token table has 196,608,000 parameters but each request gathers only its token rows. It need not be shipped into an ANE transformer subgraph, and it must not be counted as a full table read on every prediction. Its LayerNorm has no position dependence, so offline pre-normalization of table rows is real-arithmetic equivalent. It may change rounding and storage precision, requiring its own parity check. The action head consumes probabilities from **raw** marker logits, before temperature calibration; reconstructing its features from public calibrated probabilities changes the checkpoint behavior.

## Arithmetic limits on a 10× latency claim

Let `D` be hidden width, `I` gated encoder intermediate width, `N` encoder layers, and `H=2` decision-head layers. The main per-token matrix parameter count and dense arithmetic are:

```text
A = N(4D^2 + 3DI) + 12HD^2
F_dense(B,L) = 2BLA + 4B(N+H)L^2D
```

Multiply and add count separately. These equations exclude norms, activations, embeddings, scoring, masks, copies and runtime overhead. They are not a profiler. They model the current dense attention computation even for masked local layers.

| Checkpoint | D / I / N | A | FP16 main matrix bytes | Dense work at B=1,L=96 | Required effective compute for 10× below current MLX P50 |
| --- | --- | ---: | ---: | ---: | ---: |
| Multilingual | 768 / 1152 / 22 | 124,452,864 | 248.91 MB | 24.574 GFLOP | 31.23 TFLOP/s within 0.787 ms |
| English / typed architecture | 1024 / 2624 / 28 | 368,312,320 | 736.62 MB | 71.848 GFLOP | 53.89 TFLOP/s within 1.333 ms, using the English baseline |

These are required achieved rates, not asserted ANE peak specifications. A layout rewrite removes overhead but does not remove those dense projections. The hardware must also execute a sequential chain of 24 or 30 attention/MLP blocks.

An optimistic streaming model gives another conditional floor:

```text
t >= max(F / effective_compute, bytes_from_DRAM / effective_bandwidth)
```

The 40-GPU-core M3 Max is specified with [400 GB/s unified-memory bandwidth](https://support.apple.com/en-om/117737). **If** each main FP16 matrix is fetched from DRAM once per request, even full access to that bandwidth costs at least 0.622 ms for multilingual and 1.842 ms for English. Actual ANE bandwidth access may be smaller, and cached or compressed weights change the assumption. This is not an unconditional physical lower bound. It shows why 10× English latency is particularly demanding under uncompressed streaming, and why measuring energy is useful even when latency improves modestly.

For a measured fraction `f` of end-to-end time improved by factor `s`, Amdahl's law gives `S = 1 / (1-f+f/s)`. Even infinite acceleration of one region cannot reach 10× unless it occupies at least 90% of the original latency. The analogous bound uses the fraction of measured **energy**, not FLOPs, when targeting joules per decision. The final-head selected-query optimization removes only a few percent of model arithmetic; local attention sparsity is also negligible at `L<=64`, where the local window covers all positions. Neither provides a credible standalone 10× route.

## Compression and architectural changes have different contracts

| Candidate | Same real-valued checkpoint function? | What it can realistically change |
| --- | --- | --- |
| BC1L layout, 1×1 projections, static positions/masks, head splitting | Yes, when equations and inputs are preserved | Scheduling, locality, compiler partitioning, memory copies |
| CPU endpoint partition, offline embedding norm, selected queries in final head | Yes in real arithmetic; validate rounding | Unsupported operations, package footprint, some unused work |
| 8/6/4-bit palettization or weight quantization | Generally no | Weight traffic/storage and possibly inference energy/latency |
| Pruning learned nonzero weights or low-rank factorization | No, unless algebraically exact structure exists | Matrix arithmetic and traffic after recovery/calibration |
| Early exit, token pruning, fewer layers, narrower student | No | Potentially large savings; new model and quality contract |
| Hidden-state reuse across arbitrary questions | No for this bidirectional encoder | Invalid shortcut; contextual states depend on the question |
| Caching identical whole-input answers | Exact for cache hits | Workload feature; not uncached inference speed |

Apple's current [optimization overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html) points to palettization for NE memory/latency gains and identifies the newer W8A8 compute path with A17 Pro/M4. Do not extrapolate that newer hardware speedup to this M3 Max. The [quantization performance guide](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html) also warns that activation dequantization can slow CPU/GPU execution. First obtain an NE-resident baseline, then test weight palettization from 8 bits downward while preserving sensitive norms/scorers as appropriate.

Packing FP16 weights to eight or four bits gives an ideal weight-storage ratio of 2× or 4× before metadata. That is not an equal latency multiplier: decompression, activation movement and computation remain. Pruning only helps runtime if the chosen representation actually exploits the zeros. Removing arbitrary heads/layers or performing low-rank truncation requires quality recovery and cannot retain the original model identity without qualification.

For a per-input logit error bound `||z'-z||_infinity <= delta`, a sufficient argmax certificate is `top1(z)-top2(z) > 2delta`. With identical positive calibration temperature `T`, softmax's infinity-norm Lipschitz bound gives `||p'-p||_infinity <= delta/(2T)`. These are useful diagnostics on evaluated inputs, not a global certificate for quantization. Preserve separate close-margin and multilingual evaluation slices; saturated examples can hide large logit errors.

## Three experiments and acceptance gates

1. **Fixed-shape equivalent NE graph.** Export multilingual `B=1,L=96,K=4` and Snake `B=3,L=64,K=4` with BC1L projections, correct per-head RoPE, explicit attention and original LayerNorm/GELU. Compare input arrays and individual layer outputs against the original graph. Inspect which major projections and attention blocks are NE-preferred, then check actual runtime NE activity. A supported-op count alone is not success. Re-run the corresponding optimized MLX baseline in alternating blocks.
2. **One contiguous transformer island.** If the first graph fragments, move embedding and the small final tail to CPU boundaries. Compare this against the full-graph candidate under the same power measurement. Keep a candidate only if complete predictions improve latency or energy beyond observed run-to-run variation. Include all copies; a fast isolated encoder is insufficient.
3. **Energy-focused compression after placement works.** Screen 8-bit palettization, then 6/4-bit as separate approximate variants. Run the unchanged fixture gate, held-out choice/score/noul tasks, multilingual inputs, near ties and Snake trajectories. Publish accuracy, probability drift and calibration alongside performance. Aggressive compression or distillation belongs in a separately named model if it changes learned behavior.

Before a launch claim, use the same checkpoint/input hashes and an alternating baseline/candidate order; exclude cold compilation but report it separately. Use at least five sustained blocks per finalist and retain raw latency, completed-call and power samples. Require the original fixture agreement and existing probability tolerances without loosening them to pass the candidate, plus stable finite outputs and bounded memory. Measure long and shape-boundary inputs separately from short fixed-shape demos.

Declare 10× only when the relevant speed, equal-load power, or energy ratio is at least ten with uncertainty that supports the claim, while meeting the stated latency and task-quality limits. If the lower uncertainty bound does not reach ten, report the measured ratio. A smaller energy gain with verified NE execution remains useful evidence; it is not an order-of-magnitude result.

## Documentation provenance

Used the required Context7 CLI workflow: resolved `Core ML Tools` to `/apple/coremltools`, then queried transformer operator/layout lowering and NE compression behavior (three commands total). Checked Apple's research article, reference source, current Core ML optimization documentation, compute-plan documentation and the device specification linked above. Model equations, parameter counts and starting measurements were derived from this repository and its MLX sibling. No competing GPU/ANE benchmark was run by this research branch.
