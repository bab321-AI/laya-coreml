# Laya-CoreML

**Laya typed decisions, running locally through Apple's Core ML.**

An independent experimental port of [Laya](https://github.com/NandhaKishorM/laya),
with a Python inference API, reproducible `.mlpackage` conversion, and local
validation on an **M3 Max / macOS 27.2**. No autoregressive generation or cloud API.
PyTorch is used for export; inference uses Core ML and the checkpoint's tokenizer.

**Neural Engine experiment: 4.98 ms P50 / 5.31 ms P95, with 2.78× lower
whole-system energy per decision than compiled MLX FP16 on this M3 Max.**
This result is for one short multilingual decision using an explicitly rewritten
ANE graph. A separately validated 8-bit weight-palette variant reaches 4.88 ms P50 and 3.19×
lower system energy per decision. The requested 10× improvement has **not** been achieved.
[Speed, energy, hardware-trace evidence and limits](docs/ANE_BENCHMARKS.md).

**All three FP16 checkpoints: 189/189 selected answers match upstream.**
Each completed 100 repeated API calls with identical rounded public results.
The default conversion uses enumerated sequence lengths after unrestricted
length ranges exposed a GPU correctness problem on this machine.

[Benchmarks](BENCHMARKS.md) · [Conversion findings](docs/CONVERSION.md) ·
[ANE implementation](docs/ANE_ENGINEERING.md) ·
[MLX sibling project](https://github.com/mizorewww/laya-mlx) · [中文](README.zh-CN.md)

## Quick start

Apple Silicon, macOS 15+, Python 3.11–3.13. This project was executed on macOS
27.2; older macOS and iPhone/iPad execution have not been validated here.

```bash
git clone https://github.com/mizorewww/laya-coreml
cd laya-coreml
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[convert]'

# Downloads the pinned original checkpoint and exports an FP16 ML Program.
laya-coreml convert laya-multilingual models/laya-multilingual
```

```python
import laya_coreml as laya

agent = laya.load("models/laya-multilingual", compute_units="cpu_gpu")
result = agent.predict(
    "I was billed twice. Please refund the duplicate.",
    {
        "department": {
            "type": "choice",
            "instructions": "Who should handle this?",
            "criteria": ["billing", "technical", "sales"],
        }
    },
)
print(result["answers"]["department"])
```

Weights and exported models stay outside Git. Conversion performs the initial
download; inference accepts a local export directory and makes no network requests.
For an inference-only environment, install this repository without `[convert]`:
neither PyTorch, Transformers, nor MLX is required.

## Run the Neural Engine experiment

From the cloned repository and export environment above:

```bash
python -m experiments.ane_engineering.probe --source laya-multilingual \
  --kind body --length 96 --output models/ane96
python -m experiments.ane_engineering.validate \
  --package models/ane96/model.mlpackage --length 96 \
  --output artifacts/ane-validation.json
```

```python
from laya_coreml.convert import resolve_source
from experiments.ane_engineering.runtime import ANEAgent

source = resolve_source("laya-multilingual")  # Pinned initial download/cache lookup.
agent = ANEAgent(source, "models/ane96/model.mlpackage", length=96)
# The same agent.predict(state, questions) API; subsequent inference stays local.
```

The experimental path keeps the original model parameters, changes activations
to `B,C,1,L`, expresses dense projections as 1×1 convolutions, splits attention by
head, and moves embedding lookup and the small action tail to CPU boundaries.
Its complete transformer graph has 6,390 NE-preferred operations. A separate
Instruments trace also records active Neural Engine prediction intervals.

L96 is a fixed, batch-one export: **59/59 fitting validation questions agree**;
longer inputs raise a capacity error. Separately exported L192 and L1024 graphs
pass **60/60** and **63/63** respectively, with 100 stable repeated API calls each.
The 4.98 ms result applies to L96; padding a short request to L1024 takes about
88 ms. This experimental path is separate from the three-checkpoint runtime above.
Shapes and approximate compression variants have their own gates in
[the engineering report](docs/ANE_ENGINEERING.md).
Simply selecting `cpu_ne` on the ordinary export does not reproduce this result.
The ANE Snake check also matches 600/600 actions with zero deaths, but its three
sequential questions do not show a consistent speedup over batched compiled MLX.
The short-question latency above is not a full-game frame-time claim.

## Fidelity

Comparison against unmodified upstream Laya FP32 on the same 63-question suite
per checkpoint: eight languages, empty/long input, literal mask tokens, structured
criteria, 20 options, multiple questions, and choice/score/noul outputs.

| Checkpoint | Core ML precision | Selected-answer agreement | Maximum calibrated probability difference |
|---|---|---:|---:|
| Laya 421M | FP16 | **63/63** | **0.003308** |
| Laya Multilingual 322M | FP16 | **63/63** | **0.001578** |
| Laya Typed Decisions 421M | FP16 | **63/63** | **0.002727** |
| Laya Multilingual 322M | FP32 diagnostic | **63/63** | **0.000000632** |

These are port-fidelity fixtures, not a claim of 100% task accuracy. Raw logits,
token IDs, action probabilities, repeatability and process RSS are recorded in
[the reports](benchmarks/results). Core ML cache/RSS measurements should not be
equated with MLX's active-tensor memory counters.

## Performance

Measured short-question FP16 latency (P50 / P95), end to end:

| Model | Core ML CPU+GPU | MLX GPU |
|---|---:|---:|
| Multilingual 322M | 11.28 / 17.38 ms | 7.87 / 9.87 ms |
| Laya 421M | 13.71 / 14.31 ms | 13.33 / 13.73 ms |
| Typed Decisions 421M | 13.70 / 14.38 ms | 13.37 / 14.39 ms |

The **ordinary export** did not outperform MLX in this run. The separate ANE
rewrite above uses a different graph and a fresh comparison against compiled MLX.
In the paired Snake test, all **600/600 actions matched**, with zero deaths;
Core ML decision P50 was 11.89–12.26 ms versus MLX's 11.53–11.69 ms.

See [BENCHMARKS.md](BENCHMARKS.md) for freshly measured Core ML vs MLX results,
compute-unit comparisons, and a headless Snake workload. The measurement includes
prompt preparation, tokenization, tensor construction, synchronous prediction,
calibration, and formatting. Loading and warmup are excluded and recorded separately.

Core ML's default export has batch size **one**. A 10-question API call therefore
performs 10 predictions; MLX batches them. The report makes that difference explicit.
Export a fixed larger batch for an application with known shapes:

```bash
# Snake's compact prompt: 3 questions, up to 64 tokens, 4 option slots.
laya-coreml convert laya-multilingual models/laya-multilingual-snake \
  --batch-size 3 --max-length 64 --max-options 4 --fixed
```

## Conversion and device choices

```bash
laya-coreml convert laya models/laya
laya-coreml convert laya-typed-decisions models/laya-typed-decisions

# A larger FP32 diagnostic export.
laya-coreml convert laya-multilingual models/laya-multilingual-fp32 --precision float32
```

The supported compute-unit options are `cpu_gpu` (API default), `all`, `cpu_ne`,
and `cpu`. `cpu_ne` permits CPU and Neural Engine; it does not force exclusive
Neural Engine execution. Compute-plan results describe anticipated scheduling,
not a hardware execution trace or energy benchmark.

Exports preserve checkpoint context limits (512 for English Laya; 1024 for the
other two). They support up to 32 option markers by default. Smaller fixed exports
reject inputs that do not fit; they do not silently shorten the prompt beyond the
original checkpoint truncation rules.

**Known failed experiment:** bounded `RangeDim` with forced GPU gave incorrect,
nonrepeatable output on the tested system, including after switching attention
implementation. The runtime blocks this combination by default. Enumerated lengths
are the validated path. Details, raw failed results, the short-input compiler fix,
and reproduction commands are in [conversion notes](docs/CONVERSION.md).

## Reproduce validation and benchmarks

```bash
pip install -e '.[convert,dev,compare]'
pytest -q

python -m benchmarks.validate models/laya-multilingual \
  --name laya-multilingual --compute-units cpu_gpu --repeats 100 \
  --output artifacts/validation.json

python -m benchmarks.run models/laya-multilingual --compute-units cpu_gpu \
  --iterations 100 --plan --output artifacts/benchmark.json

# Source root contains the three original, unconverted checkpoint directories.
# Run after exporting the three normal packages and the Snake package above.
python -m benchmarks.campaign --source-root /path/to/original/checkpoints
```

The Snake comparison imports the published laya-mlx game and policy code as a
benchmark helper; its Core ML branch executes the Core ML model. It compares both
models on identical live states, alternates their execution order, and retains the
deterministic planner features and cycle safety shield. It is not a raw-board
reasoning test or a terminal frame-rate benchmark.

Apache-2.0. See [NOTICE](NOTICE) for attribution. Independent of Convai Innovations
and Apple; the checkpoint weights retain their upstream terms.
