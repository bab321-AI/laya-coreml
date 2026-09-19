# Conversion notes

The export loads **original FP32 Laya checkpoints**, strictly checks all state-dict
keys, traces an inference-only PyTorch implementation, and saves a Core ML ML Program.
No training, pruning or weight quantization is performed. FP16 is a conversion
precision choice; FP32 can be selected for diagnostics.

The runtime uses the checkpoint's tokenizer, prompt layout, option markers,
question-type embedding, decision head, action head, and calibration temperatures.
Choice, score, noul, structured criteria, token accounting, and zero generated
tokens follow the upstream API. The encoder is bidirectional: every question
still runs its own encoder sequence. There is no shared-state hidden-state cache.

## Validated conversion choices

- `coremltools==9.0`, `torch==2.7.0`, `numpy==2.1.3`, Python 3.12.
- TorchScript tracing with graph checking, evaluation mode, original FP32 weights.
- ML Program, macOS 15 / iOS 18 deployment target. Actual execution was tested
  on an M3 Max running macOS 27.2; iPhone/iPad and older macOS execution were not tested.
- Default sequence lengths are selected from 16, 32, 64, 96, 128, 192, 256, 384,
  512, 768, 1024, capped by the checkpoint's context limit. The runtime pads to
  the smallest available length and masks those added tokens.
- Default batch size is one, with 32 marker slots. More questions run in chunks.
  `--batch-size` and `--max-options` produce different exported signatures.
- Fixed shapes are available for a known workload. Inputs exceeding an export's
  length or option capacity raise an error; they are not silently truncated to
  fit a smaller export. Original checkpoint context truncation is preserved.

Apple documents [TorchScript conversion](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html)
and [enumerated input shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html).
Multiple enumerated inputs need the same number of shapes, matched by index;
this export pairs input IDs and attention masks accordingly.

## Failures retained for reproducibility

These are observations on this machine and OS, not claims about every Core ML version.

1. PyTorch's `__or__` boolean operator was not converted. Explicit
   `torch.logical_or` / `torch.logical_and` preserve the same mask semantics.
2. NumPy 2.5 rejected a deprecated array-to-scalar conversion inside coremltools
   9.0. The supported project dependency is pinned below NumPy 2.2. PyTorch was
   pinned to the converter's tested 2.7.0 version instead of 2.7.1.
3. `RangeDim` with forced `CPU_AND_GPU` produced large numerical errors and
   different results on repeated identical inputs. The original SDPA export
   matched only 47/63 reference answers, and explicit matmul/softmax attention
   matched 20/63. FP32 did not resolve the observed short-input GPU failure.
   CPU / automatic selection gave correct outputs for the SDPA graph.
4. Enumerated lengths restored GPU fidelity and repeatability. A separate tiny
   regression test then exposed an MPSGraph compiler `SIGTRAP` when slicing a
   constant boolean local-attention matrix. The diagnostic named
   `ElementsAttr::getValues<bool>` / `FoldStridedSliceOp`.
5. The final implementation slices **integer positions** and constructs the
   boolean local mask afterwards. This removes the compiler trap. It does not
   cure the general RangeDim GPU failure: the subsequent experiment still
   matched only 49/63 and was not repeatable. Enumerated lengths remain the default.

The runtime rejects RangeDim + `cpu_gpu` unless explicitly allowed for a
diagnostic experiment. To reproduce that failed configuration:

```bash
laya-coreml convert laya-multilingual models/range-experiment --shape-mode range
python -m benchmarks.validate models/range-experiment \
  --name laya-multilingual --compute-units cpu_gpu --allow-unvalidated-gpu \
  --repeats 10 --output artifacts/range-experiment.json
```

The test is expected to fail on the measured environment. Raw failed and
successful reports are retained in `benchmarks/results/`; reports containing
`"passed": false` must not be cited as validated configurations.

## Device evidence

`CPU_AND_NE` means CPU and Neural Engine are *allowed*, not that every operator
runs on the Neural Engine. The benchmark records the Core ML compute plan's
preferred/supported devices and estimated costs. This is an anticipated plan,
not an Instruments runtime hardware trace, power measurement, or proof of
exclusive Neural Engine execution.

## Reproducibility

Every export includes `coreml_config.json`: original weight SHA256, source
revision, shapes, precision, attention implementation, tool versions, conversion
time and hashes of every package/tokenizer/config file. Exports refuse to
overwrite existing directories. A failed export removes only its newly created
output directory.

The committed golden reference was generated from unmodified upstream Laya
revision `6a5819129eb220570792e417e49723d697efd76f` using FP32 PyTorch MPS.
It includes full input token IDs and unrounded logits. Validation compares those
tokens exactly and checks selected answers, calibrated probabilities, action
probabilities, token accounting, and repeated public results.

To regenerate the golden reference in an upstream-compatible environment:

```bash
git clone https://github.com/NandhaKishorM/laya .upstream
git -C .upstream checkout 6a5819129eb220570792e417e49723d697efd76f
python -m benchmarks.reference --upstream .upstream --model-root /path/to/original/checkpoints
```

The exact reference dependencies are recorded in the generated JSON. They are
separate from the pinned export environment; Transformers is not a runtime or
export dependency of laya-coreml.
