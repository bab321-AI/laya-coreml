"""Prepare portable, checksum-verified bundles for a subsequent hf CLI upload."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from laya_coreml.artifacts import file_digest, tree_digest, verify_files, verify_research_manifest
from laya_coreml.convert import REVISIONS

ROOT = Path(__file__).resolve().parents[1]
HOST_KEYS = ["encoder.embeddings.tok_embeddings.weight", "type_emb.weight"] + [
    "act_head." + name for name in ("0.weight", "0.bias", "2.weight", "2.bias")
]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def inventory(root):
    return {
        str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": file_digest(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def model_card(repo, config, variant):
    length = config["shape"]["max_length"]
    ane = config["format"] == "laya-coreml-ane"
    snake = variant == "snake"
    device = "CPU + Neural Engine" if ane else "CPU + GPU"
    extra = (
        "This is a separately identified **8-bit grouped K-means weight-palette** variant. "
        "Activations still use FP16, and the host action head uses FP32. It is approximate: "
        "maximum calibrated-probability drift on the 59-question short fixture is 0.014393. "
        "The long-context fixture has not been validated for this compressed package."
        if config.get("compression")
        else "This FP16 conversion retains the original trained parameters. The ANE graph's host "
        "action head uses FP32. Floating-point results can differ from the upstream FP32 runtime."
        if ane
        else "This FP16 export retains the original model architecture and decision schema. "
        "The enumerated-length GPU export is the validated general-purpose configuration."
    )
    fidelity = (
        "The matching fixed B3/L64 Snake export agreed with MLX on 600/600 actions across two "
        "300-step trajectories, with zero deaths and zero safety interventions. It is a "
        "specialized 64-token export, not the full-context general-purpose model."
        if snake
        else "The fixed L96 package agrees with upstream on 59/59 fitting fixture questions and "
        "passes 100 repeated calls. Inputs longer than 96 tokens raise an error instead of "
        "being silently shortened to fit."
        if ane
        else "The complete 63-question fixture agrees with upstream selected answers, with 100 "
        "stable repeated calls. The release bundle is checked again after packaging."
    )
    return f"""---
license: apache-2.0
library_name: coreml
pipeline_tag: text-classification
base_model: {config["source"]}
tags:
- coreml
- laya
- apple-silicon
- decision-model
- local-ai
{"- neural-engine" if ane else "- modernbert"}
---

# {repo.split("/")[-1]}

**Laya typed decisions on Apple Silicon, using {device}.**
This is a portable Core ML bundle for [laya-coreml](https://github.com/mizorewww/laya-coreml),
converted from [{config["source"]}](https://huggingface.co/{config["source"]}).
It outputs `choice`, `score`, and `noul` probabilities with **zero generated tokens**.
Inference needs no PyTorch, Transformers, MLX, remote code, or cloud API.

## Run

Apple Silicon, macOS 15+, Python 3.11–3.13. Tested on M3 Max / macOS 27.2.

```bash
pip install laya-coreml
```

```python
import laya_coreml as laya

agent = laya.load("{repo}")  # Download once; Core ML runs locally.
result = agent.predict(
    "The customer asks for a refund of a duplicate payment.",
    {{"refund": {{"type": "noul", "instructions": "Does the customer request a refund?"}}}},
)
print(result["answers"])
```

To download explicitly and then run entirely offline:

```bash
hf download {repo} --local-dir models/{variant}
pip install 'laya-coreml[demo]'
laya-coreml-snake --model models/{variant} --fps 12
```

Use `laya.load("{repo}", local_files_only=True)` for a cached snapshot or pass a
local directory. Use `revision="<Hub commit SHA>"` to pin a remote revision.

## Format and fidelity

{extra}

{fidelity}

The exported capacity is **{length} total tokens**, batch **{config["shape"]["batch_size"]}**,
and **{config["shape"]["max_options"]}** option slots. Questions/options and state share this budget.
The ANE short exports reject over-capacity prompts. Snake uses planner features and a
visible optional cycle safety shield; survival is not a claim of unaided game intelligence.

`coreml_config.json` records shapes, source revisions and per-file SHA256 checksums.
`validation.json` contains the packaging-time validation. Port fidelity on this regression
suite does not establish general task accuracy or preserved calibration on arbitrary inputs.

## Performance and limits

The multilingual **ANE L96 FP16** runtime measured **4.98 / 5.31 ms P50 / P95** for
one short question on M3 Max; W8 measured **4.88 / 5.23 ms**. Whole-system energy per
decision improved **2.78× / 3.19×**, respectively, against compiled MLX FP16 in that
experiment. Those numbers apply to the named short ANE variants, not every bundle,
long contexts, or complete Snake frames. The requested 10× improvement was not achieved.

[Measurements and scope](https://github.com/mizorewww/laya-coreml/blob/main/docs/ANE_BENCHMARKS.md)
· [General Core ML benchmarks](https://github.com/mizorewww/laya-coreml/blob/main/BENCHMARKS.md)
· [Snake demo](https://github.com/mizorewww/laya-coreml/blob/main/docs/SNAKE_DEMO.md).

## Provenance

- Original checkpoint: `{config["source"]}` at `{config["source_revision"]}`.
- Original weights SHA256: `{config["source_weights_sha256"]}`.
- Upstream implementation: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya),
  commit `6a5819129eb220570792e417e49723d697efd76f`.
- Original models and code are by Convai Innovations and contributors, Apache-2.0.
- Independent conversion; not an official Convai Innovations or Apple release.

See `LICENSE` and `NOTICE`. Model quality and task/language limitations originate
with Laya; this runtime is an inference port, not a newly trained decision model.
"""


def finish(output, config, repo, variant):
    config["repository"] = repo
    config["package_sha256"] = tree_digest(output / "model.mlpackage")
    config["files"] = inventory(output)
    save(output / "coreml_config.json", config)
    verify_files(output, config["files"])
    (output / "README.md").write_text(model_card(repo, config, variant))
    for name in ("LICENSE", "NOTICE"):
        shutil.copyfile(ROOT / name, output / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", default="aac6fef")
    parser.add_argument("--output", type=Path, default=ROOT / "models/hub")
    parser.add_argument("--source-root", type=Path, default=ROOT.parent / "laya-mlx/models")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    standard = [
        ("laya-enumerated-fp16", "laya-coreml", "validation-laya-enumerated-cpu_gpu.json"),
        (
            "multilingual-enumerated-fp16",
            "laya-multilingual-coreml",
            "validation-multilingual-enumerated-cpu_gpu.json",
        ),
        (
            "typed-enumerated-fp16",
            "laya-typed-decisions-coreml",
            "validation-typed-enumerated-cpu_gpu.json",
        ),
        ("laya-multilingual-snake", "laya-multilingual-coreml-snake", "snake.json"),
    ]
    for local, name, validation in standard:
        original = ROOT / "models" / local
        config = json.loads((original / "coreml_config.json").read_text())
        previous = json.loads((ROOT / "benchmarks/results" / validation).read_text())
        if config["files"] != previous["export"]["files"]:
            raise ValueError(f"Bundle differs from validated export: {local}")
        verify_files(original, config["files"])
        output = args.output / name
        output.mkdir(exist_ok=False)
        for entry in ("model.mlpackage", "tokenizer", "encoder"):
            shutil.copytree(original / entry, output / entry)
        shutil.copyfile(original / "rl_agent_config.json", output / "rl_agent_config.json")
        upstream = Path(config["source"]).name
        config.update(source=f"convaiinnovations/{upstream}", source_revision=REVISIONS[upstream])
        finish(
            output,
            config,
            f"{args.account}/{name}",
            "snake" if local.endswith("-snake") else upstream,
        )
        print(f"Prepared {name}", flush=True)
    source = args.source_root / "laya-multilingual"
    for experiment, suffix in (("body96", "ane"), ("body96-w8km", "ane-w8")):
        package = ROOT / "experiments/ane_engineering" / experiment / "model.mlpackage"
        manifest = verify_research_manifest(source, package, length=96)
        output = args.output / f"laya-multilingual-coreml-{suffix}"
        output.mkdir(exist_ok=False)
        shutil.copytree(package, output / "model.mlpackage")
        for folder in ("tokenizer", "encoder"):
            (output / folder).mkdir()
        for name in ("encoder/config.json", "rl_agent_config.json"):
            shutil.copyfile(source / name, output / name)
        for name in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copyfile(source / "tokenizer" / name, output / "tokenizer" / name)
        with safe_open(source / "model.safetensors", framework="numpy") as original:
            tensors = {key: original.get_tensor(key) for key in HOST_KEYS}
            save_file(tensors, output / "host_weights.safetensors")
            with safe_open(output / "host_weights.safetensors", framework="numpy") as exported:
                for key in HOST_KEYS:
                    np.testing.assert_array_equal(
                        exported.get_tensor(key), original.get_tensor(key)
                    )
        config = {
            "format": "laya-coreml-ane",
            "format_version": 1,
            "source": "convaiinnovations/laya-multilingual",
            "source_revision": REVISIONS["laya-multilingual"],
            "source_weights_sha256": manifest["source_files_sha256"]["model.safetensors"],
            "precision": "float16",
            "host_action_precision": "float32",
            "compression": manifest.get("compression"),
            "minimum_deployment_target": "macOS15 / iOS18",
            "shape": {
                "batch_size": 1,
                "max_length": 96,
                "min_length": 96,
                "max_options": 32,
                "flexible": False,
                "lengths": None,
            },
            "host_tensor_count": len(HOST_KEYS),
            "host_tensors_exactly_equal_to_source": True,
            "versions": manifest["versions"],
        }
        finish(output, config, f"{args.account}/{output.name}", suffix)
        if config["package_sha256"] != manifest["package_sha256"]:
            raise ValueError("Published ANE graph changed during packaging")
        print(f"Prepared {output.name}; verified all {len(HOST_KEYS)} host tensors", flush=True)


if __name__ == "__main__":
    main()
