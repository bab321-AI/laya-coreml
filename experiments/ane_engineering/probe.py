"""Convert and inspect isolated fixed-shape ANE candidates without changing runtime."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import coremltools as ct
import numpy as np
import torch

from benchmarks.common import compute_plan, environment, save, stats
from laya_coreml.convert import resolve_source
from laya_coreml.torch_model import load_model

from .artifact import write_manifest
from .model import ConvBody, ConvEncoderLayer, ConvMLP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="laya-multilingual")
    parser.add_argument("--kind", choices=["mlp", "layer", "body"], default="layer")
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument(
        "--compute-units", choices=["cpu_ne", "cpu_gpu", "all", "cpu"], default="cpu_ne"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()
    if args.length < 1 or args.iterations < 1:
        parser.error("length and iterations must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to reuse existing research output: {args.output}")
    source_path = resolve_source(args.source)
    torch.set_num_threads(4)
    torch.manual_seed(20260920)
    source = load_model(source_path, args.length, attention_implementation="explicit")
    width = source.encoder.embeddings.tok_embeddings.weight.shape[1]
    x = torch.randn(1, width, 1, args.length)
    mask = torch.zeros(1, args.length, 1, args.length)
    if args.kind == "mlp":
        model = ConvMLP(source.encoder.layers[1].mlp)
        inputs = {"hidden": x}
        with torch.inference_mode():
            expected = (
                source.encoder.layers[1]
                .mlp(x.squeeze(2).transpose(1, 2))
                .transpose(1, 2)
                .unsqueeze(2),
            )
    elif args.kind == "layer":
        model = ConvEncoderLayer(source.encoder.layers[0], args.length)
        inputs = {"hidden": x, "mask": mask}
        with torch.inference_mode():
            expected = (
                source.encoder.layers[0](
                    x.squeeze(2).transpose(1, 2),
                    torch.ones(1, 1, args.length, args.length, dtype=torch.bool),
                )
                .transpose(1, 2)
                .unsqueeze(2),
            )
    else:
        model = ConvBody(source, args.length)
        marker_map = torch.zeros(1, args.length, 1, 32)
        marker_map[:, 0] = 1
        positions = torch.arange(args.length)
        local = torch.where(
            (positions[:, None] - positions[None, :]).abs() <= source.encoder.window // 2,
            0.0,
            -1e4,
        )[None, :, None, :]
        inputs = {
            "embeddings": x,
            "full_mask": mask,
            "local_mask": local,
            "type_vectors": torch.zeros(1, width, 1, 1),
            "marker_map": marker_map,
        }
        expected = None
    model.eval()
    with torch.inference_mode():
        actual = model(*inputs.values())
        actual = actual if isinstance(actual, tuple) else (actual,)
        parity = (
            [float((a - b).abs().max()) for a, b in zip(actual, expected)]
            if expected is not None
            else None
        )
        traced = torch.jit.trace(model, tuple(inputs.values()), strict=True, check_trace=True)
    report = {
        "environment": environment(),
        "kind": args.kind,
        "length": args.length,
        "compute_units": args.compute_units,
        "pytorch_layout_max_abs_error": parity,
        "pytorch_layout_parity_status": "measured_against_original"
        if expected is not None
        else "not_measured",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "model.mlpackage"
    if not target.exists():
        started = time.perf_counter()
        converted = ct.convert(
            traced,
            source="pytorch",
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(name=name, shape=value.shape, dtype=np.float16)
                for name, value in inputs.items()
            ],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
            skip_model_load=True,
        )
        converted.save(str(target))
        report["artifact_manifest"] = write_manifest(
            source_path, target, kind=args.kind, length=args.length
        )
        report["conversion_seconds"] = time.perf_counter() - started
        save(args.output / "report.json", report)
    units = {
        "cpu_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "all": ct.ComputeUnit.ALL,
        "cpu": ct.ComputeUnit.CPU_ONLY,
    }[args.compute_units]
    started = time.perf_counter()
    coreml = ct.models.MLModel(str(target), compute_units=units)
    report["load_compile_seconds"] = time.perf_counter() - started
    report["compute_plan"] = compute_plan(
        SimpleNamespace(model=coreml, compute_units=args.compute_units)
    )
    save(args.output / "report.json", report)
    print(
        "PLAN",
        report["compute_plan"]["preferred_operation_counts"],
        report["load_compile_seconds"],
        flush=True,
    )
    if not args.skip_predict:
        arrays = {name: value.numpy().astype(np.float16) for name, value in inputs.items()}
        for _ in range(5):
            output = coreml.predict(arrays)
        values = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            output = coreml.predict(arrays)
            values.append((time.perf_counter() - started) * 1e3)
        report["timing"] = {**stats(values), "samples_ms": values}
        report["outputs"] = {
            name: {"shape": list(value.shape), "finite": bool(np.isfinite(value).all())}
            for name, value in output.items()
        }
        if expected is not None and len(expected) == 1 and len(output) == 1:
            value = next(iter(output.values())).astype(np.float32)
            ref = expected[0].numpy()
            report["coreml_max_abs_error"] = float(np.max(np.abs(value - ref)))
            report["coreml_rmse"] = float(np.sqrt(np.mean((value - ref) ** 2)))
        save(args.output / "report.json", report)
        print(
            "TIMING",
            report["timing"]["p50_ms"],
            "ERROR",
            report.get("coreml_max_abs_error"),
            flush=True,
        )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["environment", "compute_plan", "timing"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
