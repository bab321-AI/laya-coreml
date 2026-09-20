"""Screen conv-only weight palettization as an independent research artifact."""

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
import coremltools.optimize as cto

from benchmarks.common import environment, save

from .artifact import tree_digest, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package", type=Path, default=Path("experiments/ane_engineering/body96/model.mlpackage")
    )
    parser.add_argument("--mode", choices=["uniform", "kmeans"], default="uniform")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits", type=int, choices=[4, 6, 8], default=8)
    parser.add_argument("--workers", type=int, default=1, help="Offline K-means group workers")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    model = ct.models.MLModel(str(args.package), skip_model_load=True)
    source_manifest = json.loads((args.package.parent / "manifest.json").read_text())
    if tree_digest(args.package) != source_manifest["package_sha256"]:
        raise ValueError("Source package content does not match manifest")
    config = cto.coreml.OptimizationConfig(
        op_type_configs={
            "conv": cto.coreml.OpPalettizerConfig(
                mode=args.mode,
                nbits=args.bits,
                granularity="per_grouped_channel",
                group_size=args.group_size,
                num_kmeans_workers=args.workers,
                weight_threshold=2048,
            )
        }
    )
    started = time.perf_counter()
    compressed = cto.coreml.palettize_weights(model, config)
    args.output.mkdir(parents=True)
    compressed.save(str(args.output / "model.mlpackage"))
    report = {
        "environment": environment(),
        "source_package": str(args.package),
        "source_package_sha256": tree_digest(args.package),
        "compression": {
            "mode": args.mode,
            "nbits": args.bits,
            "granularity": "per_grouped_channel",
            "group_size": args.group_size,
            "num_kmeans_workers": args.workers,
            "operator_selection": "conv only; threshold >2048; norms/RoPE/attention left unchanged",
        },
        "conversion_seconds": time.perf_counter() - started,
        "package_bytes": sum(
            p.stat().st_size for p in (args.output / "model.mlpackage").rglob("*") if p.is_file()
        ),
        "package_sha256": tree_digest(args.output / "model.mlpackage"),
        "quality_status": "not_evaluated",
        "note": "Compression alone does not establish task precision, ANE placement or a speed/energy win. Run validation and compute-plan inspection separately.",
    }
    manifest = write_manifest(
        source_manifest["source_directory"],
        args.output / "model.mlpackage",
        kind=source_manifest["kind"],
        length=source_manifest["shape"]["length"],
        extra={
            "compression": report["compression"],
            "source_package_sha256": report["source_package_sha256"],
        },
    )
    report["versions"] = manifest["versions"]
    save(args.output / "compression.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
