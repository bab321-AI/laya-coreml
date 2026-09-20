"""Bind experimental packages to exact source files, shapes and package bytes."""

import hashlib
import importlib.metadata
import json
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(path):
    path = Path(path)
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file():
            digest.update(str(file.relative_to(path)).encode())
            digest.update(bytes.fromhex(file_digest(file)))
    return digest.hexdigest()


def write_manifest(source, package, *, kind, length, extra=None):
    source, package = Path(source), Path(package)
    with (source / "model.safetensors").open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_length))
    manifest = {
        "format": "laya-ane-research",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "shape": {"batch": 1, "length": length, "options": 32},
        "source_directory": str(source),
        "source_files_sha256": {
            name: file_digest(source / name)
            for name in ("model.safetensors", "encoder/config.json", "rl_agent_config.json")
        },
        "source_tensor_dtype_counts": dict(
            Counter(value["dtype"] for key, value in header.items() if key != "__metadata__")
        ),
        "package_sha256": tree_digest(package),
        "package_bytes": sum(p.stat().st_size for p in package.rglob("*") if p.is_file()),
        "precision": "FP16 Core ML body, original embedding lookup, FP32 CPU action head",
        "versions": {
            name: importlib.metadata.version(name) for name in ("coremltools", "torch", "numpy")
        },
        "experiment_python_sha256": {
            path.name: file_digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    try:
        manifest["versions"]["kmeans1d"] = importlib.metadata.version("kmeans1d")
    except importlib.metadata.PackageNotFoundError:
        pass
    if extra:
        manifest.update(extra)
    (package.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_manifest(source, package, *, length):
    source, package = Path(source), Path(package)
    manifest = json.loads((package.parent / "manifest.json").read_text())
    if manifest.get("format") != "laya-ane-research" or manifest.get("version") != 1:
        raise ValueError("Unsupported ANE research artifact manifest")
    if manifest.get("kind") != "body" or manifest.get("shape") != {
        "batch": 1,
        "length": length,
        "options": 32,
    }:
        raise ValueError("Requested runtime shape does not match artifact manifest")
    for name, expected in manifest["source_files_sha256"].items():
        if file_digest(source / name) != expected:
            raise ValueError(f"Source mismatch for ANE artifact: {name}")
    if tree_digest(package) != manifest["package_sha256"]:
        raise ValueError("ANE package content does not match its manifest")
    return manifest
