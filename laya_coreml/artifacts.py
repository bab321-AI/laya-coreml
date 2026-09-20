"""Integrity checks for portable Core ML bundles."""

import errno
import hashlib
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath


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


def package_for_coreml(package):
    """Materialize Hub symlinks: the native compiler can copy them into broken paths."""
    package = Path(package)
    if not package.is_symlink() and not any(path.is_symlink() for path in package.rglob("*")):
        return package
    digest = tree_digest(package)
    root = (
        Path(os.environ.get("LAYA_COREML_CACHE", Path.home() / ".cache/laya-coreml")) / "packages"
    )
    root.mkdir(parents=True, exist_ok=True)
    destination = root / digest
    target = destination / "model.mlpackage"
    if destination.exists():
        if not target.is_dir() or tree_digest(target) != digest:
            raise ValueError(f"Core ML cache integrity failure; remove {destination} and retry")
        return target
    temporary = Path(tempfile.mkdtemp(prefix=".preparing-", dir=root))
    try:
        copied = temporary / "model.mlpackage"
        shutil.copytree(package, copied, symlinks=False)
        if tree_digest(copied) != digest:
            raise ValueError("Core ML package changed while materializing cached weights")
        try:
            temporary.rename(destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not target.is_dir():
                raise
        if tree_digest(target) != digest:
            raise ValueError("Materialized Core ML package failed its integrity check")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return target


def verify_files(directory, files):
    for name, expected in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Bundle file names must be relative paths inside the model directory")
        if file_digest(Path(directory) / name) != expected["sha256"]:
            raise ValueError(f"Bundle file does not match its manifest: {name}")


def verify_research_manifest(source, package, *, length):
    import json

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
    verify_files(
        source, {name: {"sha256": sha} for name, sha in manifest["source_files_sha256"].items()}
    )
    if tree_digest(package) != manifest["package_sha256"]:
        raise ValueError("ANE package content does not match its manifest")
    return manifest
