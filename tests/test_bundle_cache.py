from pathlib import Path

import pytest

from laya_coreml.artifacts import file_digest, package_for_coreml, tree_digest, verify_files


def test_hub_weight_symlinks_materialize_as_integral_reusable_files(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYA_COREML_CACHE", str(tmp_path / "cache"))
    blob = tmp_path / "blob"
    blob.write_bytes(b"test model weights")
    package = tmp_path / "snapshot" / "model.mlpackage"
    weights = package / "Data/weights.bin"
    weights.parent.mkdir(parents=True)
    weights.symlink_to(blob)
    (package / "Manifest.json").write_text("{}")
    copied = package_for_coreml(package)
    assert copied != package and tree_digest(copied) == tree_digest(package)
    assert not (copied / "Data/weights.bin").is_symlink()
    assert (copied / "Data/weights.bin").read_bytes() == blob.read_bytes()
    assert package_for_coreml(package) == copied
    (copied / "Data/weights.bin").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="integrity failure"):
        package_for_coreml(package)


def test_regular_local_packages_are_not_duplicated(tmp_path):
    (tmp_path / "Manifest.json").write_text("{}")
    assert package_for_coreml(tmp_path) == tmp_path


def test_bundle_integrity_rejects_mutation_and_parent_paths(tmp_path):
    asset = tmp_path / "weights.bin"
    asset.write_bytes(b"original")
    expected = {"weights.bin": {"sha256": file_digest(asset)}}
    verify_files(tmp_path, expected)
    asset.write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        verify_files(tmp_path, expected)
    for name in ("../secret", "/absolute", "..\\secret"):
        with pytest.raises(ValueError, match="relative paths"):
            verify_files(Path(tmp_path), {name: {"sha256": "unused"}})
