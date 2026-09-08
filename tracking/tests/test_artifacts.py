"""Artifact-store invariants that protect the combination grid."""

from __future__ import annotations

import pytest

from mot20_tracking.artifacts import (
    TrackingArtifacts,
    embedding_cache_key,
    validate_split,
    write_manifest,
)
from mot20_tracking.variants import Combination, validate_slug


def test_embedding_cache_key_separates_reid_models() -> None:
    """The upstream failure mode: same detections, different ReID model."""
    first = embedding_cache_key("rfdetr2xl-e5-t005", "osnet-ain-msdc", "MOT20-01")
    second = embedding_cache_key("rfdetr2xl-e5-t005", "mot20-sbs-s50", "MOT20-01")
    assert first != second


def test_embedding_cache_key_separates_detectors() -> None:
    first = embedding_cache_key("rfdetr2xl-e5-t005", "osnet-ain-msdc", "MOT20-01")
    second = embedding_cache_key("yoloxx20", "osnet-ain-msdc", "MOT20-01")
    assert first != second


def test_embedding_cache_key_is_stable() -> None:
    repeated = [embedding_cache_key("yoloxx20", "osnet-ain-msdc", "MOT20-01") for _ in range(3)]
    assert len(set(repeated)) == 1


@pytest.mark.parametrize(
    "value",
    ["Rfdetr", "rfdetr_e5", "rfdetr--e5", "-rfdetr", "rfdetr-", "", "a b", "a/b", "a__b"],
)
def test_invalid_slugs_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        validate_slug(value, "detector")


@pytest.mark.parametrize("value", ["rfdetr2xl-e5-t005", "yoloxx20", "none", "btpp-default"])
def test_valid_slugs_accepted(value: str) -> None:
    assert validate_slug(value, "detector") == value


def test_combination_roundtrip() -> None:
    combination = Combination("rfdetr2xl-e5-t005", "osnet-ain-msdc", "btpp-default")
    assert Combination.parse(str(combination)) == combination


def test_combination_reports_reid_free_runs() -> None:
    assert not Combination("yoloxx20", "none", "bt-noreid").uses_reid
    assert Combination("yoloxx20", "osnet-ain-msdc", "btpp-default").uses_reid


def test_unknown_split_rejected() -> None:
    with pytest.raises(ValueError):
        validate_split("valid")


def test_manifest_is_never_overwritten(tmp_path) -> None:
    destination = tmp_path / "manifest.json"
    write_manifest(destination, {"format": "mot20.tracking.detections-manifest.v1"})
    with pytest.raises(FileExistsError):
        write_manifest(destination, {"format": "mot20.tracking.detections-manifest.v1"})


def test_manifest_requires_format(tmp_path) -> None:
    with pytest.raises(ValueError):
        write_manifest(tmp_path / "manifest.json", {"detector": "yoloxx20"})


def test_paths_compose_by_level(tmp_path) -> None:
    artifacts = TrackingArtifacts(tmp_path)
    combination = Combination("rfdetr2xl-e5-t005", "osnet-ain-msdc", "btpp-default")
    detections = artifacts.detections_file("rfdetr2xl-e5-t005", "val_half", "MOT20-01")
    embeddings = artifacts.embeddings_file(
        "rfdetr2xl-e5-t005", "osnet-ain-msdc", "val_half", "MOT20-01"
    )
    tracks = artifacts.tracks_file(combination, "val_half", "MOT20-01")
    assert detections.relative_to(tmp_path).as_posix() == (
        "detections/rfdetr2xl-e5-t005/val_half/MOT20-01/det.txt"
    )
    assert embeddings.relative_to(tmp_path).as_posix() == (
        "embeddings/rfdetr2xl-e5-t005__osnet-ain-msdc/val_half/MOT20-01.npz"
    )
    assert tracks.relative_to(tmp_path).as_posix() == (
        "tracks/rfdetr2xl-e5-t005__osnet-ain-msdc__btpp-default/val_half/MOT20-01.txt"
    )
