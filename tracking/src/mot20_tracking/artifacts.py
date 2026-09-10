"""Layered artifact store for tracking experiments.

Levels, and what each is keyed by:

- L1 detections: detector variant, split
- L2 embeddings: detector variant, ReID variant, split
- L3 tracks: detector, ReID, tracker configuration, split
- L4 metrics: same triple as L3

Every level writes a ``manifest.json`` recording its inputs, so a cell of the
grid is reproducible from its path plus its manifests. Manifests are never
overwritten; producing a level again requires an explicit new variant slug or
an explicit removal, matching the repository's rule that generated evidence is
not silently replaced.

The camera-motion (ECC) level is deliberately absent from this module: it is
derived from images alone, is keyed by sequence only, and is therefore shared
across every combination.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mot20_tracking.variants import Combination, detection_reid_id, validate_slug

DETECTIONS_FORMAT = "mot20.tracking.detections-manifest.v1"
EMBEDDINGS_FORMAT = "mot20.tracking.embeddings-manifest.v1"
TRACKS_FORMAT = "mot20.tracking.tracks-manifest.v1"
METRICS_FORMAT = "mot20.tracking.metrics-manifest.v1"

MANIFEST_NAME = "manifest.json"

_KNOWN_SPLITS = ("val_half", "train", "test")


def validate_split(split: str) -> str:
    """Return *split* if it names a MOT20 split this workstream uses."""
    if split not in _KNOWN_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {_KNOWN_SPLITS}")
    return split


class TrackingArtifacts:
    """Path resolver for the tracking artifact tree.

    Args:
        root: Directory holding the levels, normally repo-relative
            ``artifacts/tracking``.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    # -- L1 detections ----------------------------------------------------

    def detections_dir(self, detector: str, split: str) -> Path:
        validate_slug(detector, "detector")
        validate_split(split)
        return self.root / "detections" / detector / split

    def detections_file(self, detector: str, split: str, sequence: str) -> Path:
        return self.detections_dir(detector, split) / sequence / "det.txt"

    # -- L2 embeddings ----------------------------------------------------

    def embeddings_dir(self, detector: str, reid: str, split: str) -> Path:
        validate_split(split)
        return self.root / "embeddings" / detection_reid_id(detector, reid) / split

    def embeddings_file(self, detector: str, reid: str, split: str, sequence: str) -> Path:
        return self.embeddings_dir(detector, reid, split) / f"{sequence}.npz"

    # -- L3 tracks --------------------------------------------------------

    def tracks_dir(self, combination: Combination, split: str) -> Path:
        validate_split(split)
        return self.root / "tracks" / str(combination) / split

    def tracks_file(self, combination: Combination, split: str, sequence: str) -> Path:
        return self.tracks_dir(combination, split) / f"{sequence}.txt"

    # -- L4 metrics -------------------------------------------------------

    def metrics_dir(self, combination: Combination, split: str) -> Path:
        validate_split(split)
        return self.root / "metrics" / str(combination) / split

    # -- manifests --------------------------------------------------------

    def manifest_path(self, level_dir: Path) -> Path:
        return level_dir / MANIFEST_NAME

    def write_manifest(self, level_dir: Path, manifest: dict[str, Any]) -> str:
        """Write a level manifest once and return its SHA-256 digest."""
        return write_manifest(self.manifest_path(level_dir), manifest)

    def read_manifest(self, level_dir: Path) -> dict[str, Any]:
        path = self.manifest_path(level_dir)
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict) or "format" not in payload:
            raise ValueError(f"malformed manifest: {path}")
        return payload


def write_manifest(destination: Path, manifest: dict[str, Any]) -> str:
    """Write a manifest once and return its SHA-256 digest.

    Raises:
        FileExistsError: If the manifest already exists. Regenerating a level
            requires an explicit decision, not an accidental overwrite.
    """
    destination = Path(destination)
    if "format" not in manifest:
        raise ValueError("manifest requires a 'format' key")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {destination}")
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file, read incrementally."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embedding_cache_key(detector: str, reid: str, sequence: str) -> str:
    """Cache key for one sequence's embeddings.

    Upstream BoostTrack keys its embedding cache by sequence name alone, so
    changing the ReID model over unchanged detections returns the previous
    model's vectors with no error: the count guard cannot see the difference.
    Including both variant slugs is what makes a combination study trustworthy.
    """
    return f"{detection_reid_id(detector, reid)}__{sequence}"
