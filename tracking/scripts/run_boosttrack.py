"""Phase 7/8: run BoostTrack over stored detections and embeddings.

This drives ``BoostTrack`` directly instead of going through upstream
``main.py``, which couples the YOLOX detector, its letterbox dataloader, and
the tracker in one loop. Driving it here keeps the tracker-side change surface
to a single ReID loading fix and makes detector, ReID, and tracker independent
inputs.

Two contracts are handled explicitly:

- ``BoostTrack.update`` divides incoming boxes by the YOLOX letterbox scale
  factor, derived from the ratio between the model input tensor and the raw
  image. Detections here are already in original pixels, so a tensor with the
  image's own shape is passed and the factor resolves to exactly 1.0.
- Detections are supplied below the tracker's ``det_thresh`` on purpose.
  BoostTrack boosts low-confidence detections that match existing tracks before
  applying its cut, so pre-filtering would silently disable that mechanism.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOSTTRACK_ROOT = REPO_ROOT / "repos" / "BoostTrack"

sys.path.insert(0, str(REPO_ROOT / "tracking" / "src"))

from mot20_tracking.artifacts import (  # noqa: E402
    TRACKS_FORMAT,
    TrackingArtifacts,
    sha256_file,
)
from mot20_tracking.detections import read_detections  # noqa: E402
from mot20_tracking.sequences import read_split  # noqa: E402
from mot20_tracking.variants import Combination  # noqa: E402


class PrecomputedEmbedder:
    """Serve stored L2 embeddings through ``EmbeddingComputer``'s interface.

    BoostTrack asks for embeddings after boosting and thresholding, so the boxes
    it presents are a subset of the exported detections. Boosting changes scores
    only, and the letterbox rescale resolves to 1.0 here, so the box coordinates
    are unchanged and can be matched back to their stored row.
    """

    def __init__(self, rows_by_frame: dict[int, np.ndarray], matrix: np.ndarray) -> None:
        self._lookup: dict[int, dict[tuple[float, ...], list[int]]] = {}
        self._matrix = matrix
        offset = 0
        for frame_id in sorted(rows_by_frame):
            rows = rows_by_frame[frame_id]
            index: dict[tuple[float, ...], list[int]] = {}
            for position in range(rows.shape[0]):
                index.setdefault(self._key(rows[position, :4]), []).append(offset + position)
            self._lookup[frame_id] = index
            offset += rows.shape[0]
        if offset != matrix.shape[0]:
            raise ValueError(f"{offset} detection rows but {matrix.shape[0]} embedding rows")

    @staticmethod
    def _key(box: np.ndarray) -> tuple[float, ...]:
        return tuple(round(float(value), 2) for value in box[:4])

    def compute_embedding(self, img, bbox, tag):  # noqa: ANN001, ARG002
        frame_id = int(str(tag).split(":")[1])
        index = self._lookup.get(frame_id)
        if index is None:
            raise KeyError(f"no stored detections for {tag}")
        consumed: dict[tuple[float, ...], int] = {}
        selected = np.empty((bbox.shape[0], self._matrix.shape[1]), dtype=np.float32)
        for position in range(bbox.shape[0]):
            key = self._key(np.asarray(bbox[position]))
            candidates = index.get(key)
            if not candidates:
                raise KeyError(f"{tag}: box {key} has no stored embedding")
            taken = consumed.get(key, 0)
            if taken >= len(candidates):
                raise KeyError(f"{tag}: more requests than stored rows for box {key}")
            consumed[key] = taken + 1
            selected[position] = self._matrix[candidates[taken]]
        return selected

    def dump_cache(self) -> None:
        """No-op: stored embeddings are the cache, and they are read-only."""


def parse_override_value(name: str, raw_value: str, default: Any) -> bool | float | int | str:
    """Parse an override according to the upstream setting's current type.

    The upstream settings are plain dictionaries rather than a schema. Using
    their existing values as the schema keeps this runner aligned when the
    vendored tracker changes and rejects accidental string-valued sweeps.
    """
    if isinstance(default, bool):
        normalized = raw_value.lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        raise ValueError(f"{name} must be true or false, not {raw_value!r}")
    if isinstance(default, int):
        try:
            return int(raw_value)
        except ValueError as error:
            raise ValueError(f"{name} must be an integer, not {raw_value!r}") from error
    if isinstance(default, float):
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(f"{name} must be a number, not {raw_value!r}") from error
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, not {raw_value!r}")
        return value
    if isinstance(default, str):
        if not raw_value:
            raise ValueError(f"{name} must not be empty")
        return raw_value
    raise TypeError(f"{name} has unsupported setting type {type(default).__name__}")


def apply_overrides(
    raw_overrides: list[str],
    general_settings: Any,
    boost_track_settings: Any,
    boost_track_plus_plus_settings: Any,
) -> list[dict[str, bool | float | int | str]]:
    """Apply ``NAME=VALUE`` settings, including MOT20's shadowing dictionaries.

    GeneralSettings and BoostTrackSettings resolve a MOT20-specific value before
    their general value. Both destinations therefore receive each supported
    override. ``max_age`` is additionally wired into upstream's frame-rate
    helper, which otherwise ignores its dictionary entry altogether.
    """
    settings_classes = (
        ("general", general_settings),
        ("boost_track", boost_track_settings),
        ("boost_track_plus_plus", boost_track_plus_plus_settings),
    )
    applied: list[dict[str, bool | float | int | str]] = []
    seen_names: set[str] = set()

    for raw_override in raw_overrides:
        if raw_override.count("=") != 1:
            raise ValueError(f"override must be NAME=VALUE, not {raw_override!r}")
        name, raw_value = raw_override.split("=", 1)
        if not name or not raw_value:
            raise ValueError(f"override must be NAME=VALUE, not {raw_override!r}")
        if name in seen_names:
            raise ValueError(f"override specified more than once: {name}")
        seen_names.add(name)

        matches = [(kind, settings) for kind, settings in settings_classes if name in settings.values]
        if not matches:
            available = sorted(name for _, settings in settings_classes for name in settings.values)
            raise ValueError(f"unknown override {name!r}; expected one of: {', '.join(available)}")
        if len(matches) != 1:
            raise ValueError(f"ambiguous override {name!r}; it appears in multiple settings classes")

        kind, settings = matches[0]
        if kind == "general" and name in {"dataset", "test_dataset", "use_embedding", "use_ecc"}:
            raise ValueError(f"{name} is controlled by this runner and cannot be overridden with --set")
        value = parse_override_value(name, raw_value, settings.values[name])
        settings.values[name] = value
        if hasattr(settings, "dataset_specific_settings"):
            settings.dataset_specific_settings.setdefault("mot20", {})[name] = value
        applied.append({"name": name, "value": value, "settings": kind})

    max_age = next((override["value"] for override in applied if override["name"] == "max_age"), None)
    if max_age is not None:
        general_settings.max_age = staticmethod(lambda sequence_name: int(max_age))
    return applied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--reid", required=True, help="reid variant slug, or 'none'")
    parser.add_argument("--tracker", required=True, help="tracker configuration slug")
    parser.add_argument("--split", default="val_half")
    parser.add_argument("--no-cmc", action="store_true", help="disable ECC camera-motion compensation")
    parser.add_argument(
        "--det-thresh",
        type=float,
        default=None,
        help="override the tracker's post-boost detection threshold, which upstream tunes per dataset for YOLOX",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="repeatable upstream setting override; unknown names and invalid values fail",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate settings and print the resolved plan without reading inputs or tracking",
    )
    parser.add_argument("--artifact-root", default="artifacts/tracking")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combination = Combination(args.detector, args.reid, args.tracker)
    os.chdir(BOOSTTRACK_ROOT)
    sys.path.insert(0, str(BOOSTTRACK_ROOT))
    from default_settings import BoostTrackPlusPlusSettings, BoostTrackSettings, GeneralSettings

    GeneralSettings.values["dataset"] = "mot20"
    GeneralSettings.values["test_dataset"] = args.split == "test"
    GeneralSettings.values["use_embedding"] = combination.uses_reid
    GeneralSettings.values["use_ecc"] = not args.no_cmc
    if args.det_thresh is not None:
        # dataset_specific_settings overrides det_thresh per dataset, so the
        # override must go there or GeneralSettings.__class_getitem__ ignores it.
        GeneralSettings.values["det_thresh"] = args.det_thresh
        GeneralSettings.dataset_specific_settings["mot20"]["det_thresh"] = args.det_thresh
    if args.det_thresh is not None and any(
        override.startswith("det_thresh=") for override in args.overrides
    ):
        raise SystemExit("--det-thresh and --set det_thresh=... cannot be used together")
    try:
        applied_overrides = apply_overrides(
            args.overrides,
            GeneralSettings,
            BoostTrackSettings,
            BoostTrackPlusPlusSettings,
        )
    except (TypeError, ValueError) as error:
        raise SystemExit(f"invalid --set override: {error}") from error
    if args.det_thresh is not None:
        applied_overrides.insert(
            0,
            {"name": "det_thresh", "value": args.det_thresh, "settings": "general"},
        )

    settings_snapshot = {
        "general": dict(GeneralSettings.values),
        "boost_track": dict(BoostTrackSettings.values),
        "boost_track_plus_plus": dict(BoostTrackPlusPlusSettings.values),
        "applied_overrides": applied_overrides,
        "det_thresh_effective": GeneralSettings["det_thresh"],
        "dlo_boost_coef_effective": BoostTrackSettings["dlo_boost_coef"],
    }
    print(
        f"det_thresh {settings_snapshot['det_thresh_effective']} | "
        f"dlo_boost_coef {settings_snapshot['dlo_boost_coef_effective']} | "
        f"reid {args.reid} | ecc {GeneralSettings['use_ecc']}"
    )
    if args.dry_run:
        print(f"dry run: would track {combination} on {args.split}")
        print(f"applied overrides: {applied_overrides}")
        return

    artifacts = TrackingArtifacts(REPO_ROOT / args.artifact_root)
    detections_dir = artifacts.detections_dir(args.detector, args.split)
    if not artifacts.manifest_path(detections_dir).exists():
        raise SystemExit(f"detection variant not exported: {detections_dir}")
    if combination.uses_reid:
        embeddings_dir = artifacts.embeddings_dir(args.detector, args.reid, args.split)
        if not artifacts.manifest_path(embeddings_dir).exists():
            raise SystemExit(f"embedding variant not exported: {embeddings_dir}")
    level_dir = artifacts.tracks_dir(combination, args.split)
    if artifacts.manifest_path(level_dir).exists():
        raise SystemExit(f"tracks already produced: {level_dir}")

    sequences = read_split(args.split, repo_root=REPO_ROOT)

    import external  # noqa: F401
    import cv2
    import torch
    import utils
    from tracker.boost_track import BoostTrack

    entries = []
    started = time.perf_counter()
    tracked_frames = 0
    for sequence in sequences:
        detection_file = artifacts.detections_file(args.detector, args.split, sequence.name)
        detections = read_detections(detection_file, sequence.name, sequence.length)

        tracker = BoostTrack(video_name=sequence.name)
        if combination.uses_reid:
            payload = np.load(
                artifacts.embeddings_file(args.detector, args.reid, args.split, sequence.name)
            )
            tracker.embedder = PrecomputedEmbedder(detections.frames, payload["embeddings"])

        results = []
        for frame_id in range(1, sequence.length + 1):
            image = cv2.imread(str(sequence.frame_path(frame_id)))
            if image is None:
                raise SystemExit(f"unreadable frame: {sequence.frame_path(frame_id)}")
            rows = detections.rows_for(frame_id)
            # Shape-only tensor: update() derives the letterbox scale factor from
            # this against the raw image, and matching shapes make it exactly 1.0.
            shape_tensor = torch.zeros((1, 3, image.shape[0], image.shape[1]))
            targets = tracker.update(rows, shape_tensor, image, f"{sequence.name}:{frame_id}")
            tlwhs, ids, confs = utils.filter_targets(
                targets,
                GeneralSettings["aspect_ratio_thresh"],
                GeneralSettings["min_box_area"],
            )
            results.append((frame_id, tlwhs, ids, confs))
            tracked_frames += 1
        tracker.dump_cache()

        destination = artifacts.tracks_file(combination, args.split, sequence.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise SystemExit(f"refusing to overwrite tracks: {destination}")
        utils.write_results_no_score(str(destination), results)

        identities = {int(i) for _, _, ids, _ in results for i in ids}
        boxes = sum(len(tlwhs) for _, tlwhs, _, _ in results)
        entries.append(
            {
                "sequence": sequence.name,
                "frames": sequence.length,
                "track_ids": len(identities),
                "boxes": boxes,
                "detections_sha256": sha256_file(detection_file),
                "written_sha256": sha256_file(destination),
            }
        )
        print(f"{sequence.name}: {len(identities):>5} ids, {boxes:>7} boxes")

    elapsed = time.perf_counter() - started
    digest = artifacts.write_manifest(
        level_dir,
        {
            "format": TRACKS_FORMAT,
            "combination": str(combination),
            "detector_variant": args.detector,
            "reid_variant": args.reid,
            "tracker_variant": args.tracker,
            "split": args.split,
            "embeddings": "precomputed L2" if combination.uses_reid else "disabled",
            "letterbox_scale": 1.0,
            "settings": settings_snapshot,
            "wall_seconds": round(elapsed, 2),
            "fps": round(tracked_frames / max(elapsed, 1e-9), 3),
            "sequences": entries,
            "totals": {
                "sequences": len(entries),
                "frames": tracked_frames,
                "boxes": sum(int(entry["boxes"]) for entry in entries),
            },
        },
    )
    print(f"\n{tracked_frames} frames in {elapsed:.1f}s")
    print(f"manifest {artifacts.manifest_path(level_dir)} sha256 {digest}")


if __name__ == "__main__":
    main()
