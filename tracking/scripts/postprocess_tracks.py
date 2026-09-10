"""Apply BoostTrack++'s postprocessing passes to an L3 tracks artifact.

Upstream runs two passes after tracking and scores the *second* one: every
evaluation command in the BoostTrack++ README passes ``--TRACKERS_TO_EVAL
<name>_post_gbi``. Our runner writes raw tracks and stops, so every tracking
number this repository has published so far is a raw-protocol number and is not
directly comparable with upstream's.

The two passes are exposed as separate rungs, because they do different things
and only one of them is obviously safe:

``dti``       Linear interpolation. For a track with more than ``n_min`` rows,
              every interior gap shorter than ``n_dti`` is filled by sliding
              linearly between the observed rows either side. It adds boxes and
              never moves an observed one.
``dti-gbi``   The above, then a gradient-boosting smoother fitted per track and
              per coordinate. It REPLACES every box, including observed ones.

Both are pure post-hoc transforms of a text file. No detector, ReID model or
tracker is re-run, so a rung costs no GPU and cannot change track identities.

CONFIDENCE COLUMN. Upstream destroys it: ``dti`` writes ``-1`` on every row and
the smoother writes ``1`` on every row. We keep the tracker's own confidence on
observed rows and write ``-1`` only on rows we invented, so a downstream reader
(track-viz, crop selection) can tell an observation from an interpolation. This
cannot change a metric: TrackEval parses the column into ``tracker_confidences``
and none of HOTA, CLEAR or Identity ever reads it. ``--upstream-confidence``
restores the upstream behaviour for byte-level comparison.

VERIFICATION. ``--verify-against-upstream`` runs the vendored ``utils.dti`` over
the same input in a temporary directory and asserts our ``dti`` rung agrees on
frame, identity and all four coordinates. That check is what licenses the
reimplementation; without it this file would be a paraphrase of upstream rather
than a faithful port.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOSTTRACK_ROOT = REPO_ROOT / "repos" / "BoostTrack"
sys.path.insert(0, str(REPO_ROOT / "tracking" / "src"))

from mot20_tracking.artifacts import (  # noqa: E402
    TRACKS_FORMAT,
    TrackingArtifacts,
    sha256_file,
)
from mot20_tracking.sequences import read_split  # noqa: E402
from mot20_tracking.variants import Combination, validate_slug  # noqa: E402

# Upstream's effective settings, from repos/BoostTrack/main.py:130-136.
UPSTREAM_N_MIN = 25
UPSTREAM_N_DTI = 1000

# repos/BoostTrack/tracker/GBI.py:41 sets only the first three. The rest are
# scikit-learn defaults, which materially define the smoother and drift between
# versions, so they are pinned here and recorded in the manifest.
GBI_PARAMS = {
    "n_estimators": 115,
    "learning_rate": 0.065,
    "min_samples_split": 6,
    "loss": "squared_error",
    "max_depth": 3,
    "subsample": 1.0,
    "min_samples_leaf": 1,
    "max_features": None,
    "random_state": 0,
}

INTERPOLATED_CONFIDENCE = -1.0
"""Sentinel written on rows this script invented. No TrackEval metric reads it."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combination", required=True, help="source det__reid__tracker")
    parser.add_argument("--rung", required=True, choices=("dti", "dti-gbi"))
    parser.add_argument("--split", default="val_half")
    parser.add_argument("--n-min", type=int, default=UPSTREAM_N_MIN)
    parser.add_argument("--n-dti", type=int, default=UPSTREAM_N_DTI)
    parser.add_argument(
        "--upstream-confidence",
        action="store_true",
        help="overwrite every confidence as upstream does, instead of preserving observations",
    )
    parser.add_argument(
        "--verify-against-upstream",
        action="store_true",
        help="cross-check the dti rung against the vendored utils.dti",
    )
    parser.add_argument("--artifact-root", default="artifacts/tracking")
    return parser.parse_args()


def read_tracks(path: Path) -> np.ndarray:
    """Read one MOT result file as an (N, 10) float array.

    Returns a (0, 10) array for an empty file rather than raising, so a
    sequence in which the tracker emitted nothing stays a valid, empty rung.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return np.zeros((0, 10), dtype=np.float64)
    rows = [[float(value) for value in line.split(",")] for line in text.splitlines() if line]
    widths = {len(row) for row in rows}
    if widths != {10}:
        raise SystemExit(f"{path}: expected 10 columns per row, saw {sorted(widths)}")
    return np.asarray(rows, dtype=np.float64)


def write_tracks(destination: Path, rows: np.ndarray, rung: str) -> None:
    """Write rows frame-ascending then identity-ascending, at upstream's precision.

    Coordinate precision is rung-specific and matches upstream exactly, because
    rounding is not neutral: it perturbs every interpolated box by up to half a
    unit in the last written place, and IoU is scored at thresholds fine enough
    to notice. ``dti_write_results`` formats with ``str`` (full repr);
    ``GBInterpolation`` uses ``%.2f``. Writing our raw-track 1-decimal format
    here would silently make the rungs a different protocol from upstream's.
    """
    if destination.exists():
        raise SystemExit(f"refusing to overwrite tracks: {destination}")
    order = np.lexsort((rows[:, 1], rows[:, 0])) if rows.size else np.zeros((0,), dtype=int)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if rung == "dti-gbi":
        def body(r: np.ndarray) -> str:
            return f"{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},{r[6]:.2f}"
    else:
        def body(r: np.ndarray) -> str:
            return f"{r[2]},{r[3]},{r[4]},{r[5]},{r[6]}"
    lines = [f"{int(r[0])},{int(r[1])},{body(r)},-1,-1,-1" for r in rows[order]]
    with destination.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
        if lines:
            stream.write("\n")


def linear_interpolate(rows: np.ndarray, n_min: int, n_dti: int) -> tuple[np.ndarray, int]:
    """Fill interior gaps of eligible tracks. Returns (rows, interpolated_count).

    Faithful to ``repos/BoostTrack/utils.py:49-115``: a track qualifies on
    ``n_frame > n_min``, and a gap qualifies on ``1 < delta < n_dti``, both
    strict. Interpolation is linear in tlwh, which is what upstream does; it is
    not the same as interpolating the corner coordinates.
    """
    if rows.size == 0:
        return rows, 0
    keys = rows[:, :2].astype(np.int64)
    if np.unique(keys, axis=0).shape[0] != keys.shape[0]:
        raise SystemExit(
            "source tracks contain duplicate (frame, identity) rows; interpolation and the "
            "upstream comparison both assume one box per identity per frame"
        )
    produced: list[np.ndarray] = []
    for identity in np.unique(rows[:, 1]):
        track = rows[rows[:, 1] == identity]
        track = track[np.argsort(track[:, 0], kind="stable")]
        produced.append(track)
        if track.shape[0] <= n_min:
            continue
        frames = track[:, 0]
        for index in range(1, track.shape[0]):
            left_frame, right_frame = frames[index - 1], frames[index]
            delta = right_frame - left_frame
            if not (1 < delta < n_dti):
                continue
            left_box, right_box = track[index - 1, 2:6], track[index, 2:6]
            for step in range(1, int(delta)):
                current_frame = step + left_frame
                filled = np.zeros((1, 10), dtype=np.float64)
                filled[0, 0] = current_frame
                filled[0, 1] = identity
                # Upstream's exact operation order (utils.py:98-100). Reassociating
                # this as left + diff * (step / delta) changes the last ulp.
                filled[0, 2:6] = (current_frame - left_frame) * (right_box - left_box) / (
                    right_frame - left_frame
                ) + left_box
                filled[0, 6] = INTERPOLATED_CONFIDENCE
                filled[0, 7:] = -1
                produced.append(filled)
    combined = np.vstack(produced)
    return combined, int(combined.shape[0] - rows.shape[0])


def gradient_boosting_smooth(rows: np.ndarray) -> np.ndarray:
    """Refit every track's x, y, w, h against frame index and replace the boxes.

    Mirrors ``repos/BoostTrack/tracker/GBI.py:30-59``. Every row is rewritten,
    including rows the tracker actually observed, which is why this rung is
    scored separately from plain interpolation.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    if rows.size == 0:
        return rows
    # Upstream sorts by identity then frame before fitting (GBI.py:64-65). Our
    # interpolated rows are appended after each track's observations, so without
    # this the regressor would see the same points in a different order, and a
    # boosting fit's reductions are order-sensitive at the last ulp.
    ordered = rows[np.lexsort((rows[:, 0], rows[:, 1]))]
    smoothed = ordered.copy()
    for identity in np.unique(ordered[:, 1]):
        selector = ordered[:, 1] == identity
        track = ordered[selector]
        frame = track[:, 0].reshape(-1, 1)
        for column in (2, 3, 4, 5):
            regressor = GradientBoostingRegressor(**GBI_PARAMS)
            regressor.fit(frame, track[:, column].ravel())
            smoothed[selector, column] = regressor.predict(frame)
    return smoothed


def verify_against_upstream(source: Path, written: Path, n_min: int, n_dti: int) -> str:
    """Run the vendored ``utils.dti`` on *source* and compare with our WRITTEN file.

    Comparing the file rather than the in-memory array is the point: a
    serialization difference is exactly the kind of divergence that would
    otherwise pass verification and then be the thing actually evaluated.

    Compares frame, identity and all four coordinates. The confidence column is
    excluded on purpose: upstream overwrites it and we deliberately do not.
    """
    sys.path.insert(0, str(BOOSTTRACK_ROOT))
    import utils  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = Path(scratch)
        staged_in = scratch_path / "in"
        staged_out = scratch_path / "out"
        staged_in.mkdir()
        staged_out.mkdir()
        shutil.copy(source, staged_in / source.name)
        utils.dti(str(staged_in), str(staged_out), n_min=n_min, n_dti=n_dti)
        theirs = read_tracks(staged_out / source.name)
    ours = read_tracks(written)

    def key(array: np.ndarray) -> np.ndarray:
        return array[np.lexsort((array[:, 1], array[:, 0]))]

    if theirs.shape[0] != ours.shape[0]:
        raise SystemExit(
            f"{source.name}: upstream dti produced {theirs.shape[0]} rows, ours {ours.shape[0]}"
        )
    mine, other = key(ours), key(theirs)
    if not np.array_equal(mine[:, :2].astype(np.int64), other[:, :2].astype(np.int64)):
        raise SystemExit(f"{source.name}: frame/identity disagreement with upstream dti")
    delta = float(np.abs(mine[:, 2:6] - other[:, 2:6]).max()) if mine.size else 0.0
    if delta != 0.0:
        raise SystemExit(
            f"{source.name}: box disagreement with upstream dti, max {delta:.3e}. "
            f"Exact equality is required: both sides interpolate the same 1-decimal "
            f"endpoints with the same operation order, so any difference is a real defect."
        )
    return f"{theirs.shape[0]} rows, exact coordinate match"


def main() -> None:
    args = parse_args()
    source_combination = Combination.parse(args.combination)
    target_tracker = f"{source_combination.tracker}-{args.rung}"
    validate_slug(target_tracker, "derived tracker")
    target_combination = Combination(
        source_combination.detector, source_combination.reid, target_tracker
    )
    if args.n_min < 0 or args.n_dti < 2:
        raise SystemExit("--n-min must be >= 0 and --n-dti must be >= 2")

    artifacts = TrackingArtifacts(REPO_ROOT / args.artifact_root)
    source_dir = artifacts.tracks_dir(source_combination, args.split)
    if not artifacts.manifest_path(source_dir).exists():
        raise SystemExit(f"source tracks not produced: {source_dir}")
    target_dir = artifacts.tracks_dir(target_combination, args.split)
    if artifacts.manifest_path(target_dir).exists():
        raise SystemExit(f"rung already produced: {target_dir}")

    sequences = read_split(args.split, repo_root=REPO_ROOT)
    # The store commits a level by writing its manifest last, so a run that died
    # mid-sequence leaves files with no manifest. Without this the retry would
    # fail at the first existing file with no explanation of why.
    partial = [
        artifacts.tracks_file(target_combination, args.split, s.name)
        for s in sequences
        if artifacts.tracks_file(target_combination, args.split, s.name).exists()
    ]
    if partial:
        listing = "\n  ".join(str(p) for p in partial)
        raise SystemExit(
            f"{target_dir} has {len(partial)} track file(s) but no manifest, so a previous "
            f"run failed part-way. Inspect and remove them, then re-run:\n  {listing}"
        )

    started = time.perf_counter()
    entries: list[dict[str, object]] = []
    for sequence in sequences:
        source_file = artifacts.tracks_file(source_combination, args.split, sequence.name)
        if not source_file.exists():
            raise SystemExit(f"source sequence missing: {source_file}")
        raw = read_tracks(source_file)
        if raw.size and (raw[:, 0].min() < 1 or raw[:, 0].max() > sequence.length):
            raise SystemExit(
                f"{sequence.name}: source frames outside 1..{sequence.length}; "
                "the tracks and the split do not match"
            )

        interpolated, added = linear_interpolate(raw, args.n_min, args.n_dti)

        final = interpolated
        if args.rung == "dti-gbi":
            final = gradient_boosting_smooth(interpolated)
        if args.upstream_confidence:
            final = final.copy()
            final[:, 6] = 1.0 if args.rung == "dti-gbi" else -1.0

        destination = artifacts.tracks_file(target_combination, args.split, sequence.name)
        write_tracks(destination, final, args.rung)

        # Verify what was actually written, after serialization, not the array.
        verification = None
        if args.verify_against_upstream:
            if args.rung != "dti":
                raise SystemExit("--verify-against-upstream only applies to the dti rung")
            verification = verify_against_upstream(
                source_file, destination, args.n_min, args.n_dti
            )
        entries.append(
            {
                "sequence": sequence.name,
                "frames": sequence.length,
                "source_boxes": int(raw.shape[0]),
                "boxes": int(final.shape[0]),
                "interpolated_boxes": added,
                "track_ids": int(np.unique(final[:, 1]).size) if final.size else 0,
                "source_sha256": sha256_file(source_file),
                "written_sha256": sha256_file(destination),
                **({"upstream_dti_check": verification} if verification else {}),
            }
        )
        print(
            f"{sequence.name}: {int(raw.shape[0]):>7} -> {int(final.shape[0]):>7} boxes "
            f"(+{added} interpolated, {100 * added / max(raw.shape[0], 1):.2f}%)"
            + (f" | upstream dti agrees: {verification}" if verification else "")
        )

    elapsed = time.perf_counter() - started
    source_manifest = artifacts.read_manifest(source_dir)
    digest = artifacts.write_manifest(
        target_dir,
        {
            "format": TRACKS_FORMAT,
            "combination": str(target_combination),
            "detector_variant": target_combination.detector,
            "reid_variant": target_combination.reid,
            "tracker_variant": target_tracker,
            "split": args.split,
            "origin": "postprocess",
            "rung": args.rung,
            "derived_from": str(source_combination),
            "source_settings": source_manifest.get("settings"),
            "postprocess": {
                "n_min": args.n_min,
                "n_dti": args.n_dti,
                "interpolation": "linear in tlwh between the observed rows either side of a gap",
                "smoothing": GBI_PARAMS if args.rung == "dti-gbi" else None,
                "confidence_policy": (
                    "upstream: every row overwritten"
                    if args.upstream_confidence
                    else "observed rows keep the tracker's confidence; interpolated rows are -1"
                ),
                "verified_against_upstream_dti": bool(args.verify_against_upstream),
            },
            "wall_seconds": round(elapsed, 2),
            "sequences": entries,
            "totals": {
                "sequences": len(entries),
                "boxes": sum(int(e["boxes"]) for e in entries),
                "source_boxes": sum(int(e["source_boxes"]) for e in entries),
                "interpolated_boxes": sum(int(e["interpolated_boxes"]) for e in entries),
            },
        },
    )
    total_added = sum(int(e["interpolated_boxes"]) for e in entries)
    total_source = sum(int(e["source_boxes"]) for e in entries)
    print(
        f"\n{args.rung}: +{total_added} boxes on {total_source} "
        f"({100 * total_added / max(total_source, 1):.2f}%) in {elapsed:.1f}s"
    )
    print(f"manifest {artifacts.manifest_path(target_dir)} sha256 {digest}")


if __name__ == "__main__":
    main()
