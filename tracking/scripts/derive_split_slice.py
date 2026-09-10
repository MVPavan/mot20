"""Derive one split's detections and embeddings by slicing another split's.

``val_half`` is not a separate recording: it is the second half of each MOT20
``train`` sequence, with frames renumbered from 1. A detector runs per frame and
a ReID model crops per detection, so neither artifact depends on sequence
length. Slicing therefore REUSES THE SOURCE RUN'S OWN OUTPUTS for those exact
images.

What that does and does not claim. The rows carried across are the numbers the
detector and ReID model actually produced for those images, in that run, from
that checkpoint. They are *not* claimed to be bit-identical to a hypothetical
native ``val_half`` export: a re-run on GPU can differ through CUDA
nondeterminism and batch composition regardless of this script. Reuse is the
stronger position, not the weaker one -- it removes the re-run variance rather
than inheriting it.

Tracking is deliberately NOT sliceable and this script refuses to touch L3.
A tracker carries Kalman state, appearance memory and an ID counter across
frames, so its state at train frame 1393 is not the state of a run that started
fresh at val_half frame 1. Tracks must be re-run on the sliced detections.

Two gates run before anything is written, because both failure modes are silent:

- EVERY target frame is hashed against ``target_frame + offset`` in the source.
  Sampling a handful is not enough: one replaced or mislinked image elsewhere in
  the span would attach another image's detections to it and still produce a
  structurally valid file.
- The source ``row_frame`` vector is validated against the frame vector implied
  by the source detection counts, then sliced and rebased. Equal row counts
  cannot detect a frame-level permutation, so alignment is checked rather than
  assumed, and the target's ``row_frame`` is carried from the source instead of
  being synthesised.

``--audit`` re-runs both gates plus a full value comparison against artifacts
this script produced earlier, writing nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tracking" / "src"))

from mot20_tracking.artifacts import (  # noqa: E402
    DETECTIONS_FORMAT,
    EMBEDDINGS_FORMAT,
    TrackingArtifacts,
    sha256_file,
)
from mot20_tracking.detections import (  # noqa: E402
    SequenceDetections,
    read_detections,
    validate_detections,
    write_detections,
)
from mot20_tracking.sequences import Sequence, read_split  # noqa: E402
from mot20_tracking.variants import validate_slug  # noqa: E402


@dataclass(frozen=True)
class Slice:
    """One sequence's resolved mapping from source frames to target frames."""

    sequence: Sequence
    source: Sequence
    offset: int
    detections: SequenceDetections
    row_selector: np.ndarray
    source_detection_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True, help="detector variant slug")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--target-split", default="val_half")
    parser.add_argument(
        "--reid",
        default=None,
        help="also slice this ReID variant's embeddings by row selection",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="verify previously derived artifacts against the source and write nothing",
    )
    parser.add_argument("--artifact-root", default="artifacts/tracking")
    return parser.parse_args()


def _digest(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def resolve_offset(source: Sequence, target: Sequence) -> int:
    """Return the offset with ``target_frame = source_frame - offset``.

    Every target frame is hashed against its claimed source frame. A partial
    check would let a single substituted image through, and the resulting
    detections would be silently wrong rather than malformed.
    """
    if target.length > source.length:
        raise SystemExit(
            f"{target.name}: target span {target.length} exceeds source {source.length}"
        )
    offset = source.length - target.length
    for target_frame in range(1, target.length + 1):
        source_frame = target_frame + offset
        if _digest(target.frame_path(target_frame)) != _digest(source.frame_path(source_frame)):
            raise SystemExit(
                f"{target.name}: image mismatch at target frame {target_frame} "
                f"(source frame {source_frame}). Offset {offset} is wrong or an image "
                f"differs between splits; refusing to attach detections to the wrong image."
            )
    return offset


def build_slice(
    artifacts: TrackingArtifacts, detector: str, source_split: str, source: Sequence, target: Sequence
) -> Slice:
    """Resolve and verify one sequence's slice without writing anything."""
    offset = resolve_offset(source, target)
    source_detections = read_detections(
        artifacts.detections_file(detector, source_split, target.name),
        target.name,
        source.length,
    )
    per_frame = [source_detections.rows_for(f).shape[0] for f in range(1, source.length + 1)]
    row_selector = np.concatenate(
        [np.full(count, f > offset, dtype=bool) for f, count in enumerate(per_frame, start=1)]
        or [np.zeros((0,), dtype=bool)]
    )
    sliced = SequenceDetections(
        sequence=target.name,
        frames={
            f - offset: source_detections.rows_for(f) for f in range(offset + 1, source.length + 1)
        },
    )
    validate_detections(sliced, target.width, target.height, target.length)
    return Slice(
        sequence=target,
        source=source,
        offset=offset,
        detections=sliced,
        row_selector=row_selector,
        source_detection_count=source_detections.detection_count,
    )


def load_source_embeddings(
    artifacts: TrackingArtifacts, detector: str, reid: str, split: str, plan: Slice
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the source embeddings for one sequence.

    Validates the stored ``row_frame`` against the frame vector implied by the
    source detection file -- two independently written artifacts, so this is a
    real cross-check and not a tautology. Equal row totals cannot reveal a
    frame-level permutation; the per-row frame assignment is what is checked.

    What this does NOT establish: that each vector is the embedding of the box
    on its own row. A permutation of ``embeddings`` alone, with ``row_frame``
    left intact, would pass. Nothing short of recomputing the crops could catch
    that, and it is out of scope here; the check covers the alignment metadata
    this script actually relies on.
    """
    payload = np.load(artifacts.embeddings_file(detector, reid, split, plan.sequence.name))
    matrix = payload["embeddings"]
    if matrix.ndim != 2:
        raise SystemExit(f"{plan.sequence.name}: embeddings must be 2-D, got {matrix.shape}")
    if matrix.shape[0] != plan.row_selector.size:
        raise SystemExit(
            f"{plan.sequence.name}: {plan.row_selector.size} detection rows but "
            f"{matrix.shape[0]} embedding rows"
        )
    if "row_frame" not in payload:
        raise SystemExit(
            f"{plan.sequence.name}: source embeddings carry no row_frame, so alignment "
            f"cannot be verified; refusing to slice on an unverifiable assumption"
        )
    row_frame = payload["row_frame"]
    if row_frame.ndim != 1 or row_frame.shape[0] != matrix.shape[0]:
        raise SystemExit(
            f"{plan.sequence.name}: row_frame has shape {row_frame.shape}, expected "
            f"1-D of length {matrix.shape[0]}"
        )
    if not np.issubdtype(row_frame.dtype, np.integer):
        # Coercing would silently truncate a fractional value into a plausible
        # frame index, turning corrupt metadata into a passing check.
        raise SystemExit(
            f"{plan.sequence.name}: row_frame dtype is {row_frame.dtype}, expected an "
            f"integer type; refusing to coerce"
        )
    # Range-check BEFORE comparing, and compare widened. An integer dtype narrow
    # enough to wrap (uint8 holds 255; MOT20-05 runs to 3315) would otherwise
    # make corrupt metadata compare equal to an expected vector cast down to the
    # same width, since both wrap identically.
    widened = row_frame.astype(np.int64)
    if widened.size and (widened.min() < 1 or widened.max() > plan.source.length):
        raise SystemExit(
            f"{plan.sequence.name}: row_frame holds values outside 1..{plan.source.length} "
            f"(min {widened.min()}, max {widened.max()}); the dtype {row_frame.dtype} is too "
            f"narrow for this sequence or the metadata is corrupt"
        )
    if np.iinfo(row_frame.dtype).max < plan.sequence.length:
        raise SystemExit(
            f"{plan.sequence.name}: row_frame dtype {row_frame.dtype} cannot represent target "
            f"frame {plan.sequence.length}; refusing to write a rebased vector that would wrap"
        )
    expected = np.repeat(
        np.arange(1, plan.source.length + 1, dtype=np.int64),
        [
            plan.detections.rows_for(f - plan.offset).shape[0] if f > plan.offset else 0
            for f in range(1, plan.source.length + 1)
        ],
    )
    # `expected` only covers the target span; compare on that suffix, which is
    # the only part being carried across.
    if not np.array_equal(widened[plan.row_selector], expected):
        raise SystemExit(
            f"{plan.sequence.name}: stored row_frame disagrees with the detection file's "
            f"per-frame counts; the source embeddings are misaligned and must not be sliced"
        )
    return matrix, row_frame


def refuse_partial(paths: list[Path], level: str) -> None:
    """Refuse to proceed when unmanifested files from a failed run are present.

    The artifact store commits a level by writing its manifest last, so a run
    that died mid-sequence leaves files with no manifest. Those would otherwise
    make every retry fail at the first existing file with no explanation.
    """
    existing = [p for p in paths if p.exists()]
    if existing:
        listing = "\n  ".join(str(p) for p in existing[:8])
        more = "" if len(existing) <= 8 else f"\n  ... and {len(existing) - 8} more"
        raise SystemExit(
            f"{level} has {len(existing)} file(s) but no manifest, so a previous run failed "
            f"part-way. Inspect and remove them, then re-run:\n  {listing}{more}"
        )


def main() -> None:
    args = parse_args()
    validate_slug(args.detector, "detector")
    if args.source_split == args.target_split:
        raise SystemExit("source and target splits must differ")
    artifacts = TrackingArtifacts(REPO_ROOT / args.artifact_root)

    source_dir = artifacts.detections_dir(args.detector, args.source_split)
    if not artifacts.manifest_path(source_dir).exists():
        raise SystemExit(f"source detections not exported: {source_dir}")
    target_dir = artifacts.detections_dir(args.detector, args.target_split)
    source_manifest = artifacts.read_manifest(source_dir)

    embeddings_source_dir = None
    embeddings_target_dir = None
    if args.reid is not None:
        embeddings_source_dir = artifacts.embeddings_dir(args.detector, args.reid, args.source_split)
        embeddings_target_dir = artifacts.embeddings_dir(args.detector, args.reid, args.target_split)
        if not artifacts.manifest_path(embeddings_source_dir).exists():
            raise SystemExit(f"source embeddings not exported: {embeddings_source_dir}")

    source_sequences = {s.name: s for s in read_split(args.source_split, repo_root=REPO_ROOT)}
    target_sequences = read_split(args.target_split, repo_root=REPO_ROOT)
    missing = [s.name for s in target_sequences if s.name not in source_sequences]
    if missing:
        raise SystemExit(f"target sequences absent from source split: {missing}")

    if args.audit:
        if not artifacts.manifest_path(target_dir).exists():
            raise SystemExit(f"nothing to audit: {target_dir} has no manifest")
        if args.reid is not None and not artifacts.manifest_path(embeddings_target_dir).exists():
            # Without this, a failed run's leftover but complete embedding files
            # would audit clean and be reported as a committed level.
            # Reachable if a run died between the two manifest writes: detections
            # are committed, embeddings are not. There is no resume path, because
            # a partially committed pair is exactly the state that must not be
            # papered over -- say what to remove instead of guessing.
            raise SystemExit(
                f"nothing to audit: {embeddings_target_dir} has no manifest, so that level "
                f"was never committed even if its files are present.\n"
                f"To recover, remove that directory's .npz files AND the committed "
                f"detections level at {target_dir}, then re-derive both together."
            )
        audit(artifacts, args, source_sequences, target_sequences)
        return

    if artifacts.manifest_path(target_dir).exists():
        raise SystemExit(
            f"target detections already exist: {target_dir}\n"
            f"Use --audit to verify them against the source instead."
        )
    if embeddings_target_dir is not None and artifacts.manifest_path(embeddings_target_dir).exists():
        raise SystemExit(f"target embeddings already exist: {embeddings_target_dir}")
    refuse_partial(
        [artifacts.detections_file(args.detector, args.target_split, s.name) for s in target_sequences],
        f"detections level {target_dir}",
    )
    if args.reid is not None:
        refuse_partial(
            [
                artifacts.embeddings_file(args.detector, args.reid, args.target_split, s.name)
                for s in target_sequences
            ],
            f"embeddings level {embeddings_target_dir}",
        )

    detection_entries: list[dict[str, object]] = []
    embedding_entries: list[dict[str, object]] = []
    for target in target_sequences:
        source = source_sequences[target.name]
        print(f"{target.name}: hashing {target.length} frames ...", flush=True)
        plan = build_slice(artifacts, args.detector, args.source_split, source, target)
        statistics = validate_detections(
            plan.detections, target.width, target.height, target.length
        )

        destination = artifacts.detections_file(args.detector, args.target_split, target.name)
        write_detections(destination, plan.detections, target.length)
        detection_entries.append(
            {
                "sequence": target.name,
                "source_frames": source.length,
                "target_frames": target.length,
                "frame_offset": plan.offset,
                "source_frame_range": [plan.offset + 1, source.length],
                "frames_hash_verified": target.length,
                "written_sha256": sha256_file(destination),
                "width": target.width,
                "height": target.height,
                **statistics,
            }
        )
        print(
            f"{target.name}: frames {plan.offset + 1}..{source.length} -> 1..{target.length} | "
            f"{statistics['detections']:>7} of {plan.source_detection_count:>7} rows | "
            f"mean/frame {statistics['mean_per_frame']}"
        )

        if args.reid is not None:
            matrix, row_frame = load_source_embeddings(
                artifacts, args.detector, args.reid, args.source_split, plan
            )
            selected = matrix[plan.row_selector]
            if selected.shape[0] != plan.detections.detection_count:
                raise SystemExit(f"{target.name}: embedding/detection row mismatch")
            target_row_frame = (row_frame[plan.row_selector] - plan.offset).astype(row_frame.dtype)
            embedding_destination = artifacts.embeddings_file(
                args.detector, args.reid, args.target_split, target.name
            )
            embedding_destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                embedding_destination, embeddings=selected, row_frame=target_row_frame
            )
            embedding_entries.append(
                {
                    "sequence": target.name,
                    "rows": int(selected.shape[0]),
                    "dimension": int(matrix.shape[1]),
                    "dtype": str(selected.dtype),
                    "row_frame_verified": True,
                    "written_sha256": sha256_file(embedding_destination),
                }
            )

    digest = artifacts.write_manifest(
        target_dir,
        {
            "format": DETECTIONS_FORMAT,
            "variant": args.detector,
            "split": args.target_split,
            "origin": "derived-split-slice",
            "description": (
                f"{args.source_split} rows restricted to the {args.target_split} frame span "
                "and renumbered from 1. These are the source run's own detector outputs for "
                "these images, reused; no inference was repeated and no equivalence to a "
                "hypothetical native export is claimed."
            ),
            "derived_from_split": args.source_split,
            "source_manifest_sha256": sha256_file(artifacts.manifest_path(source_dir)),
            "verification": {
                "frame_alignment": (
                    "every target frame md5-compared against source frame target+offset"
                ),
                "offset_source": "seqinfo.ini seqLength difference",
            },
            "inherited": {
                key: source_manifest.get(key)
                for key in (
                    "checkpoint",
                    "checkpoint_sha256",
                    "geometry",
                    "num_select",
                    "nms",
                    "min_score",
                    "derived_from",
                )
                if key in source_manifest
            },
            "sequences": detection_entries,
            "totals": {
                "sequences": len(detection_entries),
                "detections": sum(int(e["detections"]) for e in detection_entries),
            },
        },
    )
    print(f"\ndetections manifest sha256 {digest}")

    if args.reid is not None:
        source_embeddings_manifest = artifacts.read_manifest(embeddings_source_dir)
        embedding_digest = artifacts.write_manifest(
            embeddings_target_dir,
            {
                "format": EMBEDDINGS_FORMAT,
                "detector_variant": args.detector,
                "reid_variant": args.reid,
                "split": args.target_split,
                "origin": "derived-split-slice",
                "derived_from_split": args.source_split,
                "source_manifest_sha256": sha256_file(
                    artifacts.manifest_path(embeddings_source_dir)
                ),
                "extractor": (
                    "rows selected from the source split; no ReID forward passes. The stored "
                    "row_frame was validated against the detection file's per-frame counts "
                    "before slicing, and is carried across rebased rather than synthesised."
                ),
                "inherited": {
                    key: source_embeddings_manifest.get(key)
                    for key in ("extractor", "model", "checkpoint", "checkpoint_sha256",
                                "crop_size", "normalize", "dimension")
                    if key in source_embeddings_manifest
                },
                "sequences": embedding_entries,
                "totals": {"rows": sum(int(e["rows"]) for e in embedding_entries)},
            },
        )
        print(f"embeddings manifest sha256 {embedding_digest}")


def audit(
    artifacts: TrackingArtifacts,
    args: argparse.Namespace,
    source_sequences: dict[str, Sequence],
    target_sequences: list[Sequence],
) -> None:
    """Re-derive in memory and compare against what is on disk. Writes nothing.

    Two independent questions are answered, and both are needed:

    - Does the committed manifest still describe the files on disk? A level is
      committed by writing its manifest last, so the manifest is the record of
      what was produced. Re-deriving the values would not notice a file that was
      rewritten after commit -- recompressing an ``.npz`` changes its bytes while
      leaving every array equal -- so the recorded ``written_sha256`` is checked
      directly, and the declared sequence set must match the split.
    - Do those files still equal what the current code derives from the source?

    Detections are compared as writer output bytes rather than parsed values.
    That is deliberate: the writer's serialisation is part of the artifact
    contract, `read_detections` reconstructs ``x2`` as ``x1 + width`` so a parsed
    round-trip is not exact, and a formatting change is itself something this
    audit should report rather than absorb.
    """
    failures: list[str] = []

    def check_manifest(directory: Path, level: str, hashed: dict[str, Path]) -> None:
        manifest = artifacts.read_manifest(directory)
        declared = {str(entry["sequence"]): entry for entry in manifest.get("sequences", [])}
        if set(declared) != set(hashed):
            failures.append(
                f"{level}: manifest declares {sorted(declared)} but the split is {sorted(hashed)}"
            )
            return
        # Hashing only the declared paths would let an undeclared file sit in a
        # committed level unnoticed, so enumerate the directory too: the claim
        # being made is about the level, not about a list of names.
        expected_paths = {path.resolve() for path in hashed.values()}
        stray = sorted(
            str(p.relative_to(directory))
            for p in directory.rglob("*")
            if p.is_file() and p.name != "manifest.json" and p.resolve() not in expected_paths
        )
        if stray:
            failures.append(
                f"{level}: {len(stray)} file(s) present but not declared by the manifest: {stray}"
            )
        for name, path in hashed.items():
            recorded = declared[name].get("written_sha256")
            if recorded is None:
                failures.append(f"{level}/{name}: manifest records no written_sha256")
            elif not path.exists():
                failures.append(f"{level}/{name}: manifest declares a file that is missing")
            elif sha256_file(path) != recorded:
                failures.append(
                    f"{level}/{name}: on-disk sha256 differs from the committed manifest; "
                    f"the file was modified after the level was committed"
                )

    check_manifest(
        artifacts.detections_dir(args.detector, args.target_split),
        "detections",
        {
            s.name: artifacts.detections_file(args.detector, args.target_split, s.name)
            for s in target_sequences
        },
    )
    if args.reid is not None:
        check_manifest(
            artifacts.embeddings_dir(args.detector, args.reid, args.target_split),
            "embeddings",
            {
                s.name: artifacts.embeddings_file(
                    args.detector, args.reid, args.target_split, s.name
                )
                for s in target_sequences
            },
        )
    print(f"manifest check: {'ok' if not failures else str(len(failures)) + ' problem(s)'}")

    for target in target_sequences:
        source = source_sequences[target.name]
        print(f"{target.name}: hashing {target.length} frames ...", flush=True)
        plan = build_slice(artifacts, args.detector, args.source_split, source, target)

        # Serialize the re-derived slice through the same writer and compare
        # bytes. Comparing parsed floats would need a tolerance -- read_detections
        # reconstructs x2 as x1+width, so a round-trip is not exact -- and a
        # tolerance that admits real corruption must not be reported as a match.
        stored_path = artifacts.detections_file(args.detector, args.target_split, target.name)
        with tempfile.TemporaryDirectory() as scratch:
            expected_path = Path(scratch) / "det.txt"
            write_detections(expected_path, plan.detections, target.length)
            detection_ok = expected_path.read_bytes() == stored_path.read_bytes()
        if not detection_ok:
            failures.append(f"{target.name}: stored detections differ from the re-derived slice")

        embedding_ok = None
        if args.reid is not None:
            matrix, row_frame = load_source_embeddings(
                artifacts, args.detector, args.reid, args.source_split, plan
            )
            expected_matrix = matrix[plan.row_selector]
            expected_row_frame = (row_frame[plan.row_selector] - plan.offset).astype(row_frame.dtype)
            stored = np.load(
                artifacts.embeddings_file(args.detector, args.reid, args.target_split, target.name)
            )
            embedding_ok = np.array_equal(stored["embeddings"], expected_matrix) and np.array_equal(
                stored["row_frame"], expected_row_frame
            )
            if not embedding_ok:
                failures.append(f"{target.name}: embeddings or row_frame differ from the source slice")

        print(
            f"{target.name}: offset {plan.offset} | frames verified {target.length} | "
            f"detections {'OK' if detection_ok else 'MISMATCH'}"
            + ("" if embedding_ok is None else f" | embeddings {'OK' if embedding_ok else 'MISMATCH'}")
        )

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        raise SystemExit(f"audit failed on {len(failures)} check(s)")
    print(
        "\naudit passed: the committed manifests match the files on disk, and those "
        "files match what the current code re-derives from the source"
    )


if __name__ == "__main__":
    main()
