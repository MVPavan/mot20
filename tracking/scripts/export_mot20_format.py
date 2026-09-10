#!/usr/bin/env python3
"""Emit canonical 10-column MOT20 text files from the layered artifact store.

The artifact store's own formats are internal contracts:

- L1 ``det.txt`` is 9 columns, ``frame,-1,x,y,w,h,score,class,visibility``,
  matching the layout the repository's existing detection variants use.
- L3 track files are already 10 columns, but BoostTrack writes the identity as
  a float (``53.0``), which the MOTChallenge submission spec does not define.

Both are converted here to the format the MOTChallenge benchmark documents:

    frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z

with ``x,y,z`` fixed at ``-1`` for 2D tracking, ``id`` an integer, and ``id``
set to ``-1`` for detections. That is the layout MOT20's own ``det/det.txt``
ships in, so the detection output drops into any tracker that reads public
detections, and the track output is submission-ready.

Nothing is filtered here. Detections are exported below any tracker threshold
on purpose, and re-thresholding is the consumer's decision.

Every export also carries a ``benchmark`` block stating whether the result is
held out. That block is *computed*, not asserted: the training dataset is
located from the checkpoint's run directory and its per-image
``source_sequence`` fields are intersected with the sequences being exported.
A file leaving this repository without that statement invites someone to read
a test-adapted number as a benchmark result.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tracking" / "src"))

from mot20_tracking.artifacts import TrackingArtifacts, sha256_file  # noqa: E402
from mot20_tracking.sequences import read_split  # noqa: E402
from mot20_tracking.variants import Combination  # noqa: E402

MOT20_ROW = "{frame},{identity},{left:.2f},{top:.2f},{width:.2f},{height:.2f},{conf:.4f},-1,-1,-1"

POLICY = "docs/MOTPolicy.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("detections", "tracks"), required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--detector", required=True, help="detector variant slug")
    parser.add_argument("--reid", default=None, help="reid variant slug (tracks only)")
    parser.add_argument("--tracker", default=None, help="tracker variant slug (tracks only)")
    parser.add_argument("--artifact-root", default="artifacts/tracking")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help=(
            "rewrite manifest.json and README.md for an export that already exists, "
            "leaving the .txt predictions untouched. Every recorded checksum is "
            "re-verified first; a mismatch aborts."
        ),
    )
    return parser.parse_args()


def _resolve(path_value: str) -> Path:
    """Resolve a manifest-recorded path, which may be repo-relative or absolute."""
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _trace_provenance(
    manifest: dict,
    key: str,
    artifacts: TrackingArtifacts,
    split: str,
    depth: int = 8,
) -> str | None:
    """Find *key* in a detection manifest, following derived-variant links.

    A filtered or NMS'd variant is a child of the export that produced the raw
    boxes. It records what it changed and nests what it inherited, so the
    checkpoint and training config live one or more hops up the chain. Walking
    that chain is what lets a derived variant carry the same contamination
    statement as its parent instead of degrading to "unverifiable".
    """
    if depth <= 0:
        return None
    value = manifest.get(key) or manifest.get("inherited", {}).get(key)
    if value:
        return value
    parent = manifest.get("derived_from")
    if not parent:
        return None
    level = artifacts.detections_dir(parent, split)
    if not artifacts.manifest_path(level).exists():
        return None
    return _trace_provenance(artifacts.read_manifest(level), key, artifacts, split, depth - 1)


def _training_overlap(checkpoint: str | None, sequences: list[str]) -> dict[str, object]:
    """Count training images drawn from the sequences now being exported.

    This is the contamination test, run rather than asserted. The checkpoint's
    run directory records the dataset it was trained on; that dataset's COCO
    manifest tags every image with the MOT20 sequence it came from. Anything
    that stops the chain is reported as ``verified: False`` rather than being
    quietly treated as "no overlap" — an unverifiable claim of cleanliness is
    worse than no claim.
    """
    unverified: dict[str, object] = {"verified": False, "images": 0, "per_sequence": {}}
    if not checkpoint:
        return {**unverified, "reason": "source manifest records no checkpoint"}
    provenance = _resolve(checkpoint).parent / "run-provenance.json"
    if not provenance.exists():
        return {**unverified, "reason": f"no run-provenance.json beside the checkpoint ({_repo_relative(provenance)})"}
    try:
        with provenance.open(encoding="utf-8") as stream:
            dataset_root = json.load(stream).get("dataset_root")
    except (OSError, json.JSONDecodeError) as error:
        return {**unverified, "reason": f"could not read {_repo_relative(provenance)}: {error}"}
    if not dataset_root:
        return {**unverified, "reason": f"{_repo_relative(provenance)} records no dataset_root"}

    annotations = Path(dataset_root) / "train" / "_annotations.coco.json"
    if not annotations.exists():
        return {**unverified, "reason": f"training manifest not present at {_repo_relative(annotations)}"}
    try:
        with annotations.open(encoding="utf-8") as stream:
            images = json.load(stream)["images"]
    except (OSError, json.JSONDecodeError, KeyError) as error:
        return {**unverified, "reason": f"could not read {_repo_relative(annotations)}: {error}"}

    wanted = set(sequences)
    per_sequence: dict[str, int] = {}
    tagged = 0
    for image in images:
        sequence = image.get("source_sequence")
        if sequence is None:
            continue
        tagged += 1
        if sequence in wanted:
            per_sequence[sequence] = per_sequence.get(sequence, 0) + 1
    if not tagged:
        return {
            **unverified,
            "reason": f"no image in {_repo_relative(annotations)} carries source_sequence; overlap is not checkable",
        }
    return {
        "verified": True,
        "images": sum(per_sequence.values()),
        "per_sequence": dict(sorted(per_sequence.items())),
        "training_images_total": len(images),
        "training_manifest": _repo_relative(annotations),
    }


def _repo_relative(path: Path) -> str:
    """Render a path relative to the repo root; absolute paths leak the machine."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _declared_classification(config: str | None) -> dict[str, object]:
    """Read `[run].classification` from the training config that produced the boxes."""
    if not config:
        return {"value": None, "reason": "source manifest records no training config"}
    path = _resolve(config)
    if not path.exists():
        return {"value": None, "reason": f"training config not present at {_repo_relative(path)}"}
    try:
        with path.open("rb") as stream:
            value = tomllib.load(stream).get("run", {}).get("classification")
    except (OSError, tomllib.TOMLDecodeError) as error:
        return {"value": None, "reason": f"could not read {_repo_relative(path)}: {error}"}
    return {"value": value, "config": str(path.relative_to(REPO_ROOT))}


def _reid_provenance(artifacts: TrackingArtifacts, combination: Combination, split: str) -> dict:
    """Describe the ReID model, which is a contamination axis of its own."""
    level = artifacts.embeddings_dir(combination.detector, combination.reid, split)
    manifest_path = artifacts.manifest_path(level)
    if not manifest_path.exists():
        return {"reid_variant": combination.reid, "verified": False}
    manifest = artifacts.read_manifest(level)
    return {
        "reid_variant": manifest.get("reid_variant", combination.reid),
        "reid_dataset_key": manifest.get("reid_dataset_key"),
        "extractor": manifest.get("extractor"),
        "verified": True,
    }


def _benchmark_block(
    kind: str,
    split: str,
    source_manifest: dict,
    sequences: list[str],
    reid: dict | None,
    artifacts: TrackingArtifacts,
) -> dict:
    """State plainly whether this export is a benchmark result. Usually it is not."""
    config = _trace_provenance(source_manifest, "config", artifacts, split)
    checkpoint = _trace_provenance(source_manifest, "checkpoint", artifacts, split)
    declared = _declared_classification(config)
    overlap = _training_overlap(checkpoint, sequences)

    reasons: list[str] = []
    if overlap["verified"] and overlap["images"]:
        detail = ", ".join(f"{name} ({count})" for name, count in overlap["per_sequence"].items())
        reasons.append(
            f"The detector's training set contains {overlap['images']} images drawn from the "
            f"sequences evaluated here: {detail}. Verified by source_sequence in "
            f"{overlap['training_manifest']}. These sequences are therefore not held out."
        )
    elif not overlap["verified"]:
        reasons.append(
            f"Training/evaluation overlap could not be verified ({overlap.get('reason')}). "
            "Treat this export as not held out until someone checks."
        )
    if declared["value"]:
        reasons.append(
            f"The training config declares classification = \"{declared['value']}\" "
            f"({declared.get('config')})."
        )
    if split == "test":
        reasons.append(
            "MOT20 test ships no public ground truth, so no MOTA, HOTA, or mAP can be "
            "computed locally for these files. Any number quoted alongside them came "
            "from somewhere else."
        )
    if reid and reid.get("reid_dataset_key") == "mot20":
        reasons.append(
            f"The ReID model ({reid.get('reid_variant')}) is MOT20-trained, which is a "
            "second contamination axis independent of the detector."
        )
    reasons.append(
        f"This project does not submit to the MOTChallenge evaluation server ({POLICY}), "
        "so no server score exists to settle any of the above."
    )

    contaminated = bool(overlap["images"]) or not overlap["verified"]
    classification = declared["value"] or ("local_test_adapted" if contaminated else "unclassified")
    return {
        "classification": classification,
        "leaderboard_comparable": False,
        "held_out": not contaminated and declared["value"] != "local_test_adapted",
        "locally_scoreable": split != "test",
        "policy": POLICY,
        "declared_classification": declared,
        "training_evaluation_overlap": overlap,
        "reid": reid,
        "reasons": reasons,
        "kind": kind,
    }


README_TEMPLATE = """# {title}

{headline}

## What these files are

Canonical 10-column MOT20 text files, one per sequence:

    frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z

Coordinates are top-left width/height in original image pixels and frames are
1-based. `x,y,z` are fixed at -1 for 2D tracking. {identity}

Split: `{split}`. Source: `{source}`. Sequences: {sequence_list}.

## Read this before quoting a number

{reasons}

## Provenance

Per-sequence row counts and the SHA-256 of every file here, together with the
checksum of the artifact each was derived from, are in `manifest.json`.
"""


def _write_readme(destination: Path, manifest: dict, refresh: bool) -> None:
    benchmark = manifest["benchmark"]
    kind = manifest["kind"]
    headline = (
        "**These are not benchmark results.** "
        f"Classification: `{benchmark['classification']}`. Not leaderboard-comparable."
        if not benchmark["held_out"]
        else f"Classification: `{benchmark['classification']}`."
    )
    body = README_TEMPLATE.format(
        title=f"MOT20-format {kind}: {manifest['source']}",
        headline=headline,
        identity=(
            "`id` is -1 throughout: detections carry no identity, so these drop "
            "straight into any tracker that reads public detections."
            if kind == "detections"
            else "`id` is an integer track identity."
        ),
        split=manifest["split"],
        source=manifest["source"],
        sequence_list=", ".join(entry["sequence"] for entry in manifest["sequences"]),
        reasons="\n".join(f"{n}. {reason}" for n, reason in enumerate(benchmark["reasons"], 1)),
    )
    path = destination / "README.md"
    if refresh and path.exists():
        path.unlink()
    with path.open("x", encoding="utf-8") as stream:
        stream.write(body)


def _convert_detections(source: Path, sequence_length: int) -> tuple[list[str], dict[str, float]]:
    """Rewrite a 9-column L1 detection file as canonical 10-column MOT20 rows."""
    lines: list[str] = []
    frames_seen: set[int] = set()
    min_score, max_score = 1.0, 0.0
    with source.open(encoding="utf-8") as stream:
        for number, raw in enumerate(stream, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                raise SystemExit(f"{source}:{number}: expected at least 7 fields, got {len(parts)}")
            frame = int(float(parts[0]))
            left, top, width, height, score = (float(parts[i]) for i in (2, 3, 4, 5, 6))
            if not 1 <= frame <= sequence_length:
                raise SystemExit(f"{source}:{number}: frame {frame} outside 1..{sequence_length}")
            if width <= 0 or height <= 0:
                raise SystemExit(f"{source}:{number}: non-positive box extent")
            frames_seen.add(frame)
            min_score, max_score = min(min_score, score), max(max_score, score)
            lines.append(
                MOT20_ROW.format(
                    frame=frame,
                    identity=-1,
                    left=left,
                    top=top,
                    width=width,
                    height=height,
                    conf=score,
                )
            )
    return lines, {
        "rows": len(lines),
        "frames_with_rows": len(frames_seen),
        "min_score": round(min_score, 4) if lines else 0.0,
        "max_score": round(max_score, 4) if lines else 0.0,
    }


def _convert_tracks(source: Path, sequence_length: int) -> tuple[list[str], dict[str, float]]:
    """Rewrite an L3 track file with integer identities and fixed 10 columns."""
    lines: list[str] = []
    identities: set[int] = set()
    frames_seen: set[int] = set()
    with source.open(encoding="utf-8") as stream:
        for number, raw in enumerate(stream, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                raise SystemExit(f"{source}:{number}: expected at least 7 fields, got {len(parts)}")
            frame = int(float(parts[0]))
            # BoostTrack writes identities as floats. They are integral track
            # numbers, so a non-integral value means the file is not what this
            # converter thinks it is and must not be silently rounded.
            raw_identity = float(parts[1])
            if raw_identity != int(raw_identity):
                raise SystemExit(f"{source}:{number}: non-integral track id {raw_identity}")
            identity = int(raw_identity)
            if identity < 1:
                raise SystemExit(f"{source}:{number}: track id {identity} is not a positive integer")
            left, top, width, height, conf = (float(parts[i]) for i in (2, 3, 4, 5, 6))
            if not 1 <= frame <= sequence_length:
                raise SystemExit(f"{source}:{number}: frame {frame} outside 1..{sequence_length}")
            if width <= 0 or height <= 0:
                raise SystemExit(f"{source}:{number}: non-positive box extent")
            identities.add(identity)
            frames_seen.add(frame)
            lines.append(
                MOT20_ROW.format(
                    frame=frame,
                    identity=identity,
                    left=left,
                    top=top,
                    width=width,
                    height=height,
                    conf=conf,
                )
            )
    return lines, {
        "rows": len(lines),
        "frames_with_rows": len(frames_seen),
        "track_ids": len(identities),
    }


def main() -> None:
    args = parse_args()
    artifacts = TrackingArtifacts(REPO_ROOT / args.artifact_root)
    destination = (
        args.destination if args.destination.is_absolute() else REPO_ROOT / args.destination
    )
    if destination.exists() and not args.refresh_manifest:
        raise SystemExit(f"refusing to overwrite existing destination: {destination}")
    if args.refresh_manifest and not destination.exists():
        raise SystemExit(f"--refresh-manifest needs an existing export: {destination}")

    if args.kind == "tracks":
        if not args.reid or not args.tracker:
            raise SystemExit("--reid and --tracker are required for --kind tracks")
        combination = Combination(args.detector, args.reid, args.tracker)
        level_dir = artifacts.tracks_dir(combination, args.split)
        source_label = str(combination)
    else:
        if args.reid or args.tracker:
            raise SystemExit("--reid and --tracker do not apply to --kind detections")
        combination = None
        level_dir = artifacts.detections_dir(args.detector, args.split)
        source_label = args.detector
    if not artifacts.manifest_path(level_dir).exists():
        raise SystemExit(f"source level has no manifest: {level_dir}")
    source_manifest = artifacts.read_manifest(level_dir)

    sequences = read_split(args.split, repo_root=REPO_ROOT)
    if args.refresh_manifest:
        entries = _verified_existing_entries(artifacts, destination)
    else:
        destination.mkdir(parents=True)
        entries = _convert_all(artifacts, args, combination, destination, sequences)

    reid = _reid_provenance(artifacts, combination, args.split) if combination else None
    manifest = _build_manifest(
        args, level_dir, source_manifest, source_label, entries, reid, artifacts
    )
    manifest_path = destination / "manifest.json"
    if args.refresh_manifest and manifest_path.exists():
        manifest_path.unlink()
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    _write_readme(destination, manifest, args.refresh_manifest)

    benchmark = manifest["benchmark"]
    print(f"\n{manifest['totals']['rows']} rows across {len(entries)} sequences -> {destination}")
    print(f"classification: {benchmark['classification']}, held_out: {benchmark['held_out']}")
    overlap = benchmark["training_evaluation_overlap"]
    if overlap["verified"] and overlap["images"]:
        print(f"training/eval overlap: {overlap['images']} images {overlap['per_sequence']}")
    elif not overlap["verified"]:
        print(f"training/eval overlap NOT VERIFIED: {overlap.get('reason')}")


def _verified_existing_entries(artifacts: TrackingArtifacts, destination: Path) -> list[dict]:
    """Re-check an existing export's checksums before rewriting its metadata.

    Refresh only ever touches ``manifest.json`` and ``README.md``. If a ``.txt``
    file no longer matches the checksum the manifest recorded for it, the two
    are out of step and rewriting the manifest would paper over that.
    """
    manifest_path = destination / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"--refresh-manifest needs an existing manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as stream:
        entries = json.load(stream)["sequences"]
    for entry in entries:
        target = destination / f"{entry['sequence']}.txt"
        if not target.exists():
            raise SystemExit(f"manifest lists {entry['sequence']} but {target} is missing")
        actual = sha256_file(target)
        if actual != entry["written_sha256"]:
            raise SystemExit(
                f"{target} has changed since the manifest was written "
                f"(recorded {entry['written_sha256'][:12]}, found {actual[:12]}); "
                "refusing to rewrite metadata over a file that no longer matches it"
            )
        print(f"{entry['sequence']}: {entry['rows']:>8} rows, checksum verified")
    return entries


def _convert_all(
    artifacts: TrackingArtifacts,
    args: argparse.Namespace,
    combination: Combination | None,
    destination: Path,
    sequences: list,
) -> list[dict]:
    entries = []
    for sequence in sequences:
        if args.kind == "tracks":
            source = artifacts.tracks_file(combination, args.split, sequence.name)
            lines, statistics = _convert_tracks(source, sequence.length)
        else:
            source = artifacts.detections_file(args.detector, args.split, sequence.name)
            lines, statistics = _convert_detections(source, sequence.length)
        target = destination / f"{sequence.name}.txt"
        with target.open("x", encoding="utf-8") as stream:
            stream.write("\n".join(lines))
            if lines:
                stream.write("\n")
        entries.append(
            {
                "sequence": sequence.name,
                "frames": sequence.length,
                "source": str(source.relative_to(REPO_ROOT)),
                "source_sha256": sha256_file(source),
                "written_sha256": sha256_file(target),
                **statistics,
            }
        )
        summary = (
            f"{statistics['track_ids']:>5} ids"
            if args.kind == "tracks"
            else f"score {statistics['min_score']:.4f}-{statistics['max_score']:.4f}"
        )
        print(
            f"{sequence.name}: {statistics['rows']:>8} rows, "
            f"{statistics['frames_with_rows']:>5}/{sequence.length} frames, {summary}"
        )
    return entries


def _build_manifest(
    args: argparse.Namespace,
    level_dir: Path,
    source_manifest: dict,
    source_label: str,
    entries: list[dict],
    reid: dict | None,
    artifacts: TrackingArtifacts,
) -> dict:
    # Detector provenance always starts at the detections level. A tracks
    # manifest describes the association step, not the model that produced the
    # boxes, so tracing contamination from it would find nothing.
    detections_level = artifacts.detections_dir(args.detector, args.split)
    provenance_source = (
        artifacts.read_manifest(detections_level)
        if artifacts.manifest_path(detections_level).exists()
        else source_manifest
    )
    benchmark = _benchmark_block(
        args.kind,
        args.split,
        provenance_source,
        [entry["sequence"] for entry in entries],
        reid,
        artifacts,
    )
    return {
        "format": "mot20.tracking.mot20-format-export.v2",
        "kind": args.kind,
        "split": args.split,
        "source_level": str(level_dir.relative_to(REPO_ROOT)),
        "source_manifest_format": source_manifest.get("format"),
        "source": source_label,
        "columns": "frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z",
        "identity_column": "-1 (detections carry no identity)"
        if args.kind == "detections"
        else "integer track id",
        "coordinates": "top-left width/height in original image pixels, 1-based frames",
        "filtering": "none; detections are stored below any tracker threshold on purpose",
        "locally_scoreable": args.split != "test",
        "benchmark": benchmark,
        "checkpoint": _trace_provenance(provenance_source, "checkpoint", artifacts, args.split),
        "checkpoint_sha256": _trace_provenance(
            provenance_source, "checkpoint_sha256", artifacts, args.split
        ),
        "sequences": entries,
        "totals": {
            "sequences": len(entries),
            "rows": sum(int(entry["rows"]) for entry in entries),
        },
    }


if __name__ == "__main__":
    main()
