"""Derive a higher-threshold detection variant, and its embeddings, by filtering.

Exporting detections low and filtering upward is why the export threshold was
set below any tracker threshold: a stricter variant costs no inference and no
ReID forward passes, because rows are simply dropped and the corresponding
embedding rows are selected.

This exists to remove a specific confound. The published YOLOX detections were
written at a 0.1 floor, while the RF-DETR export used 0.05, so the RF-DETR
tracker run received roughly twice as many low-confidence candidates for
BoostTrack's confidence boosting to promote. Comparing at a matched floor
separates "this detector behaves differently" from "this export was cut
differently".
"""

from __future__ import annotations

import argparse
import sys
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
from mot20_tracking.sequences import read_split  # noqa: E402
from mot20_tracking.variants import validate_slug  # noqa: E402


def greedy_nms(rows: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Return a row-order boolean mask keeping the greedy-NMS survivors.

    Standard score-descending suppression, matching what a detector with
    built-in NMS would have applied before writing its detections.
    """
    if rows.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    boxes = rows[:, :4].astype(np.float64)
    scores = rows[:, 4].astype(np.float64)
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    order = np.argsort(-scores)
    keep = np.zeros(rows.shape[0], dtype=bool)
    while order.size:
        current = order[0]
        keep[current] = True
        if order.size == 1:
            break
        rest = order[1:]
        left = np.maximum(boxes[current, 0], boxes[rest, 0])
        top = np.maximum(boxes[current, 1], boxes[rest, 1])
        right = np.minimum(boxes[current, 2], boxes[rest, 2])
        bottom = np.minimum(boxes[current, 3], boxes[rest, 3])
        inter = np.maximum(right - left, 0) * np.maximum(bottom - top, 0)
        union = areas[current] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou < iou_threshold]
    return keep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="existing detector variant slug")
    parser.add_argument("--target", required=True, help="new detector variant slug")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=None,
        help="apply greedy NMS at this IoU after the score filter; the published "
        "YOLOX detections were NMS-filtered at 0.7 while RF-DETR set prediction uses none",
    )
    parser.add_argument("--split", default="val_half")
    parser.add_argument(
        "--reid",
        default=None,
        help="also derive this ReID variant's embeddings by row selection",
    )
    parser.add_argument("--artifact-root", default="artifacts/tracking")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_slug(args.source, "detector")
    validate_slug(args.target, "detector")
    artifacts = TrackingArtifacts(REPO_ROOT / args.artifact_root)

    source_dir = artifacts.detections_dir(args.source, args.split)
    if not artifacts.manifest_path(source_dir).exists():
        raise SystemExit(f"source variant not exported: {source_dir}")
    target_dir = artifacts.detections_dir(args.target, args.split)
    if artifacts.manifest_path(target_dir).exists():
        raise SystemExit(f"target variant already exists: {target_dir}")
    source_manifest = artifacts.read_manifest(source_dir)

    embeddings_source = None
    if args.reid is not None:
        embeddings_source = artifacts.embeddings_dir(args.source, args.reid, args.split)
        if not artifacts.manifest_path(embeddings_source).exists():
            raise SystemExit(f"source embeddings not exported: {embeddings_source}")
        if artifacts.manifest_path(
            artifacts.embeddings_dir(args.target, args.reid, args.split)
        ).exists():
            raise SystemExit("target embeddings already exist")

    sequences = read_split(args.split, repo_root=REPO_ROOT)
    detection_entries = []
    embedding_entries = []
    for sequence in sequences:
        source_file = artifacts.detections_file(args.source, args.split, sequence.name)
        detections = read_detections(source_file, sequence.name, sequence.length)

        kept_mask_by_frame = {}
        filtered_frames = {}
        for frame_id in range(1, sequence.length + 1):
            rows = detections.rows_for(frame_id)
            mask = rows[:, 4] >= args.min_score if rows.size else np.zeros((0,), dtype=bool)
            if args.nms_iou is not None and mask.any():
                # Suppress within the score-surviving subset, then map the
                # result back to full row order so embedding rows stay aligned.
                survivors = np.flatnonzero(mask)
                suppressed = greedy_nms(rows[survivors], args.nms_iou)
                mask = np.zeros_like(mask)
                mask[survivors[suppressed]] = True
            kept_mask_by_frame[frame_id] = mask
            filtered_frames[frame_id] = rows[mask]
        filtered = SequenceDetections(sequence=sequence.name, frames=filtered_frames)
        statistics = validate_detections(
            filtered, sequence.width, sequence.height, sequence.length
        )
        destination = artifacts.detections_file(args.target, args.split, sequence.name)
        write_detections(destination, filtered, sequence.length)
        detection_entries.append(
            {
                "sequence": sequence.name,
                "written_sha256": sha256_file(destination),
                "width": sequence.width,
                "height": sequence.height,
                **statistics,
            }
        )
        print(
            f"{sequence.name}: {statistics['detections']:>7} kept of "
            f"{detections.detection_count:>7} "
            f"({100 * statistics['detections'] / max(detections.detection_count, 1):.1f}%), "
            f"max/frame {statistics['max_per_frame']}"
        )

        if embeddings_source is not None:
            payload = np.load(
                artifacts.embeddings_file(args.source, args.reid, args.split, sequence.name)
            )
            matrix = payload["embeddings"]
            # Row order is frame-ascending and matches det.txt, so concatenating
            # per-frame masks in the same order selects the surviving rows.
            selector = np.concatenate(
                [kept_mask_by_frame[f] for f in range(1, sequence.length + 1)]
                or [np.zeros((0,), dtype=bool)]
            )
            if selector.size != matrix.shape[0]:
                raise SystemExit(
                    f"{sequence.name}: mask covers {selector.size} rows but "
                    f"embeddings have {matrix.shape[0]}"
                )
            selected = matrix[selector]
            if selected.shape[0] != filtered.detection_count:
                raise SystemExit(f"{sequence.name}: embedding/detection row mismatch")
            embedding_destination = artifacts.embeddings_file(
                args.target, args.reid, args.split, sequence.name
            )
            embedding_destination.parent.mkdir(parents=True, exist_ok=True)
            row_frame = np.concatenate(
                [
                    np.full(int(kept_mask_by_frame[f].sum()), f, dtype=np.int32)
                    for f in range(1, sequence.length + 1)
                ]
                or [np.zeros((0,), dtype=np.int32)]
            )
            np.savez_compressed(
                embedding_destination, embeddings=selected, row_frame=row_frame
            )
            embedding_entries.append(
                {
                    "sequence": sequence.name,
                    "rows": int(selected.shape[0]),
                    "dimension": int(selected.shape[1]) if selected.size else 0,
                    "written_sha256": sha256_file(embedding_destination),
                }
            )

    digest = artifacts.write_manifest(
        target_dir,
        {
            "format": DETECTIONS_FORMAT,
            "variant": args.target,
            "split": args.split,
            "origin": "derived-filter",
            "description": f"{args.source} filtered to score >= {args.min_score}",
            "derived_from": args.source,
            "min_score": args.min_score,
            "inherited": {
                key: source_manifest.get(key)
                for key in ("checkpoint", "checkpoint_sha256", "geometry", "num_select", "nms")
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

    if embeddings_source is not None:
        embedding_digest = artifacts.write_manifest(
            artifacts.embeddings_dir(args.target, args.reid, args.split),
            {
                "format": EMBEDDINGS_FORMAT,
                "detector_variant": args.target,
                "reid_variant": args.reid,
                "split": args.split,
                "origin": "derived-row-selection",
                "derived_from": f"{args.source}__{args.reid}",
                "extractor": "rows selected from the source variant; no ReID forward passes",
                "sequences": embedding_entries,
                "totals": {"rows": sum(int(e["rows"]) for e in embedding_entries)},
            },
        )
        print(f"embeddings manifest sha256 {embedding_digest}")


if __name__ == "__main__":
    main()
