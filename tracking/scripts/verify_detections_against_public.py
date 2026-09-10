#!/usr/bin/env python3
"""Sanity-check exported detections on a split that has no ground truth.

On `val_half` the export gate is the mAP checksum: reproduce the metric the
checkpoint recorded during training and the whole geometry chain is proven. The
MOT20 *test* split ships no annotations, so that gate cannot run — COCO
evaluation over zero ground-truth boxes returns -1, not a score.

This is the substitute. MOT20 ships public detections in `det/det.txt` for every
sequence, produced by a different detector on the same pixels. They are not
ground truth and their absolute quality is poor, but they are in the same
coordinate frame. So:

- If our boxes were in the wrong coordinate frame (a resize that was not undone,
  a letterbox offset, a transposed axis), agreement with the public detections
  would collapse toward zero.
- Strong agreement does not prove the detections are *good*. It proves they are
  in the right place. Quality on test is unmeasurable locally, by construction.

Reported per sequence:

- coverage: fraction of public detections matched by some exported box at
  IoU >= the threshold. Low coverage means a geometry problem or a recall gap.
- median box height, ours and public. A systematic scale error shows here even
  when coverage looks acceptable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tracking" / "src"))

from mot20_tracking.artifacts import TrackingArtifacts  # noqa: E402
from mot20_tracking.detections import read_detections  # noqa: E402
from mot20_tracking.sequences import SPLIT_ROOTS, read_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=20, help="sample every Nth frame")
    parser.add_argument("--artifact-root", default="artifacts/tracking")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_public_detections(det_path: Path) -> dict[int, np.ndarray]:
    """Read a MOT `det/det.txt` into frame -> (N, 4) xyxy boxes."""
    frames: dict[int, list[list[float]]] = defaultdict(list)
    with det_path.open(newline="", encoding="utf-8") as stream:
        for number, columns in enumerate(csv.reader(stream), start=1):
            if not columns:
                continue
            if len(columns) < 6:
                raise SystemExit(f"{det_path}:{number}: expected at least 6 fields")
            frame = int(float(columns[0]))
            left, top, width, height = (float(columns[i]) for i in (2, 3, 4, 5))
            if width <= 0 or height <= 0:
                continue
            frames[frame].append([left, top, left + width, top + height])
    return {frame: np.asarray(rows, dtype=np.float64) for frame, rows in frames.items()}


def iou_matrix(left_boxes: np.ndarray, right_boxes: np.ndarray) -> np.ndarray:
    if left_boxes.shape[0] == 0 or right_boxes.shape[0] == 0:
        return np.zeros((left_boxes.shape[0], right_boxes.shape[0]))
    x1 = np.maximum(left_boxes[:, None, 0], right_boxes[None, :, 0])
    y1 = np.maximum(left_boxes[:, None, 1], right_boxes[None, :, 1])
    x2 = np.minimum(left_boxes[:, None, 2], right_boxes[None, :, 2])
    y2 = np.minimum(left_boxes[:, None, 3], right_boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    left_area = (left_boxes[:, 2] - left_boxes[:, 0]) * (left_boxes[:, 3] - left_boxes[:, 1])
    right_area = (right_boxes[:, 2] - right_boxes[:, 0]) * (right_boxes[:, 3] - right_boxes[:, 1])
    return inter / np.maximum(left_area[:, None] + right_area[None, :] - inter, 1e-9)


def main() -> None:
    args = parse_args()
    artifacts = TrackingArtifacts(REPO_ROOT / args.artifact_root)
    split_root = REPO_ROOT / SPLIT_ROOTS[args.split]

    results = []
    for sequence in read_split(args.split, repo_root=REPO_ROOT):
        detection_file = artifacts.detections_file(args.detector, args.split, sequence.name)
        exported = read_detections(detection_file, sequence.name, sequence.length)
        public = read_public_detections(split_root / sequence.name / "det" / "det.txt")

        sampled = [f for f in range(1, sequence.length + 1, args.stride) if f in public]
        matched = 0
        public_total = 0
        ours_heights: list[float] = []
        public_heights: list[float] = []
        for frame in sampled:
            public_boxes = public[frame]
            rows = exported.rows_for(frame)
            ours = rows[rows[:, 4] >= args.min_score][:, :4].astype(np.float64)
            public_total += public_boxes.shape[0]
            public_heights.extend((public_boxes[:, 3] - public_boxes[:, 1]).tolist())
            ours_heights.extend((ours[:, 3] - ours[:, 1]).tolist())
            if public_boxes.shape[0] and ours.shape[0]:
                best = iou_matrix(public_boxes, ours).max(axis=1)
                matched += int((best >= args.iou).sum())

        entry = {
            "sequence": sequence.name,
            "image_size": f"{sequence.width}x{sequence.height}",
            "frames_sampled": len(sampled),
            "public_detections": public_total,
            "our_detections": len(ours_heights),
            "coverage_of_public": round(matched / public_total, 4) if public_total else 0.0,
            "median_height_ours": round(float(np.median(ours_heights)), 1) if ours_heights else 0.0,
            "median_height_public": round(float(np.median(public_heights)), 1)
            if public_heights
            else 0.0,
        }
        results.append(entry)
        print(
            f"{entry['sequence']} {entry['image_size']:>9}: "
            f"coverage {entry['coverage_of_public']:.3f} of {public_total:>6} public, "
            f"median height ours {entry['median_height_ours']:>6.1f} "
            f"vs public {entry['median_height_public']:>6.1f}"
        )

    payload = {
        "format": "mot20.tracking.public-detection-crosscheck.v1",
        "detector": args.detector,
        "split": args.split,
        "min_score": args.min_score,
        "iou_threshold": args.iou,
        "frame_stride": args.stride,
        "caveat": "public detections are not ground truth; this checks coordinate frame, not quality",
        "sequences": results,
    }
    if args.output:
        destination = args.output if args.output.is_absolute() else REPO_ROOT / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
