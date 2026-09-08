"""Measure how tightly boxes align with MOT20 ground truth, and in which direction.

HOTA's ``LocA`` is the mean IoU over matched pairs, and ``DetA``/``AssA`` are
averaged over localization thresholds from 0.05 to 0.95. A detector can
therefore win at IoU 0.5 — where MOTA lives — and still lose HOTA, purely
because its boxes sit slightly differently on the same people.

This script separates "how much IoU is lost" from "why": alongside the matched
IoU distribution it reports per-edge residuals normalised by ground-truth box
size. A systematic residual on one edge is an annotation-convention mismatch
and is correctable; symmetric jitter is regression noise and is not.

It reads either an L1 detection variant or an L3 track file, so the same
measurement can be applied before and after the tracker, which distinguishes
detector localization from Kalman-filter smoothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tracking" / "src"))

from mot20_tracking.artifacts import TrackingArtifacts  # noqa: E402
from mot20_tracking.detections import read_detections  # noqa: E402
from mot20_tracking.sequences import read_split  # noqa: E402
from mot20_tracking.variants import Combination  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detections",
        nargs="*",
        default=[],
        help="L1 detector variant slugs to measure",
    )
    parser.add_argument(
        "--tracks",
        nargs="*",
        default=[],
        help="L3 combination slugs (detector__reid__tracker) to measure",
    )
    parser.add_argument("--split", default="val_half")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.4,
        help="score floor for detection variants; the tracker's det_thresh by default",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU below which a detection is not considered the same object",
    )
    parser.add_argument("--artifact-root", default="artifacts/tracking")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_ground_truth(split_root: Path, sequence: str) -> dict[int, np.ndarray]:
    """Ground-truth pedestrian boxes per frame as ``(N, 4)`` xyxy.

    TrackEval's MOT20 pedestrian class uses rows whose confidence flag is 1 and
    whose class is 1; every other row is an ignore or a distractor class.
    """
    frames: dict[int, list[list[float]]] = defaultdict(list)
    path = split_root / sequence / "gt" / "gt.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(",")
        if int(fields[6]) != 1 or int(fields[7]) != 1:
            continue
        frame = int(fields[0])
        x, y, w, h = (float(v) for v in fields[2:6])
        frames[frame].append([x, y, x + w, y + h])
    return {frame: np.asarray(boxes, dtype=np.float64) for frame, boxes in frames.items()}


def read_track_boxes(path: Path) -> dict[int, np.ndarray]:
    frames: dict[int, list[list[float]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(",")
        frame = int(fields[0])
        x, y, w, h = (float(v) for v in fields[2:6])
        frames[frame].append([x, y, x + w, y + h])
    return {frame: np.asarray(boxes, dtype=np.float64) for frame, boxes in frames.items()}


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    area_a = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    left = np.maximum(a[:, None, 0], b[None, :, 0])
    top = np.maximum(a[:, None, 1], b[None, :, 1])
    right = np.minimum(a[:, None, 2], b[None, :, 2])
    bottom = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(right - left, 0) * np.maximum(bottom - top, 0)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def measure(
    boxes_by_frame: dict[int, np.ndarray],
    gt_by_frame: dict[int, np.ndarray],
    match_iou: float,
) -> dict:
    from scipy.optimize import linear_sum_assignment

    ious: list[float] = []
    residuals: list[list[float]] = []
    scale_ratios: list[list[float]] = []
    matched = 0
    predicted = 0
    ground_truth = 0
    for frame, gt in gt_by_frame.items():
        ground_truth += gt.shape[0]
        boxes = boxes_by_frame.get(frame)
        if boxes is None or boxes.shape[0] == 0 or gt.shape[0] == 0:
            continue
        predicted += boxes.shape[0]
        overlaps = iou_matrix(boxes, gt)
        rows, columns = linear_sum_assignment(-overlaps)
        keep = overlaps[rows, columns] >= match_iou
        rows, columns = rows[keep], columns[keep]
        matched += rows.size
        if rows.size == 0:
            continue
        ious.extend(overlaps[rows, columns].tolist())
        prediction = boxes[rows]
        target = gt[columns]
        width = np.maximum(target[:, 2] - target[:, 0], 1e-6)
        height = np.maximum(target[:, 3] - target[:, 1], 1e-6)
        residuals.extend(
            np.stack(
                [
                    (prediction[:, 0] - target[:, 0]) / width,
                    (prediction[:, 1] - target[:, 1]) / height,
                    (prediction[:, 2] - target[:, 2]) / width,
                    (prediction[:, 3] - target[:, 3]) / height,
                ],
                axis=1,
            ).tolist()
        )
        scale_ratios.extend(
            np.stack(
                [
                    (prediction[:, 2] - prediction[:, 0]) / width,
                    (prediction[:, 3] - prediction[:, 1]) / height,
                ],
                axis=1,
            ).tolist()
        )

    for frame, boxes in boxes_by_frame.items():
        if frame not in gt_by_frame:
            predicted += boxes.shape[0]

    iou_array = np.asarray(ious)
    residual_array = np.asarray(residuals)
    ratio_array = np.asarray(scale_ratios)
    edges = ("x1", "y1", "x2", "y2")
    return {
        "boxes": predicted,
        "gt_boxes": ground_truth,
        "matched": matched,
        "recall_at_match_iou": round(float(matched / max(ground_truth, 1)), 4),
        "precision_at_match_iou": round(float(matched / max(predicted, 1)), 4),
        "mean_matched_iou": round(float(iou_array.mean()), 4),
        "median_matched_iou": round(float(np.median(iou_array)), 4),
        "matched_iou_ge": {
            f"{t:.2f}": round(float((iou_array >= t).mean()), 4)
            for t in (0.5, 0.7, 0.75, 0.8, 0.9)
        },
        "edge_residual_mean": {
            edge: round(float(residual_array[:, i].mean()), 4) for i, edge in enumerate(edges)
        },
        "edge_residual_median": {
            edge: round(float(np.median(residual_array[:, i])), 4)
            for i, edge in enumerate(edges)
        },
        "edge_residual_abs_mean": {
            edge: round(float(np.abs(residual_array[:, i]).mean()), 4)
            for i, edge in enumerate(edges)
        },
        "size_ratio_median": {
            "width": round(float(np.median(ratio_array[:, 0])), 4),
            "height": round(float(np.median(ratio_array[:, 1])), 4),
        },
    }


def main() -> None:
    args = parse_args()
    if not args.detections and not args.tracks:
        raise SystemExit("nothing to measure: pass --detections and/or --tracks")
    artifacts = TrackingArtifacts(REPO_ROOT / args.artifact_root)
    sequences = read_split(args.split, repo_root=REPO_ROOT)
    split_root = sequences[0].root.parent

    ground_truth = {s.name: read_ground_truth(split_root, s.name) for s in sequences}

    report: dict[str, dict] = {}
    for variant in args.detections:
        merged_boxes: dict[str, dict[int, np.ndarray]] = {}
        for sequence in sequences:
            detections = read_detections(
                artifacts.detections_file(variant, args.split, sequence.name),
                sequence.name,
                sequence.length,
            )
            frames = {}
            for frame_id in range(1, sequence.length + 1):
                rows = detections.rows_for(frame_id)
                frames[frame_id] = (
                    rows[rows[:, 4] >= args.min_score][:, :4] if rows.size else rows[:, :4]
                )
            merged_boxes[sequence.name] = frames
        report[f"det:{variant}"] = _combine(merged_boxes, ground_truth, args.match_iou)

    for combination in args.tracks:
        merged_boxes = {}
        for sequence in sequences:
            merged_boxes[sequence.name] = read_track_boxes(
                artifacts.tracks_file(
                    Combination.parse(combination), args.split, sequence.name
                )
            )
        report[f"trk:{combination}"] = _combine(merged_boxes, ground_truth, args.match_iou)

    for name, values in report.items():
        print(f"\n=== {name} ===")
        print(json.dumps(values, indent=2, sort_keys=True))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "format": "mot20.tracking.localization-analysis.v1",
                    "split": args.split,
                    "min_score": args.min_score,
                    "match_iou": args.match_iou,
                    "variants": report,
                },
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        print(f"\nwrote {args.output}")


def _combine(
    boxes_by_sequence: dict[str, dict[int, np.ndarray]],
    gt_by_sequence: dict[str, dict[int, np.ndarray]],
    match_iou: float,
) -> dict:
    per_sequence = {
        name: measure(boxes_by_sequence[name], gt_by_sequence[name], match_iou)
        for name in gt_by_sequence
    }
    flat_boxes: dict[int, np.ndarray] = {}
    flat_gt: dict[int, np.ndarray] = {}
    offset = 0
    for name in sorted(gt_by_sequence):
        # Frame ids repeat across sequences; shift them so a single pass over the
        # concatenation still matches within the correct sequence.
        for frame, boxes in boxes_by_sequence[name].items():
            flat_boxes[offset + frame] = boxes
        for frame, boxes in gt_by_sequence[name].items():
            flat_gt[offset + frame] = boxes
        offset += 1_000_000
    combined = measure(flat_boxes, flat_gt, match_iou)
    combined["per_sequence"] = per_sequence
    return combined


if __name__ == "__main__":
    main()
