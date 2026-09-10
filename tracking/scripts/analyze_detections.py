"""Score and characterise L1 detection variants under one identical protocol.

Both detectors are evaluated with pycocotools against the same ``valid``
annotations, so the comparison does not inherit either detector's own
evaluation code. ``iscrowd`` annotations carry MOT20's ignore regions and are
handled natively by COCOeval.

Beyond mAP, this reports the properties that matter to a tracker rather than to
a detection benchmark: how many boxes arrive per frame, how many survive the
tracker's threshold, how tightly matched boxes localise, and how many boxes are
near-duplicates of each other. A set-prediction detector exported without NMS
can score well while handing a tracker a different, harder input.
"""

from __future__ import annotations

import argparse
import contextlib
import io
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--split", default="val_half")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/finetuning/rfdetr-mot20-crowdhuman-byte65-test-adapted-2026-09-04"),
    )
    parser.add_argument("--max-dets", type=int, default=390)
    parser.add_argument(
        "--tracker-thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.4, 0.5, 0.6],
    )
    parser.add_argument("--duplicate-iou", type=float, default=0.75)
    parser.add_argument(
        "--gt-match-iou",
        type=float,
        default=0.5,
        help="IoU at which a prediction counts as covering a ground-truth box, for "
        "the GT-matched duplicate metric",
    )
    parser.add_argument(
        "--gt-duplicate-thresholds",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.20, 0.50],
        help="score thresholds at which to count GT-matched duplicates",
    )
    parser.add_argument("--artifact-root", default="artifacts/tracking")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU for ``(N, 4)`` xyxy boxes."""
    area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    left = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    top = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    right = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    bottom = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.maximum(right - left, 0) * np.maximum(bottom - top, 0)
    union = area[:, None] + area[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _cross_iou(left_boxes: np.ndarray, right_boxes: np.ndarray) -> np.ndarray:
    """IoU between two different sets of ``(N, 4)`` and ``(M, 4)`` xyxy boxes."""
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


def gt_matched_duplicates_multi(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    match_iou: float,
    thresholds: list[float],
) -> dict[float, tuple[int, int, int]]:
    """``gt_matched_duplicates`` at several score thresholds, one IoU matrix.

    The prediction-to-GT overlap does not depend on the score threshold, so
    computing it once and re-masking is several times faster than calling the
    single-threshold version in a loop. On this dataset that matters: a frame
    holds a few hundred predictions against a few hundred GT boxes, and the
    matrix dominates the cost.
    """
    results: dict[float, tuple[int, int, int]] = {}
    if predictions.shape[0] == 0:
        return {t: (0, 0, 0) for t in thresholds}
    if ground_truth.shape[0] == 0:
        return {t: (0, 0, int((predictions[:, 4] >= t).sum())) for t in thresholds}

    overlaps = _cross_iou(predictions[:, :4], ground_truth)
    best_gt = overlaps.argmax(axis=1)
    matched = overlaps.max(axis=1) >= match_iou
    scores = predictions[:, 4]
    for threshold in thresholds:
        kept = scores >= threshold
        selected = matched & kept
        unmatched = int((kept & ~matched).sum())
        if not selected.any():
            results[threshold] = (0, 0, unmatched)
            continue
        assigned = best_gt[selected]
        covered = np.unique(assigned).shape[0]
        results[threshold] = (int(assigned.shape[0] - covered), int(covered), unmatched)
    return results


def gt_matched_duplicates(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    match_iou: float,
) -> tuple[int, int, int]:
    """Count predictions that redundantly cover an already-covered ground-truth box.

    The pairwise duplicate metric counts predictions that overlap *each other*,
    which conflates two different things: a detector emitting the same person
    twice, and two genuinely distinct people who overlap in a dense crowd.
    MOT20 has a great deal of the latter, so that metric has a floor it cannot
    distinguish from the signal.

    This resolves it by anchoring to ground truth. Every prediction is assigned
    to the GT box it overlaps most, and among the predictions assigned to one GT
    box the highest-scoring is the legitimate detection; the rest are redundant.
    Two overlapping people produce two GT boxes and therefore no duplicates,
    however much their boxes overlap.

    Predictions matching no GT box at *match_iou* are false positives, not
    duplicates, and are counted separately rather than folded in.

    Returns ``(redundant, covered_gt, unmatched_predictions)``.
    """
    if predictions.shape[0] == 0:
        return 0, 0, 0
    if ground_truth.shape[0] == 0:
        return 0, 0, predictions.shape[0]
    overlaps = _cross_iou(predictions[:, :4], ground_truth)
    best_gt = overlaps.argmax(axis=1)
    best_iou = overlaps.max(axis=1)
    matched = best_iou >= match_iou
    unmatched = int((~matched).sum())
    if not matched.any():
        return 0, 0, unmatched
    assigned = best_gt[matched]
    # One prediction per covered GT box is legitimate; every further prediction
    # assigned to the same box is redundant.
    covered = np.unique(assigned)
    redundant = int(assigned.shape[0] - covered.shape[0])
    return redundant, int(covered.shape[0]), unmatched


def _summarize(evaluator) -> dict[str, float]:
    """Average precision and recall straight out of COCOeval's accumulator.

    ``precision`` is indexed ``[iou, recall, category, area, maxDets]`` and
    ``recall`` is ``[iou, category, area, maxDets]``; entries are -1 where a
    cell had no data. The reported values use the last ``maxDets`` entry, which
    is the cap this comparison is run at.
    """
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    thresholds = list(evaluator.params.iouThrs)
    last = len(evaluator.params.maxDets) - 1
    areas = {"all": 0, "small": 1, "medium": 2, "large": 3}

    def mean(values: np.ndarray) -> float:
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else float("nan")

    def average_precision(area: str, iou: float | None = None) -> float:
        index = slice(None) if iou is None else thresholds.index(iou)
        return mean(precision[index, :, :, areas[area], last])

    def average_recall(area: str, iou: float | None = None) -> float:
        index = slice(None) if iou is None else thresholds.index(iou)
        return mean(recall[index, :, areas[area], last])

    return {
        "mAP_50_95": average_precision("all"),
        "mAP_50": average_precision("all", 0.5),
        "mAP_75": average_precision("all", 0.75),
        "mAP_small": average_precision("small"),
        "mAP_medium": average_precision("medium"),
        "mAP_large": average_precision("large"),
        "AR_50_95": average_recall("all"),
        # Recall is reported at the same IoU thresholds and area bands as
        # precision. COCO's own summary omits these, but a precision-only
        # breakdown cannot say whether a size band is lost to missed detections
        # or to loose boxes, which is the question this comparison is for.
        "AR_50": average_recall("all", 0.5),
        "AR_75": average_recall("all", 0.75),
        "AR_small": average_recall("small"),
        "AR_medium": average_recall("medium"),
        "AR_large": average_recall("large"),
    }


def main() -> None:
    args = parse_args()
    artifacts = TrackingArtifacts(REPO_ROOT / args.artifact_root)
    sequences = read_split(args.split, repo_root=REPO_ROOT)

    annotations_path = args.dataset_root / "valid" / "_annotations.coco.json"
    with annotations_path.open(encoding="utf-8") as stream:
        coco_payload = json.load(stream)

    # Split frames are renumbered from 1; COCO keeps original MOT frame ids.
    # Ordering by source_frame_id within a sequence recovers the mapping for
    # every variant without relying on any variant's own manifest.
    by_sequence: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for image in coco_payload["images"]:
        by_sequence[str(image["source_sequence"])].append(
            (int(image["source_frame_id"]), int(image["id"]))
        )
    frame_to_image: dict[tuple[str, int], int] = {}
    for name, pairs in by_sequence.items():
        for position, (_, image_id) in enumerate(sorted(pairs), start=1):
            frame_to_image[(name, position)] = image_id

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(annotations_path))

    # Ground truth per image, ignore regions excluded: an iscrowd box is a
    # region the benchmark does not score, so a prediction landing in one is
    # neither a duplicate nor a false positive.
    gt_by_image: dict[int, np.ndarray] = {}
    for annotation in coco_payload["annotations"]:
        if annotation.get("iscrowd"):
            continue
        x, y, width, height = annotation["bbox"]
        gt_by_image.setdefault(int(annotation["image_id"]), []).append(
            [x, y, x + width, y + height]
        )
    gt_by_image = {k: np.asarray(v, dtype=np.float64) for k, v in gt_by_image.items()}

    report: dict[str, dict] = {}
    for variant in args.variants:
        results = []
        per_frame_counts: list[int] = []
        survivors = {f"{t:.2f}": 0 for t in args.tracker_thresholds}
        duplicate_pairs = 0
        duplicate_frames = 0
        total_boxes = 0
        gt_duplicates = {f"{t:.2f}": 0 for t in args.gt_duplicate_thresholds}
        gt_covered = {f"{t:.2f}": 0 for t in args.gt_duplicate_thresholds}
        gt_unmatched = {f"{t:.2f}": 0 for t in args.gt_duplicate_thresholds}
        gt_total = 0
        heights: list[float] = []
        for sequence in sequences:
            detections = read_detections(
                artifacts.detections_file(variant, args.split, sequence.name),
                sequence.name,
                sequence.length,
            )
            for frame_id in range(1, sequence.length + 1):
                rows = detections.rows_for(frame_id)
                per_frame_counts.append(rows.shape[0])
                if rows.shape[0] == 0:
                    continue
                total_boxes += rows.shape[0]
                image_id = frame_to_image[(sequence.name, frame_id)]
                for x1, y1, x2, y2, score in rows:
                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(score),
                        }
                    )
                heights.extend((rows[:, 3] - rows[:, 1]).tolist())
                for threshold in args.tracker_thresholds:
                    survivors[f"{threshold:.2f}"] += int((rows[:, 4] >= threshold).sum())
                overlaps = _iou_matrix(rows[:, :4])
                np.fill_diagonal(overlaps, 0.0)
                pairs = int((np.triu(overlaps) >= args.duplicate_iou).sum())
                duplicate_pairs += pairs
                duplicate_frames += int(pairs > 0)

                truth = gt_by_image.get(image_id)
                if truth is not None:
                    gt_total += truth.shape[0]
                    per_threshold = gt_matched_duplicates_multi(
                        rows, truth, args.gt_match_iou, args.gt_duplicate_thresholds
                    )
                    for threshold, (redundant, covered, unmatched) in per_threshold.items():
                        key = f"{threshold:.2f}"
                        gt_duplicates[key] += redundant
                        gt_covered[key] += covered
                        gt_unmatched[key] += unmatched

        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(results)
            evaluator = COCOeval(coco_gt, coco_dt, "bbox")
            # pycocotools' summarizer indexes maxDets[2], so the list must have
            # three entries; the last is the cap that AP is reported at. Its
            # summarize() is not used: _summarizeDets computes the headline
            # AP[.5:.95] with a hardcoded maxDets=100, which is absent from this
            # list, and silently yields -1. The accumulator is read directly.
            evaluator.params.maxDets = [1, 10, args.max_dets]
            evaluator.evaluate()
            evaluator.accumulate()
        stats = _summarize(evaluator)

        counts = np.asarray(per_frame_counts)
        report[variant] = {
            **{key: round(value, 4) for key, value in stats.items()},
            "boxes_total": total_boxes,
            "boxes_per_frame_mean": round(float(counts.mean()), 2),
            "boxes_per_frame_max": int(counts.max()),
            "boxes_per_frame_p95": round(float(np.percentile(counts, 95)), 1),
            "survivors_at_threshold": survivors,
            "duplicate_pairs_iou_ge": {str(args.duplicate_iou): duplicate_pairs},
            "frames_with_duplicates": duplicate_frames,
            "duplicate_pairs_per_frame": round(duplicate_pairs / max(len(counts), 1), 3),
            # GT-matched duplicates: redundant predictions on an already-covered
            # ground-truth box. Unlike the pairwise count above, two overlapping
            # people cannot inflate this, because they are two GT boxes.
            "gt_matched_duplicates": {
                key: {
                    "redundant": gt_duplicates[key],
                    "redundant_per_frame": round(gt_duplicates[key] / max(len(counts), 1), 3),
                    "redundant_per_covered_gt": round(
                        gt_duplicates[key] / max(gt_covered[key], 1), 4
                    ),
                    "covered_gt": gt_covered[key],
                    "gt_recall": round(gt_covered[key] / max(gt_total, 1), 4),
                    "unmatched_predictions": gt_unmatched[key],
                }
                for key in gt_duplicates
            },
            "gt_boxes_total": gt_total,
            "box_height_median": round(float(np.median(heights)), 1),
        }
        print(f"\n=== {variant} ===")
        print(json.dumps(report[variant], indent=2, sort_keys=True))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "format": "mot20.tracking.detection-analysis.v2",
                    "split": args.split,
                    "annotations": str(annotations_path),
                    "max_dets": args.max_dets,
                    "duplicate_iou": args.duplicate_iou,
                    "gt_match_iou": args.gt_match_iou,
                    "variants": report,
                },
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
