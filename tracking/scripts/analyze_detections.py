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

    return {
        "mAP_50_95": average_precision("all"),
        "mAP_50": average_precision("all", 0.5),
        "mAP_75": average_precision("all", 0.75),
        "mAP_small": average_precision("small"),
        "mAP_medium": average_precision("medium"),
        "mAP_large": average_precision("large"),
        "AR_50_95": mean(recall[:, :, areas["all"], last]),
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

    report: dict[str, dict] = {}
    for variant in args.variants:
        results = []
        per_frame_counts: list[int] = []
        survivors = {f"{t:.2f}": 0 for t in args.tracker_thresholds}
        duplicate_pairs = 0
        duplicate_frames = 0
        total_boxes = 0
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
            "box_height_median": round(float(np.median(heights)), 1),
        }
        print(f"\n=== {variant} ===")
        print(json.dumps(report[variant], indent=2, sort_keys=True))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "format": "mot20.tracking.detection-analysis.v1",
                    "split": args.split,
                    "annotations": str(annotations_path),
                    "max_dets": args.max_dets,
                    "duplicate_iou": args.duplicate_iou,
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
