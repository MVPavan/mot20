"""Compare the detector's training annotations against MOT20's official ground truth.

The fine-tuned detector was trained and scored against the Byte65 CVAT-derived
COCO annotations, while HOTA is scored against MOT20's `gt.txt`. If those two
label sets draw the same people with different boxes, a detector can be
faithfully accurate to its own annotations and still lose HOTA, because HOTA
averages over localization thresholds up to 0.95.

This measures the two label sets directly against each other on the same frames.
No model is involved, so the result is a property of the labels alone: it either
explains the localization gap or rules the labels out as its cause.
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

from mot20_tracking.sequences import read_split  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "tracking" / "scripts"))
from analyze_localization import iou_matrix, read_ground_truth  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val_half")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/finetuning/rfdetr-mot20-crowdhuman-byte65-test-adapted-2026-09-04"),
    )
    parser.add_argument("--coco-split", default="valid")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from scipy.optimize import linear_sum_assignment

    annotations_path = args.dataset_root / args.coco_split / "_annotations.coco.json"
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))

    # Split frames are renumbered from 1; COCO keeps the original MOT frame id.
    by_sequence: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for image in payload["images"]:
        by_sequence[str(image["source_sequence"])].append(
            (int(image["source_frame_id"]), int(image["id"]))
        )
    image_to_frame: dict[int, tuple[str, int]] = {}
    for name, pairs in by_sequence.items():
        for position, (_, image_id) in enumerate(sorted(pairs), start=1):
            image_to_frame[image_id] = (name, position)

    coco_boxes: dict[tuple[str, int], list[list[float]]] = defaultdict(list)
    ignored = 0
    for annotation in payload["annotations"]:
        if annotation.get("iscrowd", 0):
            ignored += 1
            continue
        key = image_to_frame.get(int(annotation["image_id"]))
        if key is None:
            continue
        x, y, w, h = (float(v) for v in annotation["bbox"])
        coco_boxes[key].append([x, y, x + w, y + h])

    sequences = read_split(args.split, repo_root=REPO_ROOT)
    split_root = sequences[0].root.parent

    ious: list[float] = []
    residuals: list[list[float]] = []
    coco_total = 0
    gt_total = 0
    matched_total = 0
    per_sequence: dict[str, dict] = {}
    for sequence in sequences:
        gt_frames = read_ground_truth(split_root, sequence.name)
        sequence_ious: list[float] = []
        sequence_coco = 0
        sequence_gt = 0
        sequence_matched = 0
        for frame_id in range(1, sequence.length + 1):
            gt = gt_frames.get(frame_id)
            coco = np.asarray(coco_boxes.get((sequence.name, frame_id), []), dtype=np.float64)
            sequence_coco += coco.shape[0]
            sequence_gt += 0 if gt is None else gt.shape[0]
            if gt is None or gt.shape[0] == 0 or coco.shape[0] == 0:
                continue
            overlaps = iou_matrix(coco, gt)
            rows, columns = linear_sum_assignment(-overlaps)
            keep = overlaps[rows, columns] >= args.match_iou
            rows, columns = rows[keep], columns[keep]
            sequence_matched += rows.size
            if rows.size == 0:
                continue
            sequence_ious.extend(overlaps[rows, columns].tolist())
            source = coco[rows]
            target = gt[columns]
            width = np.maximum(target[:, 2] - target[:, 0], 1e-6)
            height = np.maximum(target[:, 3] - target[:, 1], 1e-6)
            residuals.extend(
                np.stack(
                    [
                        (source[:, 0] - target[:, 0]) / width,
                        (source[:, 1] - target[:, 1]) / height,
                        (source[:, 2] - target[:, 2]) / width,
                        (source[:, 3] - target[:, 3]) / height,
                    ],
                    axis=1,
                ).tolist()
            )
        ious.extend(sequence_ious)
        coco_total += sequence_coco
        gt_total += sequence_gt
        matched_total += sequence_matched
        array = np.asarray(sequence_ious)
        per_sequence[sequence.name] = {
            "coco_boxes": sequence_coco,
            "gt_boxes": sequence_gt,
            "matched": sequence_matched,
            "mean_matched_iou": round(float(array.mean()), 4) if array.size else None,
            "fraction_iou_ge_0.9": round(float((array >= 0.9).mean()), 4) if array.size else None,
        }

    array = np.asarray(ious)
    residual_array = np.asarray(residuals)
    edges = ("x1", "y1", "x2", "y2")
    report = {
        "format": "mot20.tracking.annotation-agreement.v1",
        "split": args.split,
        "coco_annotations": str(annotations_path),
        "match_iou": args.match_iou,
        "coco_boxes": coco_total,
        "coco_ignore_regions_excluded": ignored,
        "gt_boxes": gt_total,
        "matched": matched_total,
        "gt_coverage": round(float(matched_total / max(gt_total, 1)), 4),
        "coco_coverage": round(float(matched_total / max(coco_total, 1)), 4),
        "mean_matched_iou": round(float(array.mean()), 4),
        "median_matched_iou": round(float(np.median(array)), 4),
        "matched_iou_ge": {
            f"{t:.2f}": round(float((array >= t).mean()), 4) for t in (0.7, 0.75, 0.8, 0.9, 0.95)
        },
        "edge_residual_mean": {
            edge: round(float(residual_array[:, i].mean()), 4) for i, edge in enumerate(edges)
        },
        "edge_residual_abs_mean": {
            edge: round(float(np.abs(residual_array[:, i]).mean()), 4)
            for i, edge in enumerate(edges)
        },
        "per_sequence": per_sequence,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
