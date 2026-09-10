"""Measure how often MOT20's own ground truth contains heavily overlapping boxes.

`analyze_detections.py` reports a "duplicate pairs per frame" figure by counting
every pair of predicted boxes in a frame whose mutual IoU clears a threshold. It
does not match those predictions to ground truth, so a pair of predictions that
sit on two genuinely overlapping *different* people is counted the same as a
pair of predictions stacked on one person. In a crowd dataset that confound is
real, not hypothetical.

This measures the confound's size directly. It applies the identical pairwise
rule to `gt.txt` itself, where every box is by construction a distinct annotated
person. The result is the rate of legitimate person-on-person overlap in the
data, which is the null model the prediction-side figure has to be read against:
predicted pair rates at or near this level are explainable by real crowding,
and only the excess above it is evidence of duplicate prediction.

No model is involved, so the output is a property of the annotations alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tracking" / "src"))

from mot20_tracking.sequences import SPLIT_ROOTS, read_split  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "tracking" / "scripts"))
from analyze_localization import iou_matrix, read_ground_truth  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val_half")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.75, 0.9],
        help="IoU levels to count pairs at; 0.75 is the level analyze_detections.py uses.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "tracking" / "gt-overlap-val_half.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_root = REPO_ROOT / SPLIT_ROOTS[args.split]

    thresholds = sorted(args.thresholds)
    pairs = {f"{t:.2f}": 0 for t in thresholds}
    frames_with = {f"{t:.2f}": 0 for t in thresholds}
    per_sequence: dict[str, dict[str, float]] = {}
    total_frames = 0
    total_boxes = 0
    max_iou = 0.0

    for sequence in read_split(args.split, REPO_ROOT):
        frames = read_ground_truth(split_root, sequence.name)
        seq_frames = 0
        seq_pairs = {f"{t:.2f}": 0 for t in thresholds}
        for boxes in frames.values():
            seq_frames += 1
            total_boxes += int(boxes.shape[0])
            if boxes.shape[0] < 2:
                continue
            overlaps = np.triu(iou_matrix(boxes, boxes), k=1)
            max_iou = max(max_iou, float(overlaps.max()))
            for threshold in thresholds:
                key = f"{threshold:.2f}"
                count = int((overlaps >= threshold).sum())
                seq_pairs[key] += count
                pairs[key] += count
                frames_with[key] += int(count > 0)
        total_frames += seq_frames
        per_sequence[sequence.name] = {
            "frames": seq_frames,
            **{f"pairs_per_frame_iou_{k}": round(v / seq_frames, 4) for k, v in seq_pairs.items()},
        }

    report = {
        "split": args.split,
        "frames": total_frames,
        "ground_truth_boxes": total_boxes,
        "max_gt_gt_iou": round(max_iou, 4),
        "pairs_total": pairs,
        "pairs_per_frame": {k: round(v / total_frames, 4) for k, v in pairs.items()},
        "frames_with_pair": frames_with,
        "frames_with_pair_fraction": {
            k: round(v / total_frames, 4) for k, v in frames_with.items()
        },
        "per_sequence": per_sequence,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_sequence"}, indent=2))
    print(f"\nwrote {args.output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
