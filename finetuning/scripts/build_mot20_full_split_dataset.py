#!/usr/bin/env python3
"""Build a scorable RF-DETR evaluation layout for a full-length MOT20 split.

The result is consumed by `tracking/scripts/export_detections.py --split train`,
which captures detections from the library's own validation loop so the
inference geometry matches training exactly, and by
`tracking/scripts/analyze_detections.py`, which scores them against the same
`valid/_annotations.coco.json` written here.

Unlike `build_mot20_test_dataset.py`, this layout carries ground truth, so the
capture run reports real mAP rather than the `-1.0` sentinels an annotation-free
split produces. That figure is a free consistency check on the export.

CONTAMINATION. `datasets/MOT20/train` at full length is the union of the
`train_half` and `val_half` ranges. Any detector fine-tuned on either half has
seen these frames, so scores measured here are training-set fit, not
generalization. The caller is responsible for labelling that; this script only
reports the overlap it can see.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "finetuning" / "src"))

from mot20.detection.coco_conversion import (  # noqa: E402
    assemble_rfdetr_evaluation_dataset,
    convert_mot20_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/MOT20/train"))
    parser.add_argument("--split", default="full", choices=("full", "train_half", "val_half"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path


def main() -> None:
    args = parse_args()
    source = _resolve(args.source)
    dataset_root = _resolve(args.dataset_root)

    manifest = convert_mot20_split(source, args.split)
    sequence_roots = {
        video["file_name"]: source / video["file_name"] for video in manifest["videos"]
    }
    assemble_rfdetr_evaluation_dataset(dataset_root, manifest, sequence_roots)

    per_sequence: dict[str, int] = {}
    for image in manifest["images"]:
        per_sequence[image["source_sequence"]] = per_sequence.get(image["source_sequence"], 0) + 1
    positives = sum(1 for a in manifest["annotations"] if not a["iscrowd"])
    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root.relative_to(REPO_ROOT)),
                "split": args.split,
                "images": len(manifest["images"]),
                "annotations": len(manifest["annotations"]),
                "positive_boxes": positives,
                "ignore_regions": len(manifest["annotations"]) - positives,
                "sequences": per_sequence,
                "frame_ranges": manifest["metadata"]["sequence_frame_ranges"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
