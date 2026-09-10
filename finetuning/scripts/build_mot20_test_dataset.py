#!/usr/bin/env python3
"""Build the annotation-free RF-DETR inference layout for the MOT20 test split.

The result is consumed by `tracking/scripts/export_detections.py --split test`,
which captures detections from the library's own validation loop so the
inference geometry matches training exactly.

Nothing produced from this dataset can be scored locally: MOT20 test ships no
ground truth. See `convert_mot20_test_split` for the full contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "finetuning" / "src"))

from mot20.detection.coco_conversion import (  # noqa: E402
    assemble_rfdetr_inference_dataset,
    convert_mot20_test_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/MOT20/test"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = (REPO_ROOT / args.source).resolve() if not args.source.is_absolute() else args.source
    dataset_root = (
        (REPO_ROOT / args.dataset_root).resolve()
        if not args.dataset_root.is_absolute()
        else args.dataset_root
    )

    manifest = convert_mot20_test_split(source)
    sequence_roots = {
        video["file_name"]: source / video["file_name"] for video in manifest["videos"]
    }
    assemble_rfdetr_inference_dataset(dataset_root, manifest, sequence_roots)

    per_sequence: dict[str, int] = {}
    for image in manifest["images"]:
        per_sequence[image["source_sequence"]] = per_sequence.get(image["source_sequence"], 0) + 1
    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root.relative_to(REPO_ROOT)),
                "images": len(manifest["images"]),
                "annotations": len(manifest["annotations"]),
                "sequences": per_sequence,
                "sizes": {
                    video["file_name"]: next(
                        f"{image['width']}x{image['height']}"
                        for image in manifest["images"]
                        if image["source_sequence"] == video["file_name"]
                    )
                    for video in manifest["videos"]
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
