#!/usr/bin/env python3
"""Build a small dense linked RF-DETR corpus for bounded multi-epoch probes."""

from __future__ import annotations

import argparse
from pathlib import Path

from mot20.detection.dataset_audit import audit_rfdetr_coco_dataset
from mot20.detection.probe_dataset import assemble_dense_probe_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-images", type=int, default=32)
    parser.add_argument("--valid-images", type=int, default=8)
    parser.add_argument("--required-train-source")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assemble_dense_probe_dataset(
        args.source_dataset_root,
        args.output_dir,
        args.train_images,
        args.valid_images,
        required_train_source=args.required_train_source,
    )
    print(audit_rfdetr_coco_dataset(args.output_dir))


if __name__ == "__main__":
    main()