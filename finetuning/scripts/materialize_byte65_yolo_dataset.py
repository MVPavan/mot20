#!/usr/bin/env python3
"""Materialize a new local Byte65 YOLO and COCO dataset from an archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mot20.detection.byte65 import materialize_byte65_yolo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = materialize_byte65_yolo_dataset(args.archive, args.dataset_path, args.dataset_name)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()