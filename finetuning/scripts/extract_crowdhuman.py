#!/usr/bin/env python3
"""Extract and verify CrowdHuman source archives into a new local directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mot20.detection.crowdhuman import extract_crowdhuman_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-archive", type=Path, action="append", required=True)
    parser.add_argument("--val-archive", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--val-annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-images", type=int, default=15_000)
    parser.add_argument("--expected-val-images", type=int, default=4_370)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = extract_crowdhuman_sources(
        args.train_archive,
        args.val_archive,
        args.train_annotations,
        args.val_annotations,
        args.output_dir,
        args.expected_train_images,
        args.expected_val_images,
    )
    print(json.dumps({"output_dir": str(args.output_dir), **receipt}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()