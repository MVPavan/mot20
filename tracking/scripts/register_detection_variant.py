"""Register an existing MOT-format detection directory as an L1 variant.

This makes externally produced detections, such as the published YOLOX
``det_yoloxx20`` files already in the dataset tree, ordinary cells of the
combination grid rather than a special case handled by separate code.

Example:
    python tracking/scripts/register_detection_variant.py \
        --variant yoloxx20 --split val_half \
        --source-pattern 'datasets/val_half/{sequence}/det_yoloxx20/det_yoloxx20.txt' \
        --description 'Published ByteTrack YOLOX-X MOT20 detections, as shipped in the dataset tree'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mot20_tracking.artifacts import (  # noqa: E402
    DETECTIONS_FORMAT,
    TrackingArtifacts,
    sha256_file,
)
from mot20_tracking.detections import (  # noqa: E402
    read_detections,
    validate_detections,
    write_detections,
)
from mot20_tracking.sequences import read_split  # noqa: E402
from mot20_tracking.variants import validate_slug  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="detector variant slug")
    parser.add_argument("--split", required=True, help="val_half, train, or test")
    parser.add_argument(
        "--source-pattern",
        required=True,
        help="repo-relative path template containing {sequence}",
    )
    parser.add_argument("--description", required=True, help="provenance note for the manifest")
    parser.add_argument(
        "--artifact-root", default="artifacts/tracking", help="tracking artifact root"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_slug(args.variant, "detector")
    if "{sequence}" not in args.source_pattern:
        raise SystemExit("--source-pattern must contain {sequence}")

    artifacts = TrackingArtifacts(args.artifact_root)
    level_dir = artifacts.detections_dir(args.variant, args.split)
    if artifacts.manifest_path(level_dir).exists():
        raise SystemExit(f"variant already registered: {level_dir}")

    sequences = read_split(args.split)
    entries = []
    for sequence in sequences:
        source = Path(args.source_pattern.format(sequence=sequence.name))
        if not source.is_file():
            raise SystemExit(f"missing source detections: {source}")
        detections = read_detections(source, sequence.name, sequence.length)
        statistics = validate_detections(
            detections, sequence.width, sequence.height, sequence.length
        )
        destination = artifacts.detections_file(args.variant, args.split, sequence.name)
        write_detections(destination, detections, sequence.length)
        entries.append(
            {
                "sequence": sequence.name,
                "source": str(source),
                "source_sha256": sha256_file(source),
                "written_sha256": sha256_file(destination),
                "width": sequence.width,
                "height": sequence.height,
                **statistics,
            }
        )
        print(
            f"{sequence.name}: {statistics['detections']:>7} detections, "
            f"max/frame {statistics['max_per_frame']:>3}, "
            f"score {statistics['min_score']:.4f}-{statistics['max_score']:.4f}, "
            f"empty frames {statistics['empty_frames']}"
        )

    digest = artifacts.write_manifest(
        level_dir,
        {
            "format": DETECTIONS_FORMAT,
            "variant": args.variant,
            "split": args.split,
            "origin": "registered-external",
            "description": args.description,
            "source_pattern": args.source_pattern,
            "sequences": entries,
            "totals": {
                "sequences": len(entries),
                "frames": sum(int(entry["frames"]) for entry in entries),
                "detections": sum(int(entry["detections"]) for entry in entries),
            },
        },
    )
    print(f"\nmanifest {artifacts.manifest_path(level_dir)} sha256 {digest}")


if __name__ == "__main__":
    main()
