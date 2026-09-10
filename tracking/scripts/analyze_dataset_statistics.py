#!/usr/bin/env python3
"""Compute the dataset statistics that `docs/results-reference.md` quotes.

Four descriptive blocks in `build_results_reference.py` were hardcoded literals:
the frame and identity counts in section 1.1, the size and density block in 1.3,
the CrowdHuman-vs-MOT20 comparison in 2.1, and the effective-geometry line in 3.
Measured results in that document are read from artifacts and cannot drift;
those four could, silently, because nothing recomputed them.

This script derives the first three from the data itself and writes them as
analysis artifacts, so the generator reads them the same way it reads
`gt-overlap-val_half.json`. The fourth is pure arithmetic over a sequence's
dimensions and is computed in the generator directly.

Two artifacts are written:

- ``dataset-statistics-<split>.json`` — ground-truth counts, per-image density,
  COCO size bands, and box-height percentiles for an evaluation split.
- ``source-domain-<build>.json`` — per-source composition of a training build,
  including the box-shape convention comparison from I8.

Sizes and bands follow COCO's convention: area < 32² is small, area >= 96² is
large, everything between is medium, where area is the box's width times height
in original image pixels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tracking" / "src"))

from mot20_tracking.sequences import SPLIT_ROOTS, read_split  # noqa: E402

SMALL_AREA = 32 * 32
LARGE_AREA = 96 * 96

# The training transform is SmallestMaxSize(resolution) then a long-side cap
# that only ever shrinks. Both numbers come from the model config.
DEFAULT_RESOLUTION = 1120
DEFAULT_MAX_SIZE = 1600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val_half")
    parser.add_argument(
        "--coco-annotations",
        type=Path,
        required=True,
        help="COCO valid manifest for the split; supplies size bands and the frame count",
    )
    parser.add_argument(
        "--training-build",
        type=Path,
        default=None,
        help="training dataset root whose train/ split gets the source-domain comparison",
    )
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/tracking"))
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def read_ground_truth(path: Path) -> tuple[dict[int, list[list[float]]], set[int]]:
    """Read a MOT20 ``gt.txt``, keeping only scored pedestrian boxes.

    Rows with ``conf = 0`` are marked not-considered by the benchmark and rows
    with a class other than 1 are not pedestrians. Both are excluded, which is
    the same filter the evaluation applies; counting them would inflate every
    density figure in the document.
    """
    frames: dict[int, list[list[float]]] = defaultdict(list)
    identities: set[int] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for number, columns in enumerate(csv.reader(stream), start=1):
            if not columns:
                continue
            if len(columns) < 8:
                raise SystemExit(f"{path}:{number}: expected at least 8 fields")
            if float(columns[6]) == 0 or int(float(columns[7])) != 1:
                continue
            left, top, width, height = (float(columns[i]) for i in (2, 3, 4, 5))
            if width <= 0 or height <= 0:
                continue
            frames[int(columns[0])].append([left, top, width, height])
            identities.add(int(columns[1]))
    return frames, identities


def split_statistics(split: str, coco_annotations: Path) -> dict:
    """Ground-truth counts, density, size bands, and height percentiles."""
    sequences = read_split(split, repo_root=REPO_ROOT)
    split_root = REPO_ROOT / SPLIT_ROOTS[split]

    per_frame: list[int] = []
    heights: list[float] = []
    areas: list[float] = []
    identities = 0
    per_sequence = []
    for sequence in sequences:
        frames, sequence_identities = read_ground_truth(
            split_root / sequence.name / "gt" / "gt.txt"
        )
        counts = [len(frames.get(f, [])) for f in range(1, sequence.length + 1)]
        per_frame.extend(counts)
        # Identities are numbered per sequence, so they are summed rather than
        # unioned; the same id in two sequences is two different people.
        identities += len(sequence_identities)
        for boxes in frames.values():
            for _, _, width, height in boxes:
                heights.append(height)
                areas.append(width * height)
        per_sequence.append(
            {
                "sequence": sequence.name,
                "frames": sequence.length,
                "boxes": sum(counts),
                "identities": len(sequence_identities),
                "instances_per_frame_mean": round(float(np.mean(counts)), 3),
                "instances_per_frame_max": int(max(counts)),
            }
        )

    with _resolve(coco_annotations).open(encoding="utf-8") as stream:
        coco = json.load(stream)
    coco_boxes = [a for a in coco["annotations"] if not a.get("iscrowd")]
    coco_areas = np.array([a["bbox"][2] * a["bbox"][3] for a in coco_boxes])
    total = len(coco_areas)
    bands = {
        "small": int((coco_areas < SMALL_AREA).sum()),
        "medium": int(((coco_areas >= SMALL_AREA) & (coco_areas < LARGE_AREA)).sum()),
        "large": int((coco_areas >= LARGE_AREA).sum()),
    }

    height_array = np.array(heights)
    return {
        "format": "mot20.tracking.dataset-statistics.v1",
        "split": split,
        "coco_annotations": str(coco_annotations),
        "frames": len(coco["images"]),
        "frames_from_seqinfo": sum(s.length for s in sequences),
        "ground_truth_boxes": len(heights),
        "identities": identities,
        "instances_per_frame_mean": round(float(np.mean(per_frame)), 3),
        "instances_per_frame_max": int(max(per_frame)),
        "coco_boxes_scored": total,
        "size_bands": bands,
        "size_band_fraction": {k: round(v / total, 5) for k, v in bands.items()},
        "box_height_percentiles": {
            f"p{p}": round(float(np.percentile(height_array, p)), 1) for p in (5, 25, 50, 75, 95)
        },
        "sequences": per_sequence,
    }


def rendered_size(width: int, height: int, resolution: int, max_size: int) -> dict:
    """Size an image is resized to by SmallestMaxSize then a shrink-only cap."""
    scale = resolution / min(width, height)
    long_side = max(width, height) * scale
    if long_side > max_size:
        scale *= max_size / long_side
    return {
        "native": f"{width}x{height}",
        "aspect": round(width / height, 3),
        "rendered": f"{round(width * scale)}x{round(height * scale)}",
        "scale": round(scale, 4),
        "short_side": round(min(width, height) * scale),
        "short_side_fraction_of_resolution": round(min(width, height) * scale / resolution, 4),
        "cap_binds": long_side > max_size,
    }


def source_domain(build: Path, resolution: int, max_size: int) -> dict:
    """Per-source composition and box-shape convention of a training build.

    The aspect-ratio comparison is stratified by relative box height and
    restricted to boxes that do not touch a border. Unstratified medians are
    dominated by whichever source contributes more small boxes, and
    border-touching boxes are clipped, so both would confound a convention
    comparison rather than measure one.
    """
    annotations_path = _resolve(build) / "train" / "_annotations.coco.json"
    with annotations_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    images = {i["id"]: i for i in manifest["images"]}

    per_source: dict[str, dict] = defaultdict(
        lambda: {"images": 0, "boxes": 0, "persons": 0, "long_sides": [], "shapes": []}
    )
    for image in manifest["images"]:
        entry = per_source[image.get("source_dataset", "unknown")]
        entry["images"] += 1
        entry["long_sides"].append(max(image["width"], image["height"]))
    for annotation in manifest["annotations"]:
        image = images[annotation["image_id"]]
        entry = per_source[image.get("source_dataset", "unknown")]
        entry["boxes"] += 1
        # An iscrowd box is an ignore region, not a person. Density figures that
        # count them overstate CrowdHuman by ~29% and MOT20 by ~13%, because the
        # two sources mark ignore regions at very different rates.
        if not annotation.get("iscrowd"):
            entry["persons"] += 1
        x, y, width, height = annotation["bbox"]
        if width <= 0 or height <= 0:
            continue
        interior = (
            x > 1 and y > 1 and x + width < image["width"] - 1 and y + height < image["height"] - 1
        )
        entry["shapes"].append((width / height, height / image["height"], interior))

    bands = ((0.05, 0.10), (0.10, 0.15), (0.15, 0.25), (0.25, 0.50))
    sources = {}
    for name, entry in per_source.items():
        long_sides = np.array(entry["long_sides"])
        median_long = int(np.median(long_sides))
        shapes = np.array([(a, h, i) for a, h, i in entry["shapes"]]) if entry["shapes"] else None
        by_band = {}
        for low, high in bands:
            if shapes is None:
                continue
            selected = shapes[(shapes[:, 2] > 0) & (shapes[:, 1] >= low) & (shapes[:, 1] < high)]
            if len(selected) < 100:
                continue
            by_band[f"{low:.2f}-{high:.2f}"] = {
                "boxes": int(len(selected)),
                "median_aspect": round(float(np.median(selected[:, 0])), 4),
            }
        # A square-ish image sits at the median long side; use it to show the
        # resize the training transform actually applies to this source.
        representative = next(
            i for i in manifest["images"] if max(i["width"], i["height"]) == median_long
        )
        sources[name] = {
            "images": entry["images"],
            "boxes": entry["boxes"],
            "persons": entry["persons"],
            "ignore_regions": entry["boxes"] - entry["persons"],
            "ignore_region_fraction": round(1 - entry["persons"] / entry["boxes"], 4),
            "persons_per_image_mean": round(entry["persons"] / entry["images"], 2),
            "boxes_per_image_mean": round(entry["boxes"] / entry["images"], 2),
            "median_long_side": median_long,
            "representative_image": rendered_size(
                representative["width"], representative["height"], resolution, max_size
            ),
            "median_aspect_by_height_band": by_band,
        }

    reference = "MOT20"
    if reference in sources:
        for name, payload in sources.items():
            if name == reference:
                continue
            ratios = [
                payload["median_aspect_by_height_band"][band]["median_aspect"]
                / sources[reference]["median_aspect_by_height_band"][band]["median_aspect"]
                for band in payload["median_aspect_by_height_band"]
                if band in sources[reference]["median_aspect_by_height_band"]
            ]
            payload["aspect_ratio_vs_mot20"] = (
                {
                    "bands_compared": len(ratios),
                    "median": round(float(np.median(ratios)), 4),
                    "min": round(float(np.min(ratios)), 4),
                    "max": round(float(np.max(ratios)), 4),
                }
                if ratios
                else None
            )

    return {
        "format": "mot20.tracking.source-domain.v1",
        "build": str(build),
        "resolution": resolution,
        "max_size": max_size,
        "train_images": len(manifest["images"]),
        "train_annotations": len(manifest["annotations"]),
        "aspect_comparison": (
            "median box aspect w/h, interior boxes only, stratified by relative height; "
            "aspect_ratio_vs_mot20 is the per-band ratio against MOT20"
        ),
        "sources": sources,
    }


def write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def main() -> None:
    args = parse_args()
    artifact_root = _resolve(args.artifact_root)

    statistics = split_statistics(args.split, args.coco_annotations)
    if statistics["frames"] != statistics["frames_from_seqinfo"]:
        raise SystemExit(
            f"frame count disagrees: COCO manifest has {statistics['frames']}, "
            f"seqinfo.ini totals {statistics['frames_from_seqinfo']}"
        )
    print(
        f"{args.split}: {statistics['frames']} frames, "
        f"{statistics['ground_truth_boxes']} boxes, {statistics['identities']} identities, "
        f"{statistics['instances_per_frame_mean']} per frame (max {statistics['instances_per_frame_max']})"
    )
    write(statistics, artifact_root / f"dataset-statistics-{args.split}.json")

    if args.training_build:
        domain = source_domain(args.training_build, args.resolution, args.max_size)
        for name, payload in sorted(domain["sources"].items()):
            ratio = payload.get("aspect_ratio_vs_mot20")
            suffix = f", aspect vs MOT20 {ratio['median']:.3f}" if ratio else ""
            print(
                f"  {name:<12} {payload['images']:>6} images, {payload['boxes']:>7} boxes, "
                f"{payload['persons_per_image_mean']:>6.2f}/image, "
                f"median long side {payload['median_long_side']}{suffix}"
            )
        write(domain, artifact_root / f"source-domain-{Path(args.training_build).name}.json")


if __name__ == "__main__":
    main()
