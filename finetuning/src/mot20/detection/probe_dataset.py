"""Construct small, immutable linked COCO subsets for RF-DETR probe runs."""

from __future__ import annotations

import copy
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from mot20.detection.coco_conversion import write_coco_manifest


def assemble_dense_probe_dataset(
    source_dataset_root: Path,
    destination: Path,
    train_images: int,
    valid_images: int,
    required_train_source: str | None = None,
) -> None:
    """Create a non-overwriting, dense-image subset with linked image roots."""
    if train_images < 1 or valid_images < 1:
        raise ValueError("probe image counts must be positive")
    source_dataset_root = Path(source_dataset_root)
    destination = Path(destination)
    if not source_dataset_root.is_dir():
        raise ValueError(f"source RF-DETR dataset is not a directory: {source_dataset_root}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing probe dataset: {destination}")

    manifests = {
        split: _read_manifest(source_dataset_root / split / "_annotations.coco.json")
        for split in ("train", "valid")
    }
    train_manifest = _subset_manifest(
        manifests["train"],
        train_images,
        require_ignored=True,
        required_source=required_train_source,
    )
    valid_manifest = _subset_manifest(manifests["valid"], valid_images, require_ignored=False)
    destination.mkdir(parents=True)
    for split, manifest in (("train", train_manifest), ("valid", valid_manifest)):
        split_root = destination / split
        split_root.mkdir()
        for prefix in sorted({Path(image["file_name"]).parts[0] for image in manifest["images"]}):
            source_root = source_dataset_root / split / prefix
            if not source_root.is_dir():
                raise ValueError(f"source image root is not a directory: {source_root}")
            _link_directory(split_root / prefix, source_root)
        write_coco_manifest(manifest, split_root / "_annotations.coco.json")


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing source COCO manifest: {path}")
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("images"), list) or not isinstance(manifest.get("annotations"), list):
        raise ValueError(f"invalid source COCO manifest: {path}")
    return manifest


def _subset_manifest(
    manifest: dict[str, Any],
    image_count: int,
    require_ignored: bool,
    required_source: str | None = None,
) -> dict[str, Any]:
    annotations_by_image: dict[Any, list[dict[str, Any]]] = {}
    positive_counts: Counter[Any] = Counter()
    ignored_counts: Counter[Any] = Counter()
    for annotation in manifest["annotations"]:
        image_id = annotation.get("image_id")
        annotations_by_image.setdefault(image_id, []).append(annotation)
        (ignored_counts if int(annotation.get("iscrowd", 0)) else positive_counts)[image_id] += 1
    if image_count > len(manifest["images"]):
        raise ValueError(f"requested {image_count} images from a manifest with only {len(manifest['images'])} images")
    ranked_images = sorted(
        manifest["images"],
        key=lambda image: (-positive_counts[image["id"]], -ignored_counts[image["id"]], str(image["file_name"])),
    )
    selected_images = ranked_images[:image_count]
    if required_source and not any(image.get("source_dataset") == required_source for image in selected_images):
        source_candidates = [image for image in ranked_images if image.get("source_dataset") == required_source]
        if not source_candidates:
            raise ValueError(f"source manifest contains no image from required source: {required_source}")
        selected_images[-1] = source_candidates[0]
    if require_ignored and not any(ignored_counts[image["id"]] for image in selected_images):
        ignored_candidates = [image for image in ranked_images if ignored_counts[image["id"]] and image not in selected_images]
        if not ignored_candidates:
            raise ValueError("source train manifest contains no ignored-region image")
        replacement_index = next(
            (
                index
                for index, image in enumerate(selected_images)
                if not required_source or image.get("source_dataset") != required_source
            ),
            None,
        )
        if replacement_index is None:
            raise ValueError("cannot retain both an ignored-region image and the required source")
        selected_images[replacement_index] = ignored_candidates[0]
    selected_ids = {image["id"] for image in selected_images}
    selected_videos = {
        video["id"]: video
        for video in manifest.get("videos", [])
        if video["id"] in {image.get("video_id") for image in selected_images}
    }
    metadata = copy.deepcopy(manifest.get("metadata", {}))
    metadata["probe_subset"] = {
        "selection": "highest_positive_density_with_ignored_train_image",
        "required_train_source": required_source,
        "source_image_count": len(manifest["images"]),
        "selected_image_count": len(selected_images),
    }
    return {
        "images": copy.deepcopy(selected_images),
        "annotations": [copy.deepcopy(annotation) for annotation in manifest["annotations"] if annotation.get("image_id") in selected_ids],
        "videos": [copy.deepcopy(video) for video in selected_videos.values()],
        "categories": copy.deepcopy(manifest.get("categories")),
        "metadata": metadata,
    }


def _link_directory(link_path: Path, target_path: Path) -> None:
    link_path.symlink_to(os.path.relpath(target_path.resolve(), link_path.parent.resolve()), target_is_directory=True)