"""Structural and provenance audits for assembled RF-DETR COCO datasets."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


def audit_rfdetr_coco_dataset(dataset_root: Path, group_detr: int = 13) -> dict[str, Any]:
    """Audit RF-DETR's train/valid COCO layout and return capacity evidence."""
    dataset_root = Path(dataset_root)
    if group_detr < 1:
        raise ValueError(f"group_detr must be positive, got {group_detr}")
    manifests = {
        split: _read_manifest(dataset_root / split / "_annotations.coco.json")
        for split in ("train", "valid")
    }
    split_audits = {
        split: _audit_split(dataset_root / split, manifest, split)
        for split, manifest in manifests.items()
    }
    duplicate_images = _cross_split_duplicates(split_audits["train"]["image_hashes"], split_audits["valid"]["image_hashes"])
    if duplicate_images:
        raise ValueError(f"cross-split duplicate image bytes: {duplicate_images}")
    temporal_overlap = sorted(set(split_audits["train"]["mot20_frames"]) & set(split_audits["valid"]["mot20_frames"]))
    if temporal_overlap:
        raise ValueError(f"MOT20 temporal overlap between train and valid: {temporal_overlap}")
    maximum_labels = max(
        split_audits[split]["density"][kind]["maximum"]
        for split in split_audits
        for kind in ("positive", "ignored")
        if kind == "positive"
    )
    return {
        "format": "mot20.rfdetr.dataset-audit.v1",
        "classification": manifests["train"].get("metadata", {}).get("classification", "clean_held_out_validation"),
        "split_counts": {
            split: {
                "images": split_audits[split]["image_count"],
                "annotations": split_audits[split]["annotation_count"],
            }
            for split in split_audits
        },
        "source_counts": {split: split_audits[split]["source_counts"] for split in split_audits},
        "density": {split: split_audits[split]["density"] for split in split_audits},
        "query_capacity": {
            "group_detr": group_detr,
            "maximum_loss_participating_labels": maximum_labels,
            "minimum_num_queries": _next_multiple(maximum_labels + 1, group_detr),
        },
        "cross_split_duplicate_images": duplicate_images,
        "mot20_temporal_overlap": temporal_overlap,
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing COCO manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid COCO JSON: {path}") from error
    if manifest.get("categories") != [{"id": 1, "name": "pedestrian"}]:
        raise ValueError(f"unexpected COCO categories: {path}")
    if not isinstance(manifest.get("images"), list) or not isinstance(manifest.get("annotations"), list):
        raise ValueError(f"COCO manifest requires image and annotation arrays: {path}")
    return manifest


def _audit_split(split_root: Path, manifest: dict[str, Any], split: str) -> dict[str, Any]:
    images_by_id: dict[Any, dict[str, Any]] = {}
    image_hashes: dict[str, list[str]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    mot20_frames: list[tuple[str, int]] = []
    for image in manifest["images"]:
        image_id = image.get("id")
        if image_id in images_by_id:
            raise ValueError(f"duplicate image ID {image_id!r} in {split}")
        image_path = _resolve_image_path(split_root, image.get("file_name"))
        width, height = _image_size(image_path)
        if (width, height) != (image.get("width"), image.get("height")):
            raise ValueError(f"manifest dimensions do not match image: {image_path}")
        images_by_id[image_id] = image
        image_hashes[_sha256_file(image_path)].append(str(image.get("file_name")))
        source_counts[str(image.get("source_dataset", "unknown"))] += 1
        if image.get("source_dataset") == "MOT20":
            sequence = image.get("source_sequence")
            frame_id = image.get("source_frame_id")
            if not isinstance(sequence, str) or not isinstance(frame_id, int):
                raise ValueError(f"MOT20 image is missing source sequence or frame identity in {split}")
            mot20_frames.append((sequence, frame_id))

    annotation_ids: set[Any] = set()
    positive_counts: Counter[Any] = Counter()
    ignored_counts: Counter[Any] = Counter()
    for annotation in manifest["annotations"]:
        annotation_id = annotation.get("id")
        if annotation_id in annotation_ids:
            raise ValueError(f"duplicate annotation ID {annotation_id!r} in {split}")
        annotation_ids.add(annotation_id)
        image = images_by_id.get(annotation.get("image_id"))
        if image is None:
            raise ValueError(f"annotation {annotation_id!r} references an unknown image in {split}")
        _validate_annotation_box(annotation, image, split)
        if annotation.get("category_id") != 1:
            raise ValueError(f"annotation {annotation_id!r} has a non-pedestrian category in {split}")
        counts = ignored_counts if int(annotation.get("iscrowd", 0)) == 1 else positive_counts
        counts[image["id"]] += 1

    return {
        "image_count": len(images_by_id),
        "annotation_count": len(annotation_ids),
        "source_counts": dict(sorted(source_counts.items())),
        "image_hashes": dict(image_hashes),
        "mot20_frames": mot20_frames,
        "density": {
            "positive": _density_summary(positive_counts, images_by_id),
            "ignored": _density_summary(ignored_counts, images_by_id),
        },
    }


def _resolve_image_path(split_root: Path, file_name: Any) -> Path:
    if not isinstance(file_name, str) or not file_name:
        raise ValueError(f"invalid COCO image file_name under {split_root}")
    lexical_path = split_root / file_name
    if Path(file_name).is_absolute() or ".." in Path(file_name).parts:
        raise ValueError(f"COCO image path escapes split root: {file_name}")
    image_path = lexical_path.resolve()
    if not image_path.is_file():
        raise ValueError(f"COCO image does not resolve under split root: {file_name}")
    return image_path


def _image_size(image_path: Path) -> tuple[int, int]:
    try:
        with Image.open(image_path) as image:
            image.load()
            return image.size
    except OSError as error:
        raise ValueError(f"invalid image file: {image_path}") from error


def _validate_annotation_box(annotation: dict[str, Any], image: dict[str, Any], split: str) -> None:
    box = annotation.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"annotation {annotation.get('id')!r} has an invalid box in {split}")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box):
        raise ValueError(f"annotation {annotation.get('id')!r} has non-finite geometry in {split}")
    left, top, width, height = box
    if width <= 0 or height <= 0 or left < 0 or top < 0 or left + width > image["width"] or top + height > image["height"]:
        raise ValueError(f"annotation {annotation.get('id')!r} is outside image bounds in {split}")


def _density_summary(counts: Counter[Any], images_by_id: dict[Any, dict[str, Any]]) -> dict[str, Any]:
    values = sorted(counts[image_id] for image_id in images_by_id)
    maximum = values[-1] if values else 0
    maximum_image_ids = [image_id for image_id in images_by_id if counts[image_id] == maximum]
    return {
        "maximum": maximum,
        "maximum_image_ids": maximum_image_ids,
        "percentiles": {"p50": _percentile(values, 0.5), "p90": _percentile(values, 0.9), "p95": _percentile(values, 0.95), "p99": _percentile(values, 0.99)},
    }


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    return values[min(len(values) - 1, math.ceil(len(values) * quantile) - 1)]


def _cross_split_duplicates(train_hashes: dict[str, list[str]], valid_hashes: dict[str, list[str]]) -> list[dict[str, Any]]:
    return [
        {"sha256": digest, "train": train_hashes[digest], "valid": valid_hashes[digest]}
        for digest in sorted(set(train_hashes) & set(valid_hashes))
    ]


def _next_multiple(value: int, divisor: int) -> int:
    return math.ceil(value / divisor) * divisor


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()