"""Materialize audited Byte65 post-modification labels for local training."""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from mot20.detection.coco_conversion import write_coco_manifest


_ARCHIVE_ROOT = "byte65-modified-images-yolo"
_IMAGE_PREFIX = f"{_ARCHIVE_ROOT}/images/"
_LABEL_PREFIX = f"{_ARCHIVE_ROOT}/post_modification_annotations/"


def materialize_byte65_yolo_dataset(archive_path: Path, dataset_path: Path, dataset_name: str) -> dict[str, Any]:
    """Create an immutable local Byte65 dataset from post-modification labels."""
    archive_path = Path(archive_path)
    dataset_path = Path(dataset_path)
    if not archive_path.is_file():
        raise ValueError(f"Byte65 archive is not a file: {archive_path}")
    if dataset_path.exists():
        raise FileExistsError(f"refusing to overwrite existing Byte65 dataset: {dataset_path}")
    if Path(dataset_name).name != dataset_name:
        raise ValueError(f"Byte65 dataset name must not contain a path separator: {dataset_name}")

    records = _read_archive(archive_path)
    dataset_path.mkdir(parents=True)
    labels_by_image: defaultdict[str, list[str]] = defaultdict(list)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    clipped_annotations: list[dict[str, Any]] = []
    image_id = 0
    annotation_id = 0

    for record in records:
        image_id += 1
        image_path = dataset_path / "images" / record["sequence"] / f"{record['frame_id']:06d}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(record["image_bytes"])
        file_name = image_path.relative_to(dataset_path).as_posix()
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": record["width"],
                "height": record["height"],
                "source_dataset": "Byte65",
                "source_sequence": record["sequence"],
                "source_frame_id": record["frame_id"],
                "source_image_sha256": record["image_sha256"],
                "split": "test_adapted_overlay",
            }
        )
        for source_line, parsed_box in enumerate(record["boxes"], start=1):
            annotation_id += 1
            labels_by_image[file_name].append(parsed_box["yolo_line"])
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": parsed_box["bbox"],
                    "area": parsed_box["bbox"][2] * parsed_box["bbox"][3],
                    "iscrowd": 0,
                    "source_yolo_line": source_line,
                    "source_yolo_bbox": parsed_box["source_yolo_bbox"],
                }
            )
            if parsed_box["clipped"]:
                clipped_annotations.append(
                    {
                        "image": record["source_name"],
                        "source_yolo_line": source_line,
                        "source_yolo_bbox": parsed_box["source_yolo_bbox"],
                        "clipped_bbox": parsed_box["bbox"],
                    }
                )

    for file_name, lines in labels_by_image.items():
        label_path = dataset_path / "labels" / Path(file_name).relative_to("images").with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    image_files = [image["file_name"] for image in images]
    _write_text(dataset_path / "lists" / "all.txt", "\n".join(image_files) + "\n")
    _write_text(dataset_path / "lists" / "train.txt", "\n".join(image_files) + "\n")
    _write_text(
        dataset_path / "data.yaml",
        "path: .\ntrain: lists/train.txt\nval: lists/train.txt\nnc: 1\nnames:\n  0: person\n",
    )
    manifest = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "pedestrian"}],
        "metadata": {
            "format": "mot20.rfdetr.coco.v1",
            "source": "Byte65",
            "split": "test_adapted_overlay",
            "dataset_name": dataset_name,
            "label_source": "post_modification_annotations",
            "source_archive_sha256": _sha256_file(archive_path),
        },
    }
    write_coco_manifest(manifest, dataset_path / "annotations.coco.json")
    audit = {
        "format": "mot20.byte65.audit.v1",
        "dataset_name": dataset_name,
        "classification": "local_test_adapted_overlay",
        "label_source": "post_modification_annotations",
        "source_archive": str(archive_path),
        "source_archive_sha256": _sha256_file(archive_path),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "clipped_annotation_count": len(clipped_annotations),
        "clipped_annotations": clipped_annotations,
        "test_sequences": sorted({image["source_sequence"] for image in images}),
    }
    _write_json(dataset_path / "audit.json", audit)
    return audit


def _read_archive(archive_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        image_members = sorted(name for name in archive.namelist() if name.startswith(_IMAGE_PREFIX) and name.endswith(".jpg"))
        if not image_members:
            raise ValueError(f"Byte65 archive contains no images at {_IMAGE_PREFIX}: {archive_path}")
        for image_member in image_members:
            source_name = PurePosixPath(image_member).stem
            sequence, frame_id = _source_identity(source_name, archive_path)
            label_member = f"{_LABEL_PREFIX}{source_name}.txt"
            try:
                image_bytes = archive.read(image_member)
                label_text = archive.read(label_member).decode("utf-8")
            except KeyError as error:
                raise ValueError(f"Byte65 image is missing its post-modification label: {source_name}") from error
            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.size
            boxes = _parse_yolo_boxes(label_text, width, height, label_member)
            records.append(
                {
                    "source_name": source_name,
                    "sequence": sequence,
                    "frame_id": frame_id,
                    "image_bytes": image_bytes,
                    "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "width": width,
                    "height": height,
                    "boxes": boxes,
                }
            )
    return records


def _source_identity(source_name: str, archive_path: Path) -> tuple[str, int]:
    sequence, separator, frame_text = source_name.rpartition("_")
    if not separator or not sequence or not frame_text.isdigit() or int(frame_text) < 1:
        raise ValueError(f"invalid Byte65 source image name {source_name!r}: {archive_path}")
    return sequence, int(frame_text)


def _parse_yolo_boxes(label_text: str, width: int, height: int, label_member: str) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for line_number, line in enumerate(label_text.splitlines(), start=1):
        if not line.strip():
            continue
        columns = line.split()
        if len(columns) != 5:
            raise ValueError(f"expected five YOLO fields at {label_member}:{line_number}")
        try:
            category_id = int(columns[0])
            x_center, y_center, box_width, box_height = (float(value) for value in columns[1:])
        except ValueError as error:
            raise ValueError(f"invalid YOLO annotation at {label_member}:{line_number}") from error
        if category_id != 0:
            raise ValueError(f"expected YOLO class 0 at {label_member}:{line_number}")
        values = (x_center, y_center, box_width, box_height)
        if not all(math.isfinite(value) for value in values) or box_width <= 0 or box_height <= 0:
            raise ValueError(f"invalid YOLO geometry at {label_member}:{line_number}")
        left = (x_center - box_width / 2) * width
        top = (y_center - box_height / 2) * height
        right = (x_center + box_width / 2) * width
        bottom = (y_center + box_height / 2) * height
        clipped_left = min(max(left, 0.0), float(width))
        clipped_top = min(max(top, 0.0), float(height))
        clipped_right = min(max(right, 0.0), float(width))
        clipped_bottom = min(max(bottom, 0.0), float(height))
        clipped_width = clipped_right - clipped_left
        clipped_height = clipped_bottom - clipped_top
        if clipped_width <= 0 or clipped_height <= 0:
            raise ValueError(f"YOLO box is outside image bounds at {label_member}:{line_number}")
        clipped = (clipped_left, clipped_top, clipped_width, clipped_height) != (left, top, right - left, bottom - top)
        normalized_x_center = (clipped_left + clipped_width / 2) / width
        normalized_y_center = (clipped_top + clipped_height / 2) / height
        normalized_width = clipped_width / width
        normalized_height = clipped_height / height
        boxes.append(
            {
                "bbox": [clipped_left, clipped_top, clipped_width, clipped_height],
                "clipped": clipped,
                "source_yolo_bbox": [x_center, y_center, box_width, box_height],
                "yolo_line": " ".join(
                    [
                        "0",
                        _format_float(normalized_x_center),
                        _format_float(normalized_y_center),
                        _format_float(normalized_width),
                        _format_float(normalized_height),
                    ]
                ),
            }
        )
    return boxes


def _format_float(value: float) -> str:
    return format(value, ".10g")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, content: dict[str, Any]) -> None:
    _write_text(path, json.dumps(content, indent=2, sort_keys=True) + "\n")