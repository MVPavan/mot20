"""Deterministic COCO conversion for the ByteTrack-style MOT20 data workflow."""

from __future__ import annotations

import configparser
import copy
import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Literal


MOT20_IGNORED_PERSON_CLASSES = frozenset({2, 7, 8, 12})
MOT20_NON_PERSON_CLASSES = frozenset({3, 4, 5, 6, 9, 10, 11})
MOT20_SPLIT = Literal["train_half", "val_half"]
CROWDHUMAN_SPLIT = Literal["train", "val"]
CONVERSION_REVISION = "mot20-rfdetr-coco-v2"


def convert_mot20_split(dataset_root: Path, split: MOT20_SPLIT) -> dict[str, Any]:
    """Convert one deterministic MOT20 temporal half to a COCO manifest.

    The frame boundaries match ByteTrack's ``convert_mot20_to_coco.py``:
    ``train_half`` includes frames 1 through ``floor(length / 2) + 1`` and
    ``val_half`` contains the remaining frames. Valid pedestrian rows reproduce
    ByteTrack's ``confidence == 1`` and ``class_id == 1`` filter. Its ignored
    person classes are retained as ``iscrowd`` so the project loss extension can
    remove their overlapping unmatched queries from classification supervision.
    """
    if split not in ("train_half", "val_half"):
        raise ValueError(f"unsupported MOT20 split: {split}")
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise ValueError(f"MOT20 dataset root is not a directory: {dataset_root}")

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    disposition_counts: Counter[str] = Counter()
    box_accounting: Counter[str] = Counter()
    sequence_frame_ranges: dict[str, dict[str, int]] = {}
    image_id = 0
    annotation_id = 0

    sequence_roots = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    for video_id, sequence_root in enumerate(sequence_roots, start=1):
        sequence = _read_mot20_sequence(sequence_root)
        start_frame, stop_frame = _mot20_half_bounds(sequence["length"], split)
        sequence_frame_ranges[sequence["name"]] = {"start": start_frame, "stop": stop_frame}
        videos.append({"id": video_id, "file_name": sequence["name"]})
        frame_to_image_id: dict[int, int] = {}
        for frame_id in range(start_frame, stop_frame + 1):
            image_path = sequence["image_dir"] / f"{frame_id:06d}.jpg"
            if not image_path.is_file():
                raise ValueError(f"missing MOT20 image: {image_path}")
            if _image_size(image_path) != (sequence["width"], sequence["height"]):
                raise ValueError(f"MOT20 image dimensions do not match seqinfo: {image_path}")
            image_id += 1
            frame_to_image_id[frame_id] = image_id
            images.append(
                {
                    "id": image_id,
                    "file_name": f"{sequence['name']}/img1/{frame_id:06d}.jpg",
                    "width": sequence["width"],
                    "height": sequence["height"],
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "source_dataset": "MOT20",
                    "source_sequence": sequence["name"],
                    "source_frame_id": frame_id,
                    "split": split,
                }
            )

        for row in _read_mot20_rows(sequence["gt_path"]):
            frame_id, track_id, left, top, width, height, confidence, class_id, visibility = row
            if frame_id not in frame_to_image_id:
                continue
            disposition = _mot20_disposition(confidence, class_id)
            disposition_counts[disposition] += 1
            if disposition == "excluded":
                box_accounting["excluded"] += 1
                continue
            clipped_box = _clip_box(left, top, width, height, sequence["width"], sequence["height"])
            if clipped_box is None:
                disposition_counts["rejected_invalid_box"] += 1
                box_accounting["rejected_invalid_box"] += 1
                continue
            outcome = "positive" if disposition == "positive_pedestrian" else "ignored"
            box_accounting[f"{'clipped' if _box_was_clipped(clipped_box, left, top, width, height) else 'retained'}_{outcome}"] += 1
            annotation_id += 1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": frame_to_image_id[frame_id],
                    "category_id": 1,
                    "bbox": clipped_box,
                    "area": clipped_box[2] * clipped_box[3],
                    "iscrowd": int(disposition == "ignored_person"),
                    "track_id": track_id,
                    "conf": confidence,
                    "visibility": visibility,
                    "source_class_id": class_id,
                    "source_bbox": [left, top, width, height],
                }
            )

    if not images:
        raise ValueError(f"no MOT20 sequences found under {dataset_root}")
    return {
        "images": images,
        "annotations": annotations,
        "videos": videos,
        "categories": [{"id": 1, "name": "pedestrian"}],
        "metadata": {
            "format": "mot20.rfdetr.coco.v1",
            "conversion_revision": CONVERSION_REVISION,
            "source": "MOT20",
            "split": split,
            "disposition_counts": dict(sorted(disposition_counts.items())),
            "box_accounting": dict(sorted(box_accounting.items())),
            "sequence_frame_ranges": sequence_frame_ranges,
        },
    }


def convert_crowdhuman_split(
    annotation_path: Path,
    image_root: Path,
    split: CROWDHUMAN_SPLIT,
) -> dict[str, Any]:
    """Convert one CrowdHuman ODGT split using ByteTrack's `fbox` convention."""
    if split not in ("train", "val"):
        raise ValueError(f"unsupported CrowdHuman split: {split}")
    annotation_path = Path(annotation_path)
    image_root = Path(image_root)
    if not annotation_path.is_file():
        raise ValueError(f"CrowdHuman annotations are not a file: {annotation_path}")
    if not image_root.is_dir():
        raise ValueError(f"CrowdHuman image root is not a directory: {image_root}")

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    disposition_counts: Counter[str] = Counter()
    box_accounting: Counter[str] = Counter()
    annotation_id = 0
    with annotation_path.open(encoding="utf-8") as stream:
        for image_id, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
                source_image_id = str(record["ID"])
                gtboxes = record["gtboxes"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"invalid CrowdHuman ODGT record at line {image_id}: {annotation_path}") from error
            image_path = image_root / f"{source_image_id}.jpg"
            if not image_path.is_file():
                raise ValueError(f"CrowdHuman image does not exist: {image_path}")
            image_width, image_height = _image_size(image_path)
            images.append(
                {
                    "id": image_id,
                    "file_name": f"{source_image_id}.jpg",
                    "width": image_width,
                    "height": image_height,
                    "source_dataset": "CrowdHuman",
                    "source_image_id": source_image_id,
                    "split": split,
                }
            )
            if not isinstance(gtboxes, list):
                raise ValueError(f"CrowdHuman gtboxes must be a list at line {image_id}: {annotation_path}")
            for source_box_id, gtbox in enumerate(gtboxes):
                try:
                    source_box = gtbox["fbox"]
                    left, top, width, height = (float(value) for value in source_box)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid CrowdHuman fbox at line {image_id}, index {source_box_id}: {annotation_path}"
                    ) from error
                clipped_box = _clip_box(left, top, width, height, image_width, image_height)
                if clipped_box is None:
                    disposition_counts["rejected_invalid_box"] += 1
                    box_accounting["rejected_invalid_box"] += 1
                    continue
                ignored = int(gtbox.get("extra", {}).get("ignore", 0) == 1)
                disposition_counts["ignored" if ignored else "positive"] += 1
                outcome = "ignored" if ignored else "positive"
                box_accounting[f"{'clipped' if _box_was_clipped(clipped_box, left, top, width, height) else 'retained'}_{outcome}"] += 1
                annotation_id += 1
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": clipped_box,
                        "area": clipped_box[2] * clipped_box[3],
                        "iscrowd": ignored,
                        "bbox_vis": gtbox.get("vbox"),
                        "source_tag": gtbox.get("tag"),
                        "source_box_id": gtbox.get("extra", {}).get("box_id", source_box_id),
                        "source_bbox": list(source_box),
                    }
                )

    if not images:
        raise ValueError(f"CrowdHuman annotation file is empty: {annotation_path}")
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "pedestrian"}],
        "metadata": {
            "format": "mot20.rfdetr.coco.v1",
            "conversion_revision": CONVERSION_REVISION,
            "source": "CrowdHuman",
            "split": split,
            "disposition_counts": dict(sorted(disposition_counts.items())),
            "box_accounting": dict(sorted(box_accounting.items())),
        },
    }


def merge_bytetrack_mot20_crowdhuman(
    mot20_train_half: dict[str, Any],
    crowdhuman_train: dict[str, Any],
    crowdhuman_val: dict[str, Any],
) -> dict[str, Any]:
    """Merge sources in ByteTrack's MOT20/CrowdHuman training composition."""
    sources = (
        ("mot20_train_half", "mot20_train", mot20_train_half),
        ("crowdhuman_train", "crowdhuman_train", crowdhuman_train),
        ("crowdhuman_val", "crowdhuman_val", crowdhuman_val),
    )
    _validate_source_manifests(sources)
    _validate_categories(manifest for _, _, manifest in sources)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1
    next_video_id = 1

    for source_name, file_prefix, manifest in sources:
        source_videos = {video["id"]: video for video in manifest.get("videos", [])}
        video_ids: dict[Any, int] = {}
        if source_videos:
            for source_video_id, source_video in sorted(source_videos.items()):
                video_ids[source_video_id] = next_video_id
                videos.append(
                    {
                        "id": next_video_id,
                        "file_name": f"{source_name}/{source_video['file_name']}",
                        "source_video_id": source_video_id,
                    }
                )
                next_video_id += 1
        else:
            video_ids[None] = next_video_id
            videos.append({"id": next_video_id, "file_name": source_name, "source_video_id": None})
            next_video_id += 1

        image_ids: dict[Any, int] = {}
        for source_image in manifest["images"]:
            image = copy.deepcopy(source_image)
            source_image_id = image["id"]
            image_ids[source_image_id] = next_image_id
            image["id"] = next_image_id
            image["source_manifest_image_id"] = source_image_id
            image["file_name"] = f"{file_prefix}/{image['file_name']}"
            source_video_id = image.get("video_id")
            image["video_id"] = video_ids[source_video_id] if source_video_id in video_ids else video_ids[None]
            images.append(image)
            next_image_id += 1

        for source_annotation in manifest["annotations"]:
            annotation = copy.deepcopy(source_annotation)
            source_annotation_id = annotation["id"]
            source_image_id = annotation["image_id"]
            if source_image_id not in image_ids:
                raise ValueError(f"annotation {source_annotation_id} references unknown image {source_image_id}")
            annotation["id"] = next_annotation_id
            annotation["image_id"] = image_ids[source_image_id]
            annotation["source_manifest_annotation_id"] = source_annotation_id
            annotations.append(annotation)
            next_annotation_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "videos": videos,
        "categories": [{"id": 1, "name": "pedestrian"}],
        "metadata": {
            "format": "mot20.rfdetr.coco.v1",
            "conversion_revision": CONVERSION_REVISION,
            "source": "ByteTrack MOT20/CrowdHuman mixture",
            "sources": [source_name for source_name, _, _ in sources],
        },
    }


def merge_byte65_test_adapted_overlay(
    clean_train_manifest: dict[str, Any],
    byte65_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Add explicitly human-audited Byte65 labels to a test-adapted train mix."""
    _validate_categories((clean_train_manifest, byte65_manifest))
    clean_metadata = clean_train_manifest.get("metadata", {})
    byte65_metadata = byte65_manifest.get("metadata", {})
    if clean_metadata.get("source") != "ByteTrack MOT20/CrowdHuman mixture":
        raise ValueError("expected a ByteTrack MOT20/CrowdHuman clean training manifest")
    if byte65_metadata.get("source") != "Byte65" or byte65_metadata.get("split") != "test_adapted_overlay":
        raise ValueError("expected a Byte65 test_adapted_overlay manifest")
    if byte65_metadata.get("human_audit") != "exhaustive":
        raise ValueError("Byte65 overlay requires an exhaustive human_audit record")

    images = copy.deepcopy(clean_train_manifest["images"])
    annotations = copy.deepcopy(clean_train_manifest["annotations"])
    videos = copy.deepcopy(clean_train_manifest.get("videos", []))
    image_id_map: dict[Any, int] = {}
    next_image_id = max((image["id"] for image in images), default=0) + 1
    next_annotation_id = max((annotation["id"] for annotation in annotations), default=0) + 1
    next_video_id = max((video["id"] for video in videos), default=0) + 1
    byte65_video_ids: dict[Any, int] = {}
    for source_image in byte65_manifest["images"]:
        image = copy.deepcopy(source_image)
        source_image_id = image["id"]
        image_id_map[source_image_id] = next_image_id
        image["id"] = next_image_id
        image["source_manifest_image_id"] = source_image_id
        image["file_name"] = f"byte65/{image['file_name']}"
        source_video_id = image.get("video_id", image.get("source_sequence"))
        if source_video_id not in byte65_video_ids:
            byte65_video_ids[source_video_id] = next_video_id
            videos.append(
                {
                    "id": next_video_id,
                    "file_name": f"byte65/{image.get('source_sequence', source_video_id)}",
                    "source_video_id": source_video_id,
                }
            )
            next_video_id += 1
        image["video_id"] = byte65_video_ids[source_video_id]
        images.append(image)
        next_image_id += 1
    for source_annotation in byte65_manifest["annotations"]:
        annotation = copy.deepcopy(source_annotation)
        source_image_id = annotation["image_id"]
        if source_image_id not in image_id_map:
            raise ValueError(f"Byte65 annotation {annotation['id']} references unknown image {source_image_id}")
        annotation["id"] = next_annotation_id
        annotation["image_id"] = image_id_map[source_image_id]
        annotation["source_manifest_annotation_id"] = source_annotation["id"]
        annotations.append(annotation)
        next_annotation_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "videos": videos,
        "categories": [{"id": 1, "name": "pedestrian"}],
        "metadata": {
            "format": "mot20.rfdetr.coco.v1",
            "conversion_revision": CONVERSION_REVISION,
            "source": "ByteTrack MOT20/CrowdHuman/Byte65 mixture",
            "classification": "local_test_adapted",
            "includes_mot20_test_derived_labels": True,
            "sources": [*clean_metadata.get("sources", []), "byte65_human_audited"],
            "byte65_human_audit": "exhaustive",
        },
    }


def assemble_rfdetr_coco_dataset(
    dataset_root: Path,
    train_manifest: dict[str, Any],
    val_manifest: dict[str, Any],
    mot20_root: Path,
    crowdhuman_train_root: Path,
    crowdhuman_val_root: Path,
    extra_train_image_roots: dict[str, Path] | None = None,
) -> None:
    """Build RF-DETR's COCO directory layout with linked source image roots."""
    dataset_root = Path(dataset_root)
    if dataset_root.exists():
        raise FileExistsError(f"refusing to overwrite existing RF-DETR dataset: {dataset_root}")
    _validate_categories((train_manifest, val_manifest))
    val_metadata = val_manifest.get("metadata", {})
    if val_metadata.get("source") != "MOT20" or val_metadata.get("split") != "val_half":
        raise ValueError("expected MOT20 val_half manifest for RF-DETR validation")
    source_roots = {
        "mot20_train": Path(mot20_root),
        "crowdhuman_train": Path(crowdhuman_train_root),
        "crowdhuman_val": Path(crowdhuman_val_root),
    }
    for name, source_root in (extra_train_image_roots or {}).items():
        if not name or Path(name).name != name or name in source_roots:
            raise ValueError(f"invalid additional RF-DETR train image root name: {name!r}")
        source_roots[name] = Path(source_root)
    if not all(path.is_dir() for path in source_roots.values()):
        raise ValueError("all RF-DETR source image roots must be directories")

    train_root = dataset_root / "train"
    valid_root = dataset_root / "valid"
    train_root.mkdir(parents=True)
    valid_root.mkdir()
    for name, source_root in source_roots.items():
        _link_directory(train_root / name, source_root)
    _link_directory(valid_root / "mot20_val", source_roots["mot20_train"])

    assembled_val_manifest = copy.deepcopy(val_manifest)
    for image in assembled_val_manifest["images"]:
        image["file_name"] = f"mot20_val/{image['file_name']}"
    assembled_val_manifest["metadata"] = {
        **assembled_val_manifest["metadata"],
        "assembled_image_root": "mot20_val",
    }
    write_coco_manifest(train_manifest, train_root / "_annotations.coco.json")
    write_coco_manifest(assembled_val_manifest, valid_root / "_annotations.coco.json")


def write_coco_manifest(manifest: dict[str, Any], destination: Path) -> str:
    """Write canonical COCO JSON once and return its SHA-256 digest."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing COCO manifest: {destination}")
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def _read_mot20_sequence(sequence_root: Path) -> dict[str, Any]:
    config_path = sequence_root / "seqinfo.ini"
    parser = configparser.ConfigParser()
    if not parser.read(config_path) or "Sequence" not in parser:
        raise ValueError(f"invalid MOT20 sequence metadata: {config_path}")
    values = parser["Sequence"]
    try:
        name = values["name"]
        image_dir = sequence_root / values["imDir"]
        length = int(values["seqLength"])
        width = int(values["imWidth"])
        height = int(values["imHeight"])
        extension = values["imExt"]
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid MOT20 sequence metadata: {config_path}") from error
    if name != sequence_root.name or extension.lower() != ".jpg" or length < 1 or width < 1 or height < 1:
        raise ValueError(f"unsupported MOT20 sequence contract: {config_path}")
    return {
        "name": name,
        "image_dir": image_dir,
        "length": length,
        "width": width,
        "height": height,
        "gt_path": sequence_root / "gt" / "gt.txt",
    }


def _mot20_half_bounds(length: int, split: MOT20_SPLIT) -> tuple[int, int]:
    split_frame = length // 2 + 1
    return (1, split_frame) if split == "train_half" else (split_frame + 1, length)


def _read_mot20_rows(gt_path: Path) -> list[tuple[int, int, float, float, float, float, int, int, float]]:
    if not gt_path.is_file():
        raise ValueError(f"missing MOT20 ground truth: {gt_path}")
    rows: list[tuple[int, int, float, float, float, float, int, int, float]] = []
    with gt_path.open(newline="", encoding="utf-8") as stream:
        for row_number, columns in enumerate(csv.reader(stream), start=1):
            if not columns:
                continue
            if len(columns) < 9:
                raise ValueError(f"MOT20 row {row_number} has fewer than nine fields: {gt_path}")
            try:
                rows.append(
                    (
                        int(columns[0]),
                        int(columns[1]),
                        float(columns[2]),
                        float(columns[3]),
                        float(columns[4]),
                        float(columns[5]),
                        int(float(columns[6])),
                        int(float(columns[7])),
                        float(columns[8]),
                    )
                )
            except ValueError as error:
                raise ValueError(f"invalid MOT20 row {row_number}: {gt_path}") from error
    return rows


def _mot20_disposition(confidence: int, class_id: int) -> str:
    if confidence == 1 and class_id == 1:
        return "positive_pedestrian"
    if class_id in MOT20_IGNORED_PERSON_CLASSES:
        return "ignored_person"
    if class_id in MOT20_NON_PERSON_CLASSES:
        return "excluded"
    return "excluded"


def _clip_box(left: float, top: float, width: float, height: float, image_width: int, image_height: int) -> list[float] | None:
    values = (left, top, width, height)
    if not all(math.isfinite(value) for value in values):
        return None
    right = left + width
    bottom = top + height
    clipped_left = min(max(left, 0.0), float(image_width))
    clipped_top = min(max(top, 0.0), float(image_height))
    clipped_right = min(max(right, 0.0), float(image_width))
    clipped_bottom = min(max(bottom, 0.0), float(image_height))
    clipped_width = clipped_right - clipped_left
    clipped_height = clipped_bottom - clipped_top
    if clipped_width <= 0 or clipped_height <= 0:
        return None
    return [clipped_left, clipped_top, clipped_width, clipped_height]


def _box_was_clipped(box: list[float], left: float, top: float, width: float, height: float) -> bool:
    return box != [left, top, width, height]


def _image_size(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        return image.size


def _validate_categories(manifests: Any) -> None:
    expected = [{"id": 1, "name": "pedestrian"}]
    for manifest in manifests:
        if manifest.get("categories") != expected:
            raise ValueError("all source manifests must contain only COCO category 1: pedestrian")


def _validate_source_manifests(sources: tuple[tuple[str, str, dict[str, Any]], ...]) -> None:
    expected_sources = {
        "mot20_train_half": ("MOT20", "train_half"),
        "crowdhuman_train": ("CrowdHuman", "train"),
        "crowdhuman_val": ("CrowdHuman", "val"),
    }
    for source_name, _, manifest in sources:
        expected_source, expected_split = expected_sources[source_name]
        metadata = manifest.get("metadata", {})
        if metadata.get("source") != expected_source or metadata.get("split") != expected_split:
            raise ValueError(f"expected {expected_source} {expected_split} manifest for {source_name}")


def _link_directory(link_path: Path, target_path: Path) -> None:
    link_path.symlink_to(os.path.relpath(target_path.resolve(), link_path.parent.resolve()), target_is_directory=True)