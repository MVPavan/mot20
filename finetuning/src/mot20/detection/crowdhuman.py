"""Safe extraction and identity verification for CrowdHuman source archives."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


def extract_crowdhuman_sources(
    train_archives: list[Path],
    val_archive: Path,
    train_annotations: Path,
    val_annotations: Path,
    destination: Path,
    expected_train_images: int,
    expected_val_images: int,
) -> dict[str, Any]:
    """Extract verified CrowdHuman splits into a new immutable destination."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite CrowdHuman extraction: {destination}")
    train_archives = [Path(archive) for archive in train_archives]
    val_archive = Path(val_archive)
    train_annotations = Path(train_annotations)
    val_annotations = Path(val_annotations)
    if not train_archives:
        raise ValueError("CrowdHuman extraction requires at least one training archive")
    if expected_train_images < 1 or expected_val_images < 1:
        raise ValueError("expected CrowdHuman image counts must be positive")
    for path in [*train_archives, val_archive, train_annotations, val_annotations]:
        if not path.is_file():
            raise ValueError(f"CrowdHuman source is not a file: {path}")
    train_members = _archive_members(train_archives)
    val_members = _archive_members([val_archive])
    if len(train_members) != expected_train_images or len(val_members) != expected_val_images:
        raise ValueError(
            f"unexpected CrowdHuman archive image counts: train={len(train_members)}, val={len(val_members)}"
        )
    staging = destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    _extract_members(train_members, staging / "train" / "Images")
    _extract_members(val_members, staging / "val" / "Images")
    _verify_odgt_images(train_annotations, staging / "train" / "Images", expected_train_images)
    _verify_odgt_images(val_annotations, staging / "val" / "Images", expected_val_images)
    receipt = {
        "format": "mot20.crowdhuman.extraction.v1",
        "image_counts": {"train": len(train_members), "val": len(val_members)},
        "source_sha256": {
            "train_archives": {archive.name: _sha256_file(archive) for archive in train_archives},
            "val_archive": {val_archive.name: _sha256_file(val_archive)},
            "train_annotations": _sha256_file(train_annotations),
            "val_annotations": _sha256_file(val_annotations),
        },
    }
    (staging / "extraction.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.rename(staging, destination)
    return receipt


def _archive_members(archives: list[Path]) -> list[tuple[Path, str]]:
    members: list[tuple[Path, str]] = []
    image_names: set[str] = set()
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                relative_name = _image_member_name(info.filename, archive_path)
                if relative_name in image_names:
                    raise ValueError(f"duplicate CrowdHuman image member: {relative_name}")
                image_names.add(relative_name)
                members.append((archive_path, info.filename))
    return members


def _image_member_name(member_name: str, archive_path: Path) -> str:
    member = PurePosixPath(member_name)
    if member.is_absolute() or ".." in member.parts or len(member.parts) != 2:
        raise ValueError(f"unsafe CrowdHuman archive member: {archive_path}!{member_name}")
    if member.parts[0] != "Images" or member.suffix.lower() != ".jpg":
        raise ValueError(f"unexpected CrowdHuman archive member: {archive_path}!{member_name}")
    return member.name


def _extract_members(members: list[tuple[Path, str]], image_root: Path) -> None:
    image_root.mkdir(parents=True)
    for archive_path, member_name in members:
        destination = image_root / _image_member_name(member_name, archive_path)
        with zipfile.ZipFile(archive_path) as archive, archive.open(member_name) as source, destination.open("xb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)


def _verify_odgt_images(annotation_path: Path, image_root: Path, expected_count: int) -> None:
    image_ids: set[str] = set()
    with annotation_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                image_id = str(json.loads(line)["ID"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"invalid CrowdHuman ODGT record at line {line_number}: {annotation_path}") from error
            if image_id in image_ids:
                raise ValueError(f"duplicate CrowdHuman ODGT image ID: {image_id}")
            if not (image_root / f"{image_id}.jpg").is_file():
                raise ValueError(f"CrowdHuman ODGT image does not resolve: {image_id}")
            image_ids.add(image_id)
    if len(image_ids) != expected_count:
        raise ValueError(f"unexpected CrowdHuman ODGT image count: {len(image_ids)}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()