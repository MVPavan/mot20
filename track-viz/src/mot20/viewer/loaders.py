from __future__ import annotations

import configparser
import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from mot20.viewer.config import SourceConfig, ViewerConfig, provenance_diagnostics, resolve_source_paths
from mot20.viewer.contracts import (
    Capability,
    ContractError,
    Diagnostic,
    IdStatus,
    Observation,
    Sequence,
    parse_gt_row,
    parse_result_row,
)
from mot20.viewer.indexes import SequenceIndexes, build_indexes


class SourceError(ValueError):
    """Raised when a present source violates its declared contract."""


@dataclass(frozen=True)
class LoadedSource:
    config: SourceConfig
    sequence: Sequence
    source_hash: str
    source_rows: tuple[Observation, ...]
    observations: tuple[Observation, ...]
    capability: Capability
    indexes: SequenceIndexes


@dataclass(frozen=True)
class UnavailableSource:
    config: SourceConfig
    diagnostic: Diagnostic


@dataclass(frozen=True)
class SourceRegistry:
    sources: tuple[LoadedSource, ...]
    unavailable: tuple[UnavailableSource, ...]
    diagnostics: tuple[Diagnostic, ...]


def load_registry(config: ViewerConfig, repository_root: Path) -> SourceRegistry:
    sources: list[LoadedSource] = []
    unavailable: list[UnavailableSource] = []
    diagnostics = list(provenance_diagnostics(config))
    for source in config.sources:
        paths = resolve_source_paths(source, repository_root)
        missing = tuple(name for name in ("seqinfo", "images", "annotations") if not getattr(paths, name).exists())
        if missing:
            diagnostic = Diagnostic(
                code="source_unavailable",
                message=f"source {source.key} is unavailable because configured paths are absent",
                source_key=source.key,
                fields=missing,
            )
            unavailable.append(UnavailableSource(config=source, diagnostic=diagnostic))
            diagnostics.append(diagnostic)
            continue
        sources.append(load_source(source, repository_root))
    return SourceRegistry(sources=tuple(sources), unavailable=tuple(unavailable), diagnostics=tuple(diagnostics))


def load_source(source: SourceConfig, repository_root: Path) -> LoadedSource:
    paths = resolve_source_paths(source, repository_root)
    if not paths.seqinfo.is_file() or not paths.annotations.is_file() or not paths.images.is_dir():
        raise SourceError(f"source {source.key} is missing a required file or directory")
    sequence = _load_sequence(source, paths.seqinfo, paths.images)
    try:
        annotation_bytes = paths.annotations.read_bytes()
        annotation_text = annotation_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise SourceError(f"cannot read annotations for source {source.key}") from error
    source_hash = hashlib.sha256(annotation_bytes).hexdigest()
    lines = annotation_text.splitlines()
    if any(not line.strip() for line in lines):
        raise SourceError(f"source {source.key} contains a blank annotation row")
    parser = parse_gt_row if source.adapter == "mot_gt_9" else parse_result_row
    try:
        rows = tuple(
            parser(
                line,
                source_key=source.key,
                sequence=source.sequence,
                row_index=row_index,
                source_hash=source_hash,
                image_width=sequence.width,
                image_height=sequence.height,
                sequence_length=sequence.length,
            )
            for row_index, line in enumerate(lines, start=1)
        )
    except ContractError as error:
        raise SourceError(f"invalid annotations for source {source.key}: {error}") from error
    observations = tuple(row for row in rows if row.reviewable)
    return LoadedSource(
        config=source,
        sequence=sequence,
        source_hash=source_hash,
        source_rows=rows,
        observations=observations,
        capability=_derive_capability(source.key, rows),
        indexes=build_indexes(sequence, observations),
    )


def _load_sequence(source: SourceConfig, seqinfo_path: Path, image_root: Path) -> Sequence:
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with seqinfo_path.open(encoding="utf-8") as stream:
            parser.read_file(stream)
        values = parser["Sequence"]
        sequence_name = values["name"]
        configured_image_root = (seqinfo_path.parent / values["imDir"]).resolve(strict=False)
        frame_rate = int(values["frameRate"])
        length = int(values["seqLength"])
        width = int(values["imWidth"])
        height = int(values["imHeight"])
        extension = values["imExt"]
    except (OSError, KeyError, ValueError, configparser.Error) as error:
        raise SourceError(f"invalid sequence metadata for source {source.key}") from error
    if (
        sequence_name != source.sequence
        or configured_image_root != image_root
        or extension != ".jpg"
        or min(frame_rate, length, width, height) < 1
    ):
        raise SourceError(f"unsupported sequence metadata for source {source.key}")
    expected_names = tuple(f"{frame:06d}.jpg" for frame in range(1, length + 1))
    actual_names = tuple(sorted(path.name for path in image_root.glob("*.jpg") if path.is_file()))
    if actual_names != expected_names:
        raise SourceError(f"image count or names do not match seqinfo.ini for source {source.key}")
    for image_name in actual_names:
        image_path = image_root / image_name
        if not image_path.resolve(strict=True).is_relative_to(image_root):
            raise SourceError(f"JPEG path escapes configured image directory: {image_path}")
        try:
            with Image.open(image_path) as image:
                if image.format != "JPEG" or image.size != (width, height):
                    raise SourceError(f"JPEG dimensions or format do not match seqinfo.ini: {image_path}")
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            raise SourceError(f"invalid JPEG for source {source.key}: {image_path}") from error
    return Sequence(
        name=sequence_name,
        length=length,
        width=width,
        height=height,
        frame_rate=frame_rate,
        image_names=actual_names,
    )


def _derive_capability(source_key: str, rows: tuple[Observation, ...]) -> Capability:
    usable_ids = tuple(sorted({row.usable_track_id for row in rows if row.usable_track_id is not None}))
    raw_ids = tuple(row.raw_track_id for row in rows)
    diagnostics: tuple[Diagnostic, ...] = ()
    if usable_ids:
        status: IdStatus = "tracked"
    elif raw_ids and all(track_id == -1 for track_id in raw_ids):
        status = "sentinel_only"
        diagnostics = (
            Diagnostic(
                code="sentinel_only_ids",
                message=f"source {source_key} has no usable positive track IDs",
                source_key=source_key,
            ),
        )
    else:
        status = "unusable"
        diagnostics = (
            Diagnostic(
                code="unusable_ids",
                message=f"source {source_key} has no usable track ID column values",
                source_key=source_key,
            ),
        )
    return Capability(
        id_status=status,
        track_features=bool(usable_ids),
        usable_track_ids=usable_ids,
        diagnostics=diagnostics,
    )