from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

type IdStatus = Literal["tracked", "sentinel_only", "unusable"]


class ContractError(ValueError):
    """Raised when source data violates the normalized viewer contract."""


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    source_key: str | None = None
    fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class Sequence:
    name: str
    length: int
    width: int
    height: int
    frame_rate: int
    image_names: tuple[str, ...]


@dataclass(frozen=True)
class Capability:
    id_status: IdStatus
    track_features: bool
    usable_track_ids: tuple[int, ...]
    diagnostics: tuple[Diagnostic, ...]


@dataclass(frozen=True)
class Observation:
    source_key: str
    sequence: str
    frame: int
    row_index: int
    row_hash: str
    source_hash: str
    raw_row: str
    raw_fields: tuple[str, ...]
    raw_track_id: int
    usable_track_id: int | None
    raw_xywh: tuple[float, float, float, float]
    display_box: tuple[float, float, float, float]
    score: float | None
    mark: int | None
    class_id: int | None
    visibility: float | None
    opaque_result_fields: tuple[float, float, float] | None
    reviewable: bool


def parse_gt_row(
    raw_row: str,
    *,
    source_key: str,
    sequence: str,
    row_index: int,
    source_hash: str,
    image_width: int,
    image_height: int,
    sequence_length: int,
) -> Observation:
    fields = _split_exact_fields(raw_row, expected=9, row_index=row_index)
    frame = _parse_integer(fields[0], "frame", row_index)
    track_id = _parse_integer(fields[1], "track ID", row_index)
    left, top, width, height = (
        _parse_finite_float(value, name, row_index)
        for value, name in zip(fields[2:6], ("left", "top", "width", "height"), strict=True)
    )
    mark = _parse_integer(fields[6], "mark", row_index)
    class_id = _parse_integer(fields[7], "class", row_index)
    visibility = _parse_finite_float(fields[8], "visibility", row_index)
    display_box = _validate_geometry(
        left,
        top,
        width,
        height,
        frame=frame,
        row_index=row_index,
        sequence_length=sequence_length,
        image_width=image_width,
        image_height=image_height,
    )
    return Observation(
        source_key=source_key,
        sequence=sequence,
        frame=frame,
        row_index=row_index,
        row_hash=hashlib.sha256(raw_row.encode("utf-8")).hexdigest(),
        source_hash=source_hash,
        raw_row=raw_row,
        raw_fields=fields,
        raw_track_id=track_id,
        usable_track_id=track_id if track_id > 0 else None,
        raw_xywh=(left, top, width, height),
        display_box=display_box,
        score=None,
        mark=mark,
        class_id=class_id,
        visibility=visibility,
        opaque_result_fields=None,
        reviewable=mark == 1 and class_id == 1,
    )


def parse_result_row(
    raw_row: str,
    *,
    source_key: str,
    sequence: str,
    row_index: int,
    source_hash: str,
    image_width: int,
    image_height: int,
    sequence_length: int,
) -> Observation:
    fields = _split_exact_fields(raw_row, expected=10, row_index=row_index)
    frame = _parse_integer(fields[0], "frame", row_index)
    track_id = _parse_integer(fields[1], "track ID", row_index)
    left, top, width, height = (
        _parse_finite_float(value, name, row_index)
        for value, name in zip(fields[2:6], ("left", "top", "width", "height"), strict=True)
    )
    score = _parse_finite_float(fields[6], "score", row_index)
    opaque_fields = (
        _parse_finite_float(fields[7], "opaque field 8", row_index),
        _parse_finite_float(fields[8], "opaque field 9", row_index),
        _parse_finite_float(fields[9], "opaque field 10", row_index),
    )
    display_box = _validate_geometry(
        left,
        top,
        width,
        height,
        frame=frame,
        row_index=row_index,
        sequence_length=sequence_length,
        image_width=image_width,
        image_height=image_height,
    )
    return Observation(
        source_key=source_key,
        sequence=sequence,
        frame=frame,
        row_index=row_index,
        row_hash=hashlib.sha256(raw_row.encode("utf-8")).hexdigest(),
        source_hash=source_hash,
        raw_row=raw_row,
        raw_fields=fields,
        raw_track_id=track_id,
        usable_track_id=track_id if track_id > 0 else None,
        raw_xywh=(left, top, width, height),
        display_box=display_box,
        score=score,
        mark=None,
        class_id=None,
        visibility=None,
        opaque_result_fields=opaque_fields,
        reviewable=True,
    )


def _split_exact_fields(raw_row: str, *, expected: int, row_index: int) -> tuple[str, ...]:
    fields = tuple(field.strip() for field in raw_row.split(","))
    if len(fields) != expected:
        raise ContractError(f"row {row_index} must contain exactly {expected} fields; found {len(fields)}")
    return fields


def _parse_integer(value: str, name: str, row_index: int) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ContractError(f"row {row_index} has invalid integer {name}: {value!r}") from error
    return parsed


def _parse_finite_float(value: str, name: str, row_index: int) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ContractError(f"row {row_index} has invalid numeric {name}: {value!r}") from error
    if not math.isfinite(parsed):
        raise ContractError(f"row {row_index} has non-finite {name}: {value!r}")
    return parsed


def _validate_geometry(
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    frame: int,
    row_index: int,
    sequence_length: int,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    if not 1 <= frame <= sequence_length:
        raise ContractError(f"row {row_index} frame {frame} is outside 1..{sequence_length}")
    if width <= 0 or height <= 0:
        raise ContractError(f"row {row_index} width and height must be positive")
    right = left + width
    bottom = top + height
    if right <= 0 or bottom <= 0 or left >= image_width or top >= image_height:
        raise ContractError(f"row {row_index} box has no image intersection")
    return (
        max(0.0, left),
        max(0.0, top),
        min(float(image_width), right),
        min(float(image_height), bottom),
    )