from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import stat
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Query, Request
from PIL import Image, UnidentifiedImageError

from mot20.viewer.api import (
    ApiModel,
    DisplayGeometryResponse,
    ErrorDetail,
    RawGeometryResponse,
    ViewerApiError,
    _enumerated_frame_path,
    _require_source,
)
from mot20.viewer.contracts import Observation
from mot20.viewer.loaders import LoadedSource, SourceRegistry

MIN_CROP_SIZE = 32
MAX_CROP_SIZE = 1024
MAX_CROP_PADDING = 128


class CropGeometryResponse(ApiModel):
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int
    padding: int
    max_size: int
    output_width: int
    output_height: int


class CropResponse(ApiModel):
    source_key: str
    sequence: str
    source_hash: str
    source_image_hash: str
    frame: int
    row_index: int
    row_hash: str
    raw_geometry: RawGeometryResponse
    display_geometry: DisplayGeometryResponse
    crop_geometry: CropGeometryResponse
    cache_key: str
    cache_status: Literal["created", "existing"]
    media_type: Literal["image/jpeg"] = "image/jpeg"
    image_base64: str


crop_router = APIRouter()


@crop_router.get(
    "/api/sequences/{source_key}/observations/{row_index}/crop",
    response_model=CropResponse,
)
def observation_crop(
    request: Request,
    source_key: str,
    row_index: int,
    source_hash: str,
    padding: int = Query(default=16, ge=0, le=MAX_CROP_PADDING),
    max_size: int = Query(default=256, ge=MIN_CROP_SIZE, le=MAX_CROP_SIZE),
) -> CropResponse:
    registry: SourceRegistry = request.app.state.registry
    source = _require_source(registry, source_key, source_hash)
    observation = _require_source_row(source, row_index)
    frame_path = _enumerated_frame_path(request.app.state.repository_root, source, observation.frame)
    try:
        image_bytes = frame_path.read_bytes()
    except OSError as error:
        raise ViewerApiError(
            409,
            ErrorDetail(
                code="frame_unavailable",
                message=f"enumerated frame {observation.frame} is no longer readable",
                source_key=source.config.key,
                frame=observation.frame,
            ),
        ) from error
    source_image_hash = hashlib.sha256(image_bytes).hexdigest()
    crop_bytes, crop_geometry = _render_crop(
        image_bytes,
        observation,
        padding=padding,
        max_size=max_size,
    )
    cache_key = _crop_cache_key(
        source,
        observation,
        source_image_hash=source_image_hash,
        padding=padding,
        max_size=max_size,
    )
    cache_root = _safe_cache_root(request.app.state.repository_root)
    cache_path = cache_root / cache_key[:2] / f"{cache_key}.jpg"
    created = _write_once(cache_root, cache_path, crop_bytes)
    cached_bytes = _read_cache(cache_root, cache_path)
    left, top, width, height = observation.raw_xywh
    x1, y1, x2, y2 = observation.display_box
    return CropResponse(
        source_key=source.config.key,
        sequence=source.sequence.name,
        source_hash=source.source_hash,
        source_image_hash=source_image_hash,
        frame=observation.frame,
        row_index=observation.row_index,
        row_hash=observation.row_hash,
        raw_geometry=RawGeometryResponse(x=left, y=top, width=width, height=height),
        display_geometry=DisplayGeometryResponse(x1=x1, y1=y1, x2=x2, y2=y2),
        crop_geometry=crop_geometry,
        cache_key=cache_key,
        cache_status="created" if created else "existing",
        image_base64=base64.b64encode(cached_bytes).decode("ascii"),
    )


def _require_source_row(source: LoadedSource, row_index: int) -> Observation:
    observation = next(
        (candidate for candidate in source.source_rows if candidate.row_index == row_index),
        None,
    )
    if observation is None:
        raise ViewerApiError(
            404,
            ErrorDetail(
                code="observation_not_found",
                message=f"row {row_index} is not present in source {source.config.key!r}",
                source_key=source.config.key,
            ),
        )
    return observation


def _render_crop(
    image_bytes: bytes,
    observation: Observation,
    *,
    padding: int,
    max_size: int,
) -> tuple[bytes, CropGeometryResponse]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as source_image:
            source_image.load()
            image = source_image.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise ViewerApiError(
            409,
            ErrorDetail(
                code="frame_unavailable",
                message=f"enumerated frame {observation.frame} is not a readable JPEG",
                source_key=observation.source_key,
                frame=observation.frame,
            ),
        ) from error
    x1, y1, x2, y2 = observation.display_box
    crop_x1 = max(0, math.floor(x1) - padding)
    crop_y1 = max(0, math.floor(y1) - padding)
    crop_x2 = min(image.width, math.ceil(x2) + padding)
    crop_y2 = min(image.height, math.ceil(y2) + padding)
    cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    cropped.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    cropped.save(output, format="JPEG", quality=90, optimize=True)
    return output.getvalue(), CropGeometryResponse(
        x1=crop_x1,
        y1=crop_y1,
        x2=crop_x2,
        y2=crop_y2,
        width=crop_x2 - crop_x1,
        height=crop_y2 - crop_y1,
        padding=padding,
        max_size=max_size,
        output_width=cropped.width,
        output_height=cropped.height,
    )


def _crop_cache_key(
    source: LoadedSource,
    observation: Observation,
    *,
    source_image_hash: str,
    padding: int,
    max_size: int,
) -> str:
    identity = {
        "version": 1,
        "source_key": source.config.key,
        "sequence": source.sequence.name,
        "source_hash": source.source_hash,
        "source_image_hash": source_image_hash,
        "frame": observation.frame,
        "row_index": observation.row_index,
        "row_hash": observation.row_hash,
        "raw_xywh": observation.raw_xywh,
        "padding": padding,
        "max_size": max_size,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_cache_root(repository_root: Path) -> Path:
    root = Path(repository_root).resolve(strict=True)
    cache_root = (root / "track-viz" / "artifacts" / "cache").resolve(strict=False)
    if not cache_root.is_relative_to(root):
        raise ViewerApiError(
            409,
            ErrorDetail(
                code="unsafe_cache_root",
                message="viewer cache root escapes the repository",
            ),
        )
    cache_root.mkdir(parents=True, exist_ok=True)
    resolved = cache_root.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ViewerApiError(
            409,
            ErrorDetail(
                code="unsafe_cache_root",
                message="viewer cache root escapes the repository",
            ),
        )
    return resolved


def _write_once(cache_root: Path, destination: Path, content: bytes) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = destination.parent.resolve(strict=True)
    if not resolved_parent.is_relative_to(cache_root):
        raise ViewerApiError(
            409,
            ErrorDetail(code="unsafe_cache_path", message="crop cache path escapes its root"),
        )
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=".crop-", dir=resolved_parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            return False
        return True
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_cache(cache_root: Path, cache_path: Path) -> bytes:
    resolved_parent = cache_path.parent.resolve(strict=True)
    if not resolved_parent.is_relative_to(cache_root):
        raise ViewerApiError(
            409,
            ErrorDetail(code="unsafe_cache_path", message="crop cache path escapes its root"),
        )
    try:
        file_descriptor = os.open(cache_path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise ViewerApiError(
            409,
            ErrorDetail(code="unsafe_cache_path", message="crop cache entry is not a safe file"),
        ) from error
    with os.fdopen(file_descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ViewerApiError(
                409,
                ErrorDetail(code="unsafe_cache_path", message="crop cache entry is not a regular file"),
            )
        return stream.read()