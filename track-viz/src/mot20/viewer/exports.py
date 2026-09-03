from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
from typing import Literal, cast

import numpy as np
import supervision as sv
from fastapi import APIRouter, Request
from numpy.typing import NDArray
from PIL import Image

from mot20.viewer.api import (
    ApiModel,
    ErrorDetail,
    ViewerApiError,
    _require_frame,
    _require_source,
    require_track_capability,
)
from mot20.viewer.colors import COLOR_CONTRACT_VERSION, TrackColor, track_color
from mot20.viewer.context import CONTEXT_HARD_CAP, rank_context
from mot20.viewer.contracts import Observation
from mot20.viewer.loaders import LoadedSource, SourceRegistry
from mot20.viewer.supervision_adapter import observations_to_detections
from mot20.viewer.tracks import _require_track

INTERACTIVE_FRAME_CAP = 300
OFFLINE_DEFAULT_FRAME_CAP = 3_000
OFFLINE_HARD_FRAME_CAP = 100_000
EXPORT_SCHEMA_VERSION = "mot20-viewer-export-v1"
VIDEO_FILENAME = "track.mp4"
METADATA_FILENAME = "metadata.json"
type ExportKind = Literal["focused_clip", "offline_track_video"]
type ExportStatus = Literal["created", "existing"]
type ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class ExportParameters:
    track_id: int
    start_frame: int
    end_frame: int
    context_count: int = 3
    trace_length: int = 30


@dataclass(frozen=True)
class ExportArtifact:
    export_id: str
    status: ExportStatus
    artifact_directory: Path
    video_path: Path
    metadata_path: Path


class ExportValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ExportArtifactCollisionError(RuntimeError):
    """Raised when a hash-keyed path contains an incomplete or different artifact."""


class FocusedExportRequest(ApiModel):
    source_hash: str | None = None
    track_id: int
    start_frame: int
    end_frame: int
    context_count: int = 3
    trace_length: int = 30


class ExportResponse(ApiModel):
    export_id: str
    status: ExportStatus
    artifact_directory: str
    video_path: str
    metadata_path: str


class ExportRenderer:
    def __init__(
        self,
        *,
        sequence: str,
        focal_track_id: int,
        context_track_ids: Sequence[int],
        trace_length: int,
    ) -> None:
        self.sequence = sequence
        self.focal_track_id = focal_track_id
        self.context_track_ids = tuple(context_track_ids)
        focal_color = _supervision_color(track_color(sequence, focal_track_id))
        self.trace_annotator = sv.TraceAnnotator(
            color=focal_color,
            position=sv.Position.CENTER,
            trace_length=trace_length,
            thickness=2,
            smooth=False,
            color_lookup=sv.ColorLookup.TRACK,
        )
        self._focal_box_annotator = sv.BoxAnnotator(
            color=focal_color,
            thickness=4,
            color_lookup=sv.ColorLookup.TRACK,
        )
        self._label_annotator = sv.LabelAnnotator(
            color=focal_color,
            color_lookup=sv.ColorLookup.TRACK,
            text_color=sv.Color(17, 23, 25),
            text_scale=0.5,
            text_thickness=1,
            text_padding=6,
            text_position=sv.Position.TOP_LEFT,
            smart_position=True,
        )
        self._context_annotators = {
            track_id: sv.BoxCornerAnnotator(
                color=_supervision_color(track_color(sequence, track_id)),
                thickness=2,
                corner_length=12,
                color_lookup=sv.ColorLookup.TRACK,
            )
            for track_id in self.context_track_ids
        }

    def annotate(
        self,
        scene: ImageArray,
        observations: Sequence[Observation],
    ) -> ImageArray:
        rendered = scene.copy()
        focal = tuple(
            observation
            for observation in observations
            if observation.usable_track_id == self.focal_track_id
        )
        focal_detections = observations_to_detections(focal)
        rendered = cast(ImageArray, self.trace_annotator.annotate(rendered, focal_detections))
        for track_id, annotator in self._context_annotators.items():
            context = tuple(
                observation
                for observation in observations
                if observation.usable_track_id == track_id
            )
            if context:
                rendered = cast(
                    ImageArray,
                    annotator.annotate(rendered, observations_to_detections(context)),
                )
        if focal:
            rendered = cast(
                ImageArray,
                self._focal_box_annotator.annotate(rendered, focal_detections),
            )
            rendered = cast(
                ImageArray,
                self._label_annotator.annotate(
                    rendered,
                    focal_detections,
                    labels=[f"ID {self.focal_track_id}"] * len(focal),
                ),
            )
        return rendered


export_router = APIRouter()


@export_router.post(
    "/api/sequences/{source_key}/exports",
    response_model=ExportResponse,
)
def create_focused_export(
    request: Request,
    source_key: str,
    export_request: FocusedExportRequest,
) -> ExportResponse:
    _require_export_origin(request, source_key)
    if export_request.source_hash is None:
        raise ViewerApiError(
            428,
            ErrorDetail(
                code="missing_source_hash",
                message="focused export requires the current source hash",
                source_key=source_key,
            ),
        )
    registry: SourceRegistry = request.app.state.registry
    source = _require_source(registry, source_key, export_request.source_hash)
    require_track_capability(source)
    _require_track(source, export_request.track_id)
    _validate_interactive_range(source_key, export_request.start_frame, export_request.end_frame)
    _require_frame(source, export_request.start_frame)
    _require_frame(source, export_request.end_frame)
    parameters = ExportParameters(
        track_id=export_request.track_id,
        start_frame=export_request.start_frame,
        end_frame=export_request.end_frame,
        context_count=export_request.context_count,
        trace_length=export_request.trace_length,
    )
    try:
        artifact = write_track_video(
            source=source,
            repository_root=request.app.state.repository_root,
            parameters=parameters,
            kind="focused_clip",
            frame_limit=INTERACTIVE_FRAME_CAP,
        )
    except ExportValidationError as error:
        raise ViewerApiError(
            412 if error.code == "source_result_changed" else 422,
            ErrorDetail(code=error.code, message=str(error), source_key=source_key),
        ) from error
    except ExportArtifactCollisionError as error:
        raise ViewerApiError(
            409,
            ErrorDetail(code="export_artifact_collision", message=str(error), source_key=source_key),
        ) from error
    except (OSError, RuntimeError, ValueError) as error:
        raise ViewerApiError(
            500,
            ErrorDetail(
                code="export_failed",
                message="focused export failed without publishing an artifact",
                source_key=source_key,
            ),
        ) from error
    return _export_response(artifact, request.app.state.repository_root)


def write_track_video(
    *,
    source: LoadedSource,
    repository_root: Path,
    parameters: ExportParameters,
    kind: ExportKind,
    frame_limit: int,
    frame_reader: Callable[[Path], bytes] = Path.read_bytes,
) -> ExportArtifact:
    root = Path(repository_root).resolve(strict=True)
    _validate_annotation_hash(root, source)
    frames = _validate_export_parameters(source, parameters, frame_limit)
    frame_inputs = tuple(_frame_input(root, source, frame, frame_reader) for frame in frames)
    context_track_ids = tuple(
        competitor.track_id
        for competitor in rank_context(
            source,
            _require_track(source, parameters.track_id),
            parameters.track_id,
            parameters.start_frame,
            parameters.end_frame,
        )[: parameters.context_count]
    )
    identity = _export_identity(source, parameters, kind, frame_inputs, context_track_ids)
    export_id = hashlib.sha256(_canonical_json(identity)).hexdigest()
    export_root = root / "track-viz" / "artifacts" / "exports"
    export_root.mkdir(parents=True, exist_ok=True)
    final_directory = export_root / export_id
    if final_directory.exists():
        return _existing_artifact(final_directory, export_id)
    lock_path = export_root / f".lock-{export_id}"
    try:
        lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ExportArtifactCollisionError(
            f"export {export_id} is already being published; refusing concurrent overwrite"
        ) from error
    os.close(lock_descriptor)
    temporary_directory: Path | None = None
    try:
        if final_directory.exists():
            return _existing_artifact(final_directory, export_id)
        temporary_directory = Path(tempfile.mkdtemp(prefix=".tmp-", dir=export_root))
        temporary_video = temporary_directory / VIDEO_FILENAME
        renderer = ExportRenderer(
            sequence=source.sequence.name,
            focal_track_id=parameters.track_id,
            context_track_ids=context_track_ids,
            trace_length=parameters.trace_length,
        )
        video_info = sv.VideoInfo(
            width=source.sequence.width,
            height=source.sequence.height,
            fps=source.sequence.frame_rate,
            total_frames=len(frames),
        )
        with sv.VideoSink(str(temporary_video), video_info, codec="mp4v") as video_sink:
            for frame_input in frame_inputs:
                frame_bytes = frame_reader(frame_input.path)
                if hashlib.sha256(frame_bytes).hexdigest() != frame_input.sha256:
                    raise ExportValidationError(
                        "source_frame_changed",
                        f"source frame {frame_input.frame} changed during export",
                    )
                scene = _decode_bgr_frame(frame_bytes, source)
                rendered = renderer.annotate(scene, source.indexes.frames[frame_input.frame])
                video_sink.write_frame(rendered)
        if not temporary_video.is_file() or temporary_video.stat().st_size == 0:
            raise RuntimeError("video encoder produced no output")
        output_sha256 = _sha256_file(temporary_video)
        metadata = _export_metadata(
            export_id=export_id,
            source=source,
            parameters=parameters,
            kind=kind,
            frame_inputs=frame_inputs,
            context_track_ids=context_track_ids,
            output_sha256=output_sha256,
        )
        temporary_metadata = temporary_directory / METADATA_FILENAME
        with temporary_metadata.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        with temporary_video.open("rb") as stream:
            os.fsync(stream.fileno())
        try:
            temporary_directory.rename(final_directory)
        except FileExistsError:
            shutil.rmtree(temporary_directory, ignore_errors=True)
            return _existing_artifact(final_directory, export_id)
    except BaseException:
        if temporary_directory is not None:
            shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    finally:
        lock_path.unlink(missing_ok=True)
    return ExportArtifact(
        export_id=export_id,
        status="created",
        artifact_directory=final_directory,
        video_path=final_directory / VIDEO_FILENAME,
        metadata_path=final_directory / METADATA_FILENAME,
    )


@dataclass(frozen=True)
class _FrameInput:
    frame: int
    image_name: str
    path: Path
    sha256: str


def _validate_export_parameters(
    source: LoadedSource,
    parameters: ExportParameters,
    frame_limit: int,
) -> tuple[int, ...]:
    if frame_limit < 1:
        raise ExportValidationError("invalid_export_frame_limit", "export frame limit must be positive")
    if parameters.track_id < 1 or parameters.track_id not in source.indexes.tracks:
        raise ExportValidationError("track_not_found", f"track {parameters.track_id} is not available")
    if parameters.start_frame < 1 or parameters.end_frame < parameters.start_frame:
        raise ExportValidationError(
            "invalid_export_frame_range",
            "export frames must be an ordered one-based inclusive range",
        )
    if parameters.end_frame > source.sequence.length:
        raise ExportValidationError(
            "invalid_export_frame_range",
            f"export frame range exceeds sequence length {source.sequence.length}",
        )
    frames = tuple(range(parameters.start_frame, parameters.end_frame + 1))
    if len(frames) > frame_limit:
        raise ExportValidationError(
            "export_frame_limit_exceeded",
            f"export contains {len(frames)} frames but its configured limit is {frame_limit}",
        )
    if not 0 <= parameters.context_count <= CONTEXT_HARD_CAP:
        raise ExportValidationError(
            "invalid_export_context_count",
            f"context count must be between 0 and {CONTEXT_HARD_CAP}",
        )
    if not 1 <= parameters.trace_length <= INTERACTIVE_FRAME_CAP:
        raise ExportValidationError(
            "invalid_export_trace_length",
            f"trace length must be between 1 and {INTERACTIVE_FRAME_CAP}",
        )
    if not any(
        parameters.start_frame <= observation.frame <= parameters.end_frame
        for observation in source.indexes.tracks[parameters.track_id]
    ):
        raise ExportValidationError(
            "track_not_in_export_range",
            f"track {parameters.track_id} has no observation in the export range",
        )
    return frames


def _frame_input(
    repository_root: Path,
    source: LoadedSource,
    frame: int,
    frame_reader: Callable[[Path], bytes],
) -> _FrameInput:
    image_name = source.sequence.image_names[frame - 1]
    path = (repository_root / source.config.images / image_name).resolve(strict=True)
    if not path.is_relative_to(repository_root):
        raise ExportValidationError("unsafe_source_frame", f"source frame {frame} escapes the repository")
    content = frame_reader(path)
    return _FrameInput(
        frame=frame,
        image_name=image_name,
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _decode_bgr_frame(content: bytes, source: LoadedSource) -> ImageArray:
    with Image.open(BytesIO(content)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape != (source.sequence.height, source.sequence.width, 3):
        raise ExportValidationError("source_frame_changed", "source frame dimensions changed during export")
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _export_identity(
    source: LoadedSource,
    parameters: ExportParameters,
    kind: ExportKind,
    frame_inputs: Sequence[_FrameInput],
    context_track_ids: Sequence[int],
) -> dict[str, object]:
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "kind": kind,
        "source_key": source.config.key,
        "sequence": source.sequence.name,
        "annotation_result_sha256": source.source_hash,
        "frames": [{"frame": item.frame, "image_sha256": item.sha256} for item in frame_inputs],
        "parameters": asdict(parameters),
        "context_track_ids": list(context_track_ids),
        "color_contract_version": COLOR_CONTRACT_VERSION,
        "supervision_version": sv.__version__,
        "tool_version": _tool_version(),
    }


def _export_metadata(
    *,
    export_id: str,
    source: LoadedSource,
    parameters: ExportParameters,
    kind: ExportKind,
    frame_inputs: Sequence[_FrameInput],
    context_track_ids: Sequence[int],
    output_sha256: str,
) -> dict[str, object]:
    focal_color = track_color(source.sequence.name, parameters.track_id)
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "export_id": export_id,
        "kind": kind,
        "tool": {
            "name": "mot20-viewer",
            "version": _tool_version(),
            "supervision_version": sv.__version__,
            "color_contract_version": COLOR_CONTRACT_VERSION,
        },
        "source": {
            "key": source.config.key,
            "sequence": source.sequence.name,
            "adapter": source.config.adapter,
            "annotation_result_sha256": source.source_hash,
            "frames": [
                {
                    "frame": item.frame,
                    "image_name": item.image_name,
                    "image_sha256": item.sha256,
                    "observation_row_hashes": [
                        observation.row_hash
                        for observation in source.indexes.frames[item.frame]
                    ],
                }
                for item in frame_inputs
            ],
        },
        "selection": {
            "track_id": parameters.track_id,
            "start_frame": parameters.start_frame,
            "end_frame": parameters.end_frame,
        },
        "render": {
            "width": source.sequence.width,
            "height": source.sequence.height,
            "fps": source.sequence.frame_rate,
            "frame_count": len(frame_inputs),
            "geometry_basis": "display_clamped_xyxy",
            "context_count": parameters.context_count,
            "context_track_ids": list(context_track_ids),
            "trace_length": parameters.trace_length,
            "focal_color": focal_color.model_dump(mode="json"),
            "context_colors": [
                track_color(source.sequence.name, track_id).model_dump(mode="json")
                for track_id in context_track_ids
            ],
            "annotators": {
                "focal": "BoxAnnotator",
                "context": "BoxCornerAnnotator",
                "selected_label": "LabelAnnotator(smart_position=True)",
                "trace": "TraceAnnotator",
                "trace_color_lookup": sv.ColorLookup.TRACK.value,
            },
        },
        "incoming_provenance": asdict(source.config.provenance),
        "policy_classification": _policy_classification(source),
        "output": {
            "filename": VIDEO_FILENAME,
            "container": "mp4",
            "codec": "mp4v",
            "sha256": output_sha256,
        },
    }


def _existing_artifact(directory: Path, export_id: str) -> ExportArtifact:
    video_path = directory / VIDEO_FILENAME
    metadata_path = directory / METADATA_FILENAME
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        recorded_hash = metadata["output"]["sha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ExportArtifactCollisionError(
            f"export path {export_id} exists without valid metadata; refusing to overwrite"
        ) from error
    if (
        metadata.get("export_id") != export_id
        or not video_path.is_file()
        or not isinstance(recorded_hash, str)
        or _sha256_file(video_path) != recorded_hash
    ):
        raise ExportArtifactCollisionError(
            f"export path {export_id} contains a different artifact; refusing to overwrite"
        )
    return ExportArtifact(
        export_id=export_id,
        status="existing",
        artifact_directory=directory,
        video_path=video_path,
        metadata_path=metadata_path,
    )


def _export_response(artifact: ExportArtifact, repository_root: Path) -> ExportResponse:
    return ExportResponse(
        export_id=artifact.export_id,
        status=artifact.status,
        artifact_directory=artifact.artifact_directory.relative_to(repository_root).as_posix(),
        video_path=artifact.video_path.relative_to(repository_root).as_posix(),
        metadata_path=artifact.metadata_path.relative_to(repository_root).as_posix(),
    )


def _require_export_origin(request: Request, source_key: str) -> None:
    configured_origin: str | None = request.app.state.application_origin
    incoming_origin = request.headers.get("origin")
    if configured_origin is None or incoming_origin != configured_origin:
        raise ViewerApiError(
            403,
            ErrorDetail(
                code="invalid_export_origin",
                message="focused export requires the configured same-origin Origin",
                source_key=source_key,
            ),
        )


def _validate_interactive_range(source_key: str, start_frame: int, end_frame: int) -> None:
    if start_frame < 1 or end_frame < start_frame:
        raise ViewerApiError(
            422,
            ErrorDetail(
                code="invalid_export_frame_range",
                message="export frames must be an ordered one-based inclusive range",
                source_key=source_key,
            ),
        )
    frame_count = end_frame - start_frame + 1
    if frame_count > INTERACTIVE_FRAME_CAP:
        raise ViewerApiError(
            413,
            ErrorDetail(
                code="interactive_export_frame_cap_exceeded",
                message=(
                    f"interactive exports are limited to {INTERACTIVE_FRAME_CAP} frames; "
                    "use track-viz/scripts/export_track_video.py for a bounded offline export"
                ),
                source_key=source_key,
            ),
        )


def _policy_classification(source: LoadedSource) -> str:
    if source.config.adapter == "mot_gt_9":
        return "ground_truth_training_source"
    if source.config.sequence in {"MOT20-04", "MOT20-06", "MOT20-07", "MOT20-08"}:
        return "local_test_adapted_development_material"
    return "local_development_material"


def _validate_annotation_hash(repository_root: Path, source: LoadedSource) -> None:
    annotation_path = (repository_root / source.config.annotations).resolve(strict=True)
    if not annotation_path.is_relative_to(repository_root):
        raise ExportValidationError(
            "unsafe_source_result",
            "configured annotation result escapes the repository",
        )
    if _sha256_file(annotation_path) != source.source_hash:
        raise ExportValidationError(
            "source_result_changed",
            "annotation result changed after startup; reload the viewer and use its current hash",
        )


def _supervision_color(color: TrackColor) -> sv.Color:
    return sv.Color(*color.rgb)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool_version() -> str:
    try:
        return version("mot20-viewer")
    except PackageNotFoundError:
        return "unknown"