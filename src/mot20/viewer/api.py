from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse, Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from mot20.viewer.config import (
    AdapterKind,
    ConfigError,
    Provenance,
    config_from_paths,
    resolve_source_paths,
)
from mot20.viewer.contracts import Capability, Diagnostic, Observation
from mot20.viewer.loaders import LoadedSource, SourceError, SourceRegistry, load_registry


class ApiModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class DiagnosticResponse(ApiModel):
    code: str
    message: str
    source_key: str | None
    fields: tuple[str, ...]


class ProvenanceResponse(ApiModel):
    producer: str | None
    detector: str | None
    checkpoint: str | None
    tracker: str | None
    post_processing: str | None
    adaptation_iterations: int | None
    notes: str | None


class CapabilityResponse(ApiModel):
    id_status: Literal["tracked", "sentinel_only", "unusable"]
    track_features: bool
    usable_track_ids: tuple[int, ...]
    diagnostics: tuple[DiagnosticResponse, ...]


class SourceMetadataResponse(ApiModel):
    source_key: str
    sequence: str
    frame_numbering: Literal["one_based"] = "one_based"
    frame_count: int
    width: int
    height: int
    frame_rate: int
    source_hash: str
    adapter: AdapterKind
    source_class: Literal["ground_truth", "tracker_result"]
    policy_classification: Literal[
        "ground_truth_training_source",
        "local_test_adapted_development_material",
        "local_development_material",
    ]
    source_row_count: int
    observation_count: int
    capability: CapabilityResponse
    provenance: ProvenanceResponse
    diagnostics: tuple[DiagnosticResponse, ...]


class UnavailableSourceResponse(ApiModel):
    source_key: str
    sequence: str
    diagnostic: DiagnosticResponse


class SequenceListResponse(ApiModel):
    sources: tuple[SourceMetadataResponse, ...]
    unavailable: tuple[UnavailableSourceResponse, ...]
    diagnostics: tuple[DiagnosticResponse, ...]


class SourcePathRequest(ApiModel):
    images: str
    annotations: str


class SourcePathResponse(ApiModel):
    images: str | None
    annotations: str | None


class SourcePathEntryResponse(ApiModel):
    path: str
    entry_type: Literal["directory", "file"]


class SourcePathSuggestionsResponse(ApiModel):
    kind: Literal["images", "annotations"]
    query: str
    directory: str
    parent: str | None
    entries: tuple[SourcePathEntryResponse, ...]
    suggestions: tuple[str, ...]


class HealthResponse(ApiModel):
    status: Literal["ok"] = "ok"
    source_count: int
    unavailable_count: int


class RawGeometryResponse(ApiModel):
    x: float
    y: float
    width: float
    height: float


class DisplayGeometryResponse(ApiModel):
    x1: float
    y1: float
    x2: float
    y2: float


class GroundTruthResponse(ApiModel):
    mark: int
    class_id: int
    visibility: float


class ObservationResponse(ApiModel):
    source_key: str
    sequence: str
    frame: int
    row_index: int
    row_hash: str
    source_hash: str
    raw_track_id: int
    usable_track_id: int | None
    raw_geometry: RawGeometryResponse
    display_geometry: DisplayGeometryResponse
    score: float | None
    ground_truth: GroundTruthResponse | None
    opaque_result_fields: tuple[float, float, float] | None
    score_semantics: Literal["tracker_score", "not_defined"]
    ground_truth_semantics: Literal["mot_mark_class_visibility", "not_defined"]


class FrameObservationsResponse(ApiModel):
    source_key: str
    sequence: str
    frame: int
    frame_numbering: Literal["one_based"] = "one_based"
    source_hash: str
    observations: tuple[ObservationResponse, ...]


class ErrorDetail(ApiModel):
    code: str
    message: str
    source_key: str | None = None
    frame: int | None = None
    expected_source_hash: str | None = None
    actual_source_hash: str | None = None


class ErrorResponse(ApiModel):
    error: ErrorDetail


class ViewerApiError(Exception):
    def __init__(self, status_code: int, detail: ErrorDetail) -> None:
        super().__init__(detail.message)
        self.status_code = status_code
        self.detail = detail


class SpaStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            response = await super().get_response(path, scope)
        except HTTPException as error:
            if error.status_code != 404 or _is_api_path(path):
                raise
        else:
            if response.status_code != 404 or _is_api_path(path):
                return response
        return await super().get_response("index.html", scope)


def create_app(
    *,
    registry: SourceRegistry,
    repository_root: Path,
    trusted_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "testserver"),
    development_origin: str | None = None,
    application_origin: str | None = None,
    extension_routers: Sequence[APIRouter] = (),
    frame_reader: Callable[[Path], bytes] = Path.read_bytes,
) -> FastAPI:
    app = FastAPI(title="MOT20 Viewer", docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(trusted_hosts))
    if development_origin is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[development_origin],
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "If-None-Match"],
        )
    app.state.registry = registry
    app.state.repository_root = Path(repository_root).resolve(strict=True)
    app.state.frame_reader = frame_reader
    app.state.application_origin = application_origin or development_origin
    app.state.source_selection_lock = asyncio.Lock()

    @app.exception_handler(ViewerApiError)
    async def viewer_api_error(_request: Request, error: ViewerApiError) -> JSONResponse:
        response = ErrorResponse(error=error.detail)
        return JSONResponse(status_code=error.status_code, content=response.model_dump(mode="json"))

    @app.get("/api/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        active_registry: SourceRegistry = request.app.state.registry
        return HealthResponse(
            source_count=len(active_registry.sources),
            unavailable_count=len(active_registry.unavailable),
        )

    @app.get("/api/sequences", response_model=SequenceListResponse)
    async def list_sequences(request: Request) -> SequenceListResponse:
        return _registry_response(request.app.state.registry)

    @app.get("/api/source-selection", response_model=SourcePathResponse)
    async def source_selection(request: Request) -> SourcePathResponse:
        active_registry: SourceRegistry = request.app.state.registry
        if not active_registry.sources:
            return SourcePathResponse(images=None, annotations=None)
        paths = resolve_source_paths(
            active_registry.sources[0].config,
            request.app.state.repository_root,
        )
        return SourcePathResponse(images=str(paths.images), annotations=str(paths.annotations))

    @app.post("/api/source-selection", response_model=SequenceListResponse)
    async def replace_source(request: Request, selection: SourcePathRequest) -> SequenceListResponse:
        _require_same_origin(request)
        async with request.app.state.source_selection_lock:
            try:
                config = config_from_paths(Path(selection.images), Path(selection.annotations))
                replacement = await run_in_threadpool(
                    load_registry,
                    config,
                    request.app.state.repository_root,
                )
            except (ConfigError, SourceError) as error:
                raise ViewerApiError(
                    400,
                    ErrorDetail(code="invalid_source_paths", message=str(error)),
                ) from error
            request.app.state.registry = replacement
        return _registry_response(replacement)

    @app.get("/api/source-path-suggestions", response_model=SourcePathSuggestionsResponse)
    async def source_path_suggestions(
        request: Request,
        kind: Literal["images", "annotations"],
        query: str = "",
    ) -> SourcePathSuggestionsResponse:
        directory, parent, entries = await run_in_threadpool(
            _server_path_suggestions,
            request.app.state.repository_root,
            kind,
            query,
        )
        return SourcePathSuggestionsResponse(
            kind=kind,
            query=query,
            directory=directory,
            parent=parent,
            entries=entries,
            suggestions=tuple(entry.path for entry in entries),
        )

    @app.get("/api/sequences/{source_key}", response_model=SourceMetadataResponse)
    async def sequence_detail(
        request: Request,
        source_key: str,
        source_hash: str | None = None,
    ) -> SourceMetadataResponse:
        active_registry: SourceRegistry = request.app.state.registry
        source = _require_source(active_registry, source_key, source_hash)
        return _source_metadata(source, active_registry.diagnostics)

    @app.get(
        "/api/sequences/{source_key}/frames/{frame}/observations",
        response_model=FrameObservationsResponse,
    )
    async def frame_observations(
        request: Request,
        source_key: str,
        frame: int,
        source_hash: str | None = None,
    ) -> FrameObservationsResponse:
        source = _require_source(request.app.state.registry, source_key, source_hash)
        _require_frame(source, frame)
        return FrameObservationsResponse(
            source_key=source.config.key,
            sequence=source.sequence.name,
            frame=frame,
            source_hash=source.source_hash,
            observations=tuple(_observation_response(row) for row in source.indexes.frames[frame]),
        )

    @app.get("/api/sequences/{source_key}/frames/{frame}", response_class=Response)
    def frame_image(
        request: Request,
        source_key: str,
        frame: int,
        source_hash: str | None = None,
    ) -> Response:
        source = _require_source(request.app.state.registry, source_key, source_hash)
        _require_frame(source, frame)
        frame_path = _enumerated_frame_path(app.state.repository_root, source, frame)
        try:
            content = frame_reader(frame_path)
        except OSError as error:
            raise ViewerApiError(
                409,
                ErrorDetail(
                    code="frame_unavailable",
                    message=f"enumerated frame {frame} is no longer readable",
                    source_key=source_key,
                    frame=frame,
                ),
            ) from error
        etag = f'"{hashlib.sha256(content).hexdigest()}"'
        headers = {
            "ETag": etag,
            "Cache-Control": "private, max-age=31536000, immutable",
        }
        if _etag_matches(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers=headers)
        return Response(content=content, media_type="image/jpeg", headers=headers)

    for router in extension_routers:
        app.include_router(router)

    distribution = app.state.repository_root / "web" / "dist"
    if (distribution / "index.html").is_file():
        app.mount("/", SpaStaticFiles(directory=distribution, html=True), name="frontend")

    return app


def _enumerated_frame_path(repository_root: Path, source: LoadedSource, frame: int) -> Path:
    image_name = source.sequence.image_names[frame - 1]
    configured_root = Path(source.config.images)
    image_root = (
        configured_root.resolve(strict=True)
        if configured_root.is_absolute()
        else (repository_root / configured_root).resolve(strict=True)
    )
    try:
        frame_path = (image_root / image_name).resolve(strict=True)
    except OSError as error:
        raise ViewerApiError(
            409,
            ErrorDetail(
                code="frame_unavailable",
                message=f"enumerated frame {frame} is no longer available",
                source_key=source.config.key,
                frame=frame,
            ),
        ) from error
    if not frame_path.is_relative_to(image_root):
        raise ViewerApiError(
            409,
            ErrorDetail(
                code="frame_unavailable",
                message=f"enumerated frame {frame} is outside the configured image directory",
                source_key=source.config.key,
                frame=frame,
            ),
        )
    return frame_path


def _require_same_origin(request: Request) -> None:
    configured_origin: str | None = request.app.state.application_origin
    request_origin = f"{request.url.scheme}://{request.url.netloc}"
    allowed_origins = {request_origin}
    if configured_origin is not None:
        allowed_origins.add(configured_origin.rstrip("/"))
    if request.headers.get("origin", "").rstrip("/") not in allowed_origins:
        raise ViewerApiError(
            403,
            ErrorDetail(
                code="invalid_source_selection_origin",
                message="source selection requires a same-origin request",
            ),
        )


def _server_path_suggestions(
    repository_root: Path,
    kind: Literal["images", "annotations"],
    query: str,
) -> tuple[str, str | None, tuple[SourcePathEntryResponse, ...]]:
    typed = Path(query).expanduser() if query.strip() else Path("/")
    if not typed.is_absolute():
        typed = repository_root / typed
    parent = typed if typed.is_dir() else typed.parent
    prefix = "" if typed.is_dir() else typed.name
    try:
        directory = parent.resolve(strict=True)
        entries = sorted(directory.iterdir(), key=lambda path: path.name)
    except OSError:
        fallback = str(parent.resolve(strict=False))
        return fallback, None, ()
    suggestions: list[SourcePathEntryResponse] = []
    for entry in entries:
        if entry.name.startswith(".") or not entry.name.startswith(prefix):
            continue
        try:
            entry_type: Literal["directory", "file"] | None = (
                "directory"
                if entry.is_dir()
                else "file"
                if kind == "annotations" and entry.is_file()
                else None
            )
        except OSError:
            continue
        if entry_type is not None:
            suggestions.append(
                SourcePathEntryResponse(
                    path=str(entry.resolve(strict=False)),
                    entry_type=entry_type,
                )
            )
        if len(suggestions) == 25:
            break
    resolved_directory = str(directory)
    resolved_parent = None if directory.parent == directory else str(directory.parent)
    return resolved_directory, resolved_parent, tuple(suggestions)


def _etag_matches(if_none_match: str | None, etag: str) -> bool:
    if if_none_match is None:
        return False
    candidates: Sequence[str] = tuple(value.strip() for value in if_none_match.split(","))
    return "*" in candidates or etag in candidates


def _is_api_path(path: str) -> bool:
    normalized = path.lstrip("/")
    return normalized == "api" or normalized.startswith("api/")


def require_track_capability(source: LoadedSource) -> None:
    if not source.capability.track_features:
        raise ViewerApiError(
            409,
            ErrorDetail(
                code="unsupported_track_capability",
                message=f"source {source.config.key!r} has no usable track identities",
                source_key=source.config.key,
            ),
        )


def _require_source(registry: SourceRegistry, source_key: str, source_hash: str | None) -> LoadedSource:
    source = next((candidate for candidate in registry.sources if candidate.config.key == source_key), None)
    if source is None:
        raise ViewerApiError(
            404,
            ErrorDetail(
                code="unknown_source",
                message=f"source {source_key!r} is not available",
                source_key=source_key,
            ),
        )
    if source_hash is not None and source_hash != source.source_hash:
        raise ViewerApiError(
            412,
            ErrorDetail(
                code="stale_source_hash",
                message=f"source {source_key!r} no longer matches the requested hash",
                source_key=source_key,
                expected_source_hash=source_hash,
                actual_source_hash=source.source_hash,
            ),
        )
    return source


def _require_frame(source: LoadedSource, frame: int) -> None:
    if not 1 <= frame <= source.sequence.length:
        raise ViewerApiError(
            404,
            ErrorDetail(
                code="frame_out_of_range",
                message=f"frame {frame} is outside 1..{source.sequence.length}",
                source_key=source.config.key,
                frame=frame,
            ),
        )


def _observation_response(observation: Observation) -> ObservationResponse:
    left, top, width, height = observation.raw_xywh
    x1, y1, x2, y2 = observation.display_box
    ground_truth = None
    if observation.mark is not None and observation.class_id is not None and observation.visibility is not None:
        ground_truth = GroundTruthResponse(
            mark=observation.mark,
            class_id=observation.class_id,
            visibility=observation.visibility,
        )
    return ObservationResponse(
        source_key=observation.source_key,
        sequence=observation.sequence,
        frame=observation.frame,
        row_index=observation.row_index,
        row_hash=observation.row_hash,
        source_hash=observation.source_hash,
        raw_track_id=observation.raw_track_id,
        usable_track_id=observation.usable_track_id,
        raw_geometry=RawGeometryResponse(x=left, y=top, width=width, height=height),
        display_geometry=DisplayGeometryResponse(x1=x1, y1=y1, x2=x2, y2=y2),
        score=observation.score,
        ground_truth=ground_truth,
        opaque_result_fields=observation.opaque_result_fields,
        score_semantics="tracker_score" if observation.score is not None else "not_defined",
        ground_truth_semantics="mot_mark_class_visibility" if ground_truth is not None else "not_defined",
    )


def _registry_response(registry: SourceRegistry) -> SequenceListResponse:
    return SequenceListResponse(
        sources=tuple(_source_metadata(source, registry.diagnostics) for source in registry.sources),
        unavailable=tuple(
            UnavailableSourceResponse(
                source_key=source.config.key,
                sequence=source.config.sequence,
                diagnostic=_diagnostic_response(source.diagnostic),
            )
            for source in registry.unavailable
        ),
        diagnostics=tuple(_diagnostic_response(diagnostic) for diagnostic in registry.diagnostics),
    )


def _source_metadata(
    source: LoadedSource,
    registry_diagnostics: tuple[Diagnostic, ...],
) -> SourceMetadataResponse:
    source_class: Literal["ground_truth", "tracker_result"]
    policy_classification: Literal[
        "ground_truth_training_source",
        "local_test_adapted_development_material",
        "local_development_material",
    ]
    if source.config.adapter == "mot_gt_9":
        source_class = "ground_truth"
        policy_classification = "ground_truth_training_source"
    else:
        source_class = "tracker_result"
        policy_classification = (
            "local_test_adapted_development_material"
            if source.config.sequence in {"MOT20-04", "MOT20-06", "MOT20-07", "MOT20-08"}
            else "local_development_material"
        )
    return SourceMetadataResponse(
        source_key=source.config.key,
        sequence=source.sequence.name,
        frame_count=source.sequence.length,
        width=source.sequence.width,
        height=source.sequence.height,
        frame_rate=source.sequence.frame_rate,
        source_hash=source.source_hash,
        adapter=source.config.adapter,
        source_class=source_class,
        policy_classification=policy_classification,
        source_row_count=len(source.source_rows),
        observation_count=len(source.observations),
        capability=_capability_response(source.capability),
        provenance=_provenance_response(source.config.provenance),
        diagnostics=tuple(
            _diagnostic_response(diagnostic)
            for diagnostic in registry_diagnostics
            if diagnostic.source_key == source.config.key
        ),
    )


def _capability_response(capability: Capability) -> CapabilityResponse:
    return CapabilityResponse(
        id_status=capability.id_status,
        track_features=capability.track_features,
        usable_track_ids=capability.usable_track_ids,
        diagnostics=tuple(_diagnostic_response(diagnostic) for diagnostic in capability.diagnostics),
    )


def _provenance_response(provenance: Provenance) -> ProvenanceResponse:
    return ProvenanceResponse(**vars(provenance))


def _diagnostic_response(diagnostic: Diagnostic) -> DiagnosticResponse:
    return DiagnosticResponse(**vars(diagnostic))