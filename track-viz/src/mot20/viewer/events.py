from __future__ import annotations

import math
from itertools import pairwise
from typing import Literal

from fastapi import APIRouter, Query, Request

from mot20.viewer.api import (
    ApiModel,
    DiagnosticResponse,
    ObservationResponse,
    _diagnostic_response,
    _observation_response,
    _require_source,
    require_track_capability,
)
from mot20.viewer.context import _edge_distance
from mot20.viewer.contracts import Diagnostic, Observation
from mot20.viewer.loaders import LoadedSource, SourceRegistry
from mot20.viewer.tracks import _require_track

DEFAULT_DISPLACEMENT_THRESHOLD = 0.5
DEFAULT_SCALE_CHANGE_THRESHOLD = 0.5
DEFAULT_CLOSE_INTERACTION_THRESHOLD = 0.25
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
MAX_NORMALIZED_THRESHOLD = 100.0


class EventSettingsResponse(ApiModel):
    displacement_enabled: bool
    displacement_threshold: float
    displacement_operator: Literal["greater_than_or_equal"] = "greater_than_or_equal"
    scale_change_enabled: bool
    scale_change_threshold: float
    scale_change_operator: Literal["greater_than_or_equal"] = "greater_than_or_equal"
    close_interaction_enabled: bool
    close_interaction_threshold: float
    close_interaction_operator: Literal["less_than_or_equal"] = "less_than_or_equal"


class ConfidenceCapabilityResponse(ApiModel):
    status: Literal["meaningful", "absent", "constant", "sentinel"]
    meaningful: bool
    score_semantics: Literal["tracker_score", "not_defined"]
    threshold: float
    threshold_operator: Literal["less_than_or_equal"] = "less_than_or_equal"
    diagnostic: DiagnosticResponse | None


class DisplacementEventResponse(ApiModel):
    from_frame: int
    to_frame: int
    from_row_index: int
    to_row_index: int
    frame_delta: int
    center_displacement_pixels: float
    normalization_box_height: float
    normalized_displacement: float
    threshold: float


class ScaleChangeEventResponse(ApiModel):
    from_frame: int
    to_frame: int
    from_row_index: int
    to_row_index: int
    frame_delta: int
    absolute_height_change_pixels: float
    normalization_box_height: float
    normalized_scale_change: float
    threshold: float


class CloseInteractionEventResponse(ApiModel):
    frame: int
    focal_row_index: int
    competitor_track_id: int
    competitor_row_index: int
    edge_distance_pixels: float
    focal_box_height: float
    normalized_edge_proximity: float
    threshold: float


class TimelineEventsResponse(ApiModel):
    source_key: str
    sequence: str
    source_hash: str
    track_id: int
    geometry_basis: str = "raw_xywh"
    settings: EventSettingsResponse
    confidence: ConfidenceCapabilityResponse
    displacement_events: tuple[DisplacementEventResponse, ...]
    scale_change_events: tuple[ScaleChangeEventResponse, ...]
    close_interaction_events: tuple[CloseInteractionEventResponse, ...]
    low_confidence_observations: tuple[ObservationResponse, ...]


event_router = APIRouter()


@event_router.get(
    "/api/sequences/{source_key}/tracks/{track_id}/events",
    response_model=TimelineEventsResponse,
)
def track_events(
    request: Request,
    source_key: str,
    track_id: int,
    source_hash: str | None = None,
    enable_displacement: bool = False,
    displacement_threshold: float = Query(
        default=DEFAULT_DISPLACEMENT_THRESHOLD,
        ge=0.0,
        le=MAX_NORMALIZED_THRESHOLD,
    ),
    enable_scale_change: bool = False,
    scale_change_threshold: float = Query(
        default=DEFAULT_SCALE_CHANGE_THRESHOLD,
        ge=0.0,
        le=MAX_NORMALIZED_THRESHOLD,
    ),
    enable_close_interaction: bool = False,
    close_interaction_threshold: float = Query(
        default=DEFAULT_CLOSE_INTERACTION_THRESHOLD,
        ge=0.0,
        le=MAX_NORMALIZED_THRESHOLD,
    ),
    confidence_threshold: float = Query(
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        allow_inf_nan=False,
    ),
) -> TimelineEventsResponse:
    registry: SourceRegistry = request.app.state.registry
    source = _require_source(registry, source_key, source_hash)
    require_track_capability(source)
    observations = tuple(
        sorted(_require_track(source, track_id), key=lambda item: (item.frame, item.row_index))
    )
    confidence = _confidence_capability(source, confidence_threshold)
    low_confidence = (
        tuple(
            _observation_response(observation)
            for observation in observations
            if observation.score is not None and observation.score <= confidence_threshold
        )
        if confidence.meaningful
        else ()
    )
    return TimelineEventsResponse(
        source_key=source.config.key,
        sequence=source.sequence.name,
        source_hash=source.source_hash,
        track_id=track_id,
        settings=EventSettingsResponse(
            displacement_enabled=enable_displacement,
            displacement_threshold=displacement_threshold,
            scale_change_enabled=enable_scale_change,
            scale_change_threshold=scale_change_threshold,
            close_interaction_enabled=enable_close_interaction,
            close_interaction_threshold=close_interaction_threshold,
        ),
        confidence=confidence,
        displacement_events=(
            _displacement_events(observations, displacement_threshold)
            if enable_displacement
            else ()
        ),
        scale_change_events=(
            _scale_change_events(observations, scale_change_threshold)
            if enable_scale_change
            else ()
        ),
        close_interaction_events=(
            _close_interaction_events(source, observations, track_id, close_interaction_threshold)
            if enable_close_interaction
            else ()
        ),
        low_confidence_observations=low_confidence,
    )


def _confidence_capability(
    source: LoadedSource,
    threshold: float,
) -> ConfidenceCapabilityResponse:
    if source.config.adapter != "mot_result_10":
        return _confidence_response(
            source,
            threshold,
            status="absent",
            code="confidence_not_defined",
            message="ground-truth visibility is not tracker confidence",
        )
    scores = tuple(row.score for row in source.source_rows if row.score is not None)
    if not scores:
        return _confidence_response(
            source,
            threshold,
            status="absent",
            code="confidence_not_defined",
            message="source has no tracker score values",
        )
    if any(score == -1.0 for score in scores):
        return _confidence_response(
            source,
            threshold,
            status="sentinel",
            code="sentinel_confidence",
            message="tracker score column contains sentinel values",
        )
    if len(set(scores)) == 1:
        return _confidence_response(
            source,
            threshold,
            status="constant",
            code="constant_confidence",
            message="tracker score column is constant and not discriminative",
        )
    return ConfidenceCapabilityResponse(
        status="meaningful",
        meaningful=True,
        score_semantics="tracker_score",
        threshold=threshold,
        diagnostic=None,
    )


def _confidence_response(
    source: LoadedSource,
    threshold: float,
    *,
    status: Literal["absent", "constant", "sentinel"],
    code: str,
    message: str,
) -> ConfidenceCapabilityResponse:
    diagnostic = Diagnostic(code=code, message=message, source_key=source.config.key)
    return ConfidenceCapabilityResponse(
        status=status,
        meaningful=False,
        score_semantics="tracker_score" if source.config.adapter == "mot_result_10" else "not_defined",
        threshold=threshold,
        diagnostic=_diagnostic_response(diagnostic),
    )


def _displacement_events(
    observations: tuple[Observation, ...],
    threshold: float,
) -> tuple[DisplacementEventResponse, ...]:
    events: list[DisplacementEventResponse] = []
    for previous, current in pairwise(observations):
        frame_delta = current.frame - previous.frame
        if frame_delta <= 0:
            continue
        previous_center = _box_center(previous.raw_xywh)
        current_center = _box_center(current.raw_xywh)
        displacement = math.dist(previous_center, current_center)
        normalization_height = previous.raw_xywh[3]
        normalized = displacement / (normalization_height * frame_delta)
        if normalized >= threshold:
            events.append(
                DisplacementEventResponse(
                    from_frame=previous.frame,
                    to_frame=current.frame,
                    from_row_index=previous.row_index,
                    to_row_index=current.row_index,
                    frame_delta=frame_delta,
                    center_displacement_pixels=displacement,
                    normalization_box_height=normalization_height,
                    normalized_displacement=normalized,
                    threshold=threshold,
                )
            )
    return tuple(events)


def _scale_change_events(
    observations: tuple[Observation, ...],
    threshold: float,
) -> tuple[ScaleChangeEventResponse, ...]:
    events: list[ScaleChangeEventResponse] = []
    for previous, current in pairwise(observations):
        frame_delta = current.frame - previous.frame
        if frame_delta <= 0:
            continue
        previous_height = previous.raw_xywh[3]
        height_change = abs(current.raw_xywh[3] - previous_height)
        normalized = height_change / previous_height
        if normalized >= threshold:
            events.append(
                ScaleChangeEventResponse(
                    from_frame=previous.frame,
                    to_frame=current.frame,
                    from_row_index=previous.row_index,
                    to_row_index=current.row_index,
                    frame_delta=frame_delta,
                    absolute_height_change_pixels=height_change,
                    normalization_box_height=previous_height,
                    normalized_scale_change=normalized,
                    threshold=threshold,
                )
            )
    return tuple(events)


def _close_interaction_events(
    source: LoadedSource,
    observations: tuple[Observation, ...],
    focal_track_id: int,
    threshold: float,
) -> tuple[CloseInteractionEventResponse, ...]:
    events: list[CloseInteractionEventResponse] = []
    for focal in observations:
        focal_height = focal.raw_xywh[3]
        for competitor in source.indexes.frames[focal.frame]:
            competitor_track_id = competitor.usable_track_id
            if competitor_track_id is None or competitor_track_id == focal_track_id:
                continue
            edge_distance = _edge_distance(focal.raw_xywh, competitor.raw_xywh)
            normalized = edge_distance / focal_height
            if normalized <= threshold:
                events.append(
                    CloseInteractionEventResponse(
                        frame=focal.frame,
                        focal_row_index=focal.row_index,
                        competitor_track_id=competitor_track_id,
                        competitor_row_index=competitor.row_index,
                        edge_distance_pixels=edge_distance,
                        focal_box_height=focal_height,
                        normalized_edge_proximity=normalized,
                        threshold=threshold,
                    )
                )
    return tuple(
        sorted(
            events,
            key=lambda item: (
                item.frame,
                item.focal_row_index,
                item.competitor_track_id,
                item.competitor_row_index,
            ),
        )
    )


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    left, top, width, height = box
    return left + width / 2.0, top + height / 2.0