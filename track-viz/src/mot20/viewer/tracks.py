from __future__ import annotations

from itertools import pairwise

from fastapi import APIRouter, Query, Request

from mot20.viewer.api import (
    ApiModel,
    ErrorDetail,
    ObservationResponse,
    ViewerApiError,
    _observation_response,
    _require_source,
    require_track_capability,
)
from mot20.viewer.contracts import Observation
from mot20.viewer.loaders import LoadedSource, SourceRegistry


class TrackGapResponse(ApiModel):
    start_frame: int
    end_frame: int
    length: int


class TrackEvidenceResponse(ApiModel):
    source_key: str
    sequence: str
    source_hash: str
    track_id: int
    observation_frames: tuple[int, ...]
    gaps: tuple[TrackGapResponse, ...]
    first_observation: ObservationResponse
    last_observation: ObservationResponse
    previous_observation: ObservationResponse | None
    next_observation: ObservationResponse | None
    observations: tuple[ObservationResponse, ...]


track_router = APIRouter()


@track_router.get(
    "/api/sequences/{source_key}/tracks/{track_id}",
    response_model=TrackEvidenceResponse,
)
def track_evidence(
    request: Request,
    source_key: str,
    track_id: int,
    source_hash: str | None = None,
    current_row_index: int | None = Query(default=None, ge=1),
) -> TrackEvidenceResponse:
    registry: SourceRegistry = request.app.state.registry
    source = _require_source(registry, source_key, source_hash)
    require_track_capability(source)
    observations = _require_track(source, track_id)
    ordered = tuple(sorted(observations, key=lambda observation: (observation.frame, observation.row_index)))
    frames = tuple(observation.frame for observation in ordered)
    unique_frames = tuple(dict.fromkeys(frames))
    try:
        previous, following = _navigation_observations(ordered, current_row_index)
    except ValueError as error:
        raise ViewerApiError(
            404,
            ErrorDetail(
                code="current_observation_not_found",
                message=(
                    f"row {current_row_index} is not an observation in track {track_id} "
                    f"for source {source.config.key!r}"
                ),
                source_key=source.config.key,
            ),
        ) from error
    return TrackEvidenceResponse(
        source_key=source.config.key,
        sequence=source.sequence.name,
        source_hash=source.source_hash,
        track_id=track_id,
        observation_frames=frames,
        gaps=_track_gaps(unique_frames),
        first_observation=_observation_response(ordered[0]),
        last_observation=_observation_response(ordered[-1]),
        previous_observation=None if previous is None else _observation_response(previous),
        next_observation=None if following is None else _observation_response(following),
        observations=tuple(_observation_response(observation) for observation in ordered),
    )


@track_router.get(
    "/api/sequences/{source_key}/tracks",
    response_model=TrackEvidenceResponse,
)
def search_track(
    request: Request,
    source_key: str,
    track_id: int = Query(ge=1),
    source_hash: str | None = None,
) -> TrackEvidenceResponse:
    return track_evidence(
        request=request,
        source_key=source_key,
        track_id=track_id,
        source_hash=source_hash,
        current_row_index=None,
    )


def _require_track(source: LoadedSource, track_id: int) -> tuple[Observation, ...]:
    observations = source.indexes.tracks.get(track_id)
    if observations is None:
        raise ViewerApiError(
            404,
            ErrorDetail(
                code="track_not_found",
                message=f"track {track_id} is not present in source {source.config.key!r}",
                source_key=source.config.key,
            ),
        )
    return observations


def _track_gaps(frames: tuple[int, ...]) -> tuple[TrackGapResponse, ...]:
    return tuple(
        TrackGapResponse(
            start_frame=first + 1,
            end_frame=second - 1,
            length=second - first - 1,
        )
        for first, second in pairwise(frames)
        if second > first + 1
    )


def _navigation_observations(
    observations: tuple[Observation, ...],
    current_row_index: int | None,
) -> tuple[Observation | None, Observation | None]:
    if current_row_index is None:
        return None, None
    current_position = next(
        (
            position
            for position, observation in enumerate(observations)
            if observation.row_index == current_row_index
        ),
        None,
    )
    if current_position is None:
        raise ValueError(f"row {current_row_index} is not an observation in the track")
    previous = observations[current_position - 1] if current_position > 0 else None
    following = observations[current_position + 1] if current_position + 1 < len(observations) else None
    return previous, following