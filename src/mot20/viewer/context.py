from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from fastapi import APIRouter, Query, Request

from mot20.viewer.api import ApiModel, _require_frame, _require_source, require_track_capability
from mot20.viewer.contracts import Observation
from mot20.viewer.loaders import LoadedSource, SourceRegistry
from mot20.viewer.tracks import _require_track

CONTEXT_HARD_CAP = 8
CONTEXT_MAX_WINDOW_RADIUS = 64


class ContextWindowResponse(ApiModel):
    center_frame: int
    start_frame: int
    end_frame: int
    radius: int


class ContextPairEvidenceResponse(ApiModel):
    frame: int
    focal_row_index: int
    competitor_row_index: int
    focal_raw_xywh: tuple[float, float, float, float]
    competitor_raw_xywh: tuple[float, float, float, float]
    iou: float
    edge_distance_pixels: float
    focal_box_height: float
    normalized_edge_proximity: float


class ContextCompetitorResponse(ApiModel):
    rank: int
    track_id: int
    best_iou: float
    best_normalized_edge_proximity: float
    comparison_count: int
    evidence: tuple[ContextPairEvidenceResponse, ...]


class ContextResponse(ApiModel):
    source_key: str
    sequence: str
    source_hash: str
    track_id: int
    geometry_basis: str = "raw_xywh"
    window: ContextWindowResponse
    requested_count: int
    hard_cap: int = CONTEXT_HARD_CAP
    total_competitors: int
    competitors: tuple[ContextCompetitorResponse, ...]


@dataclass(frozen=True)
class ContextPairEvidence:
    frame: int
    focal_row_index: int
    competitor_row_index: int
    focal_raw_xywh: tuple[float, float, float, float]
    competitor_raw_xywh: tuple[float, float, float, float]
    iou: float
    edge_distance_pixels: float
    focal_box_height: float
    normalized_edge_proximity: float


@dataclass(frozen=True)
class RankedCompetitor:
    track_id: int
    best_iou: float
    best_normalized_edge_proximity: float
    evidence: tuple[ContextPairEvidence, ...]


context_router = APIRouter()


@context_router.get(
    "/api/sequences/{source_key}/tracks/{track_id}/context",
    response_model=ContextResponse,
)
def track_context(
    request: Request,
    source_key: str,
    track_id: int,
    frame: int = Query(ge=1),
    source_hash: str | None = None,
    window_radius: int = Query(default=3, ge=0, le=CONTEXT_MAX_WINDOW_RADIUS),
    count: int = Query(default=3, ge=0, le=CONTEXT_HARD_CAP),
) -> ContextResponse:
    registry: SourceRegistry = request.app.state.registry
    source = _require_source(registry, source_key, source_hash)
    require_track_capability(source)
    _require_frame(source, frame)
    focal_observations = _require_track(source, track_id)
    start_frame = max(1, frame - window_radius)
    end_frame = min(source.sequence.length, frame + window_radius)
    ranked = rank_context(source, focal_observations, track_id, start_frame, end_frame)
    selected = ranked[:count]
    return ContextResponse(
        source_key=source.config.key,
        sequence=source.sequence.name,
        source_hash=source.source_hash,
        track_id=track_id,
        window=ContextWindowResponse(
            center_frame=frame,
            start_frame=start_frame,
            end_frame=end_frame,
            radius=window_radius,
        ),
        requested_count=count,
        total_competitors=len(ranked),
        competitors=tuple(
            ContextCompetitorResponse(
                rank=rank,
                track_id=competitor.track_id,
                best_iou=competitor.best_iou,
                best_normalized_edge_proximity=competitor.best_normalized_edge_proximity,
                comparison_count=len(competitor.evidence),
                evidence=tuple(
                    ContextPairEvidenceResponse(**evidence.__dict__)
                    for evidence in competitor.evidence
                ),
            )
            for rank, competitor in enumerate(selected, start=1)
        ),
    )


def rank_context(
    source: LoadedSource,
    focal_observations: tuple[Observation, ...],
    focal_track_id: int,
    start_frame: int,
    end_frame: int,
) -> tuple[RankedCompetitor, ...]:
    evidence_by_track: defaultdict[int, list[ContextPairEvidence]] = defaultdict(list)
    for focal in focal_observations:
        if not start_frame <= focal.frame <= end_frame:
            continue
        for competitor in source.indexes.frames[focal.frame]:
            competitor_track_id = competitor.usable_track_id
            if competitor_track_id is None or competitor_track_id == focal_track_id:
                continue
            iou = _intersection_over_union(focal.raw_xywh, competitor.raw_xywh)
            edge_distance = _edge_distance(focal.raw_xywh, competitor.raw_xywh)
            focal_height = focal.raw_xywh[3]
            evidence_by_track[competitor_track_id].append(
                ContextPairEvidence(
                    frame=focal.frame,
                    focal_row_index=focal.row_index,
                    competitor_row_index=competitor.row_index,
                    focal_raw_xywh=focal.raw_xywh,
                    competitor_raw_xywh=competitor.raw_xywh,
                    iou=iou,
                    edge_distance_pixels=edge_distance,
                    focal_box_height=focal_height,
                    normalized_edge_proximity=edge_distance / focal_height,
                )
            )
    competitors = tuple(
        RankedCompetitor(
            track_id=competitor_track_id,
            best_iou=max(item.iou for item in evidence),
            best_normalized_edge_proximity=min(item.normalized_edge_proximity for item in evidence),
            evidence=tuple(
                sorted(
                    evidence,
                    key=lambda item: (item.frame, item.focal_row_index, item.competitor_row_index),
                )
            ),
        )
        for competitor_track_id, evidence in evidence_by_track.items()
    )
    return tuple(
        sorted(
            competitors,
            key=lambda item: (-item.best_iou, item.best_normalized_edge_proximity, item.track_id),
        )
    )


def _intersection_over_union(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    intersection_width = max(
        0.0,
        min(first_x + first_width, second_x + second_width) - max(first_x, second_x),
    )
    intersection_height = max(
        0.0,
        min(first_y + first_height, second_y + second_height) - max(first_y, second_y),
    )
    intersection_area = intersection_width * intersection_height
    union_area = first_width * first_height + second_width * second_height - intersection_area
    return intersection_area / union_area


def _edge_distance(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    horizontal_gap = max(first_x - (second_x + second_width), second_x - (first_x + first_width), 0.0)
    vertical_gap = max(first_y - (second_y + second_height), second_y - (first_y + first_height), 0.0)
    return math.hypot(horizontal_gap, vertical_gap)