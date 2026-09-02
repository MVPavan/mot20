import type { ContextCompetitor, Observation } from "./api";
import type { Point } from "./viewport";

export const CONTEXT_HARD_CAP = 8;
export const FOCAL_STROKE_WIDTH = 4;
export const CONTEXT_STROKE_WIDTH = 1.25;

export interface Rectangle {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface ContextMarkerSegment {
  from: Point;
  to: Point;
}

export interface ContextMarkerCommand {
  type: "contextMarker";
  trackId: number;
  strokeWidth: number;
  segments: ContextMarkerSegment[];
}

export interface ContextOverlayPlan {
  commands: ContextMarkerCommand[];
  focalArea: number;
  contextInkArea: number;
  labelIntersectionCount: number;
  focalStrokeWidth: number;
  contextStrokeWidth: number;
}

export function stabilizeContextCompetitors(
  previousTrackIds: readonly number[],
  ranked: readonly ContextCompetitor[],
  requestedCount: number,
): ContextCompetitor[] {
  const count = Math.min(CONTEXT_HARD_CAP, Math.max(0, Math.trunc(requestedCount)));
  const byTrackId = new Map(ranked.map((competitor) => [competitor.track_id, competitor]));
  const selected: ContextCompetitor[] = [];

  previousTrackIds.forEach((trackId) => {
    const competitor = byTrackId.get(trackId);
    if (competitor !== undefined && selected.length < count) {
      selected.push(competitor);
    }
  });
  ranked.forEach((competitor) => {
    if (
      selected.length < count &&
      !selected.some((selectedCompetitor) => selectedCompetitor.track_id === competitor.track_id)
    ) {
      selected.push(competitor);
    }
  });
  return selected;
}

function cornerSegments(rectangle: Rectangle, length: number): ContextMarkerSegment[] {
  return [
    { from: { x: rectangle.x1, y: rectangle.y1 }, to: { x: rectangle.x1 + length, y: rectangle.y1 } },
    { from: { x: rectangle.x1, y: rectangle.y1 }, to: { x: rectangle.x1, y: rectangle.y1 + length } },
    { from: { x: rectangle.x2 - length, y: rectangle.y1 }, to: { x: rectangle.x2, y: rectangle.y1 } },
    { from: { x: rectangle.x2, y: rectangle.y1 }, to: { x: rectangle.x2, y: rectangle.y1 + length } },
    { from: { x: rectangle.x1, y: rectangle.y2 }, to: { x: rectangle.x1 + length, y: rectangle.y2 } },
    { from: { x: rectangle.x1, y: rectangle.y2 - length }, to: { x: rectangle.x1, y: rectangle.y2 } },
    { from: { x: rectangle.x2 - length, y: rectangle.y2 }, to: { x: rectangle.x2, y: rectangle.y2 } },
    { from: { x: rectangle.x2, y: rectangle.y2 - length }, to: { x: rectangle.x2, y: rectangle.y2 } },
  ];
}

function segmentIntersectsRectangle(
  segment: ContextMarkerSegment,
  rectangle: Rectangle,
  padding: number,
): boolean {
  return (
    Math.max(Math.min(segment.from.x, segment.to.x), rectangle.x1 - padding) <=
      Math.min(Math.max(segment.from.x, segment.to.x), rectangle.x2 + padding) &&
    Math.max(Math.min(segment.from.y, segment.to.y), rectangle.y1 - padding) <=
      Math.min(Math.max(segment.from.y, segment.to.y), rectangle.y2 + padding)
  );
}

export function buildContextOverlayPlan(
  focal: Observation,
  competitors: readonly ContextCompetitor[],
  frame: number,
  focalLabelRect: Rectangle,
): ContextOverlayPlan {
  const focalGeometry = focal.display_geometry;
  const focalArea = Math.max(0, focalGeometry.x2 - focalGeometry.x1) *
    Math.max(0, focalGeometry.y2 - focalGeometry.y1);
  const evidence = competitors.flatMap((competitor) => {
    const current = competitor.evidence.find((item) => item.frame === frame);
    return current === undefined ? [] : [{ competitor, current }];
  });
  const segmentCount = evidence.length * 8;
  const inkBudget = focalArea * 0.05;
  const budgetedLength = segmentCount === 0
    ? 0
    : inkBudget / (CONTEXT_STROKE_WIDTH * segmentCount);
  let contextInkArea = 0;
  const commands = evidence.map(({ competitor, current }) => {
    const [x, y, width, height] = current.competitor_raw_xywh;
    const rectangle = { x1: x, y1: y, x2: x + width, y2: y + height };
    const desiredLength = Math.max(0, Math.min(width, height) * 0.15);
    const length = Math.min(desiredLength, budgetedLength);
    const segments = cornerSegments(rectangle, length).filter(
      (segment) => !segmentIntersectsRectangle(segment, focalLabelRect, CONTEXT_STROKE_WIDTH / 2),
    );
    contextInkArea += segments.reduce(
      (total, segment) => total + Math.hypot(segment.to.x - segment.from.x, segment.to.y - segment.from.y) * CONTEXT_STROKE_WIDTH,
      0,
    );
    return {
      type: "contextMarker" as const,
      trackId: competitor.track_id,
      strokeWidth: CONTEXT_STROKE_WIDTH,
      segments,
    };
  });

  return {
    commands,
    focalArea,
    contextInkArea,
    labelIntersectionCount: commands.flatMap((command) => command.segments).filter(
      (segment) => segmentIntersectsRectangle(segment, focalLabelRect, CONTEXT_STROKE_WIDTH / 2),
    ).length,
    focalStrokeWidth: FOCAL_STROKE_WIDTH,
    contextStrokeWidth: CONTEXT_STROKE_WIDTH,
  };
}