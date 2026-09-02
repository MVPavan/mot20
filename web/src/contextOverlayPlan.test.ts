import { describe, expect, it } from "vitest";

import type { ContextCompetitor, Observation } from "./api";
import { buildContextOverlayPlan, stabilizeContextCompetitors } from "./contextOverlayPlan";

function competitor(
  trackId: number,
  rank: number,
  geometry: [number, number, number, number] = [0, 0, 10, 20],
): ContextCompetitor {
  return {
    rank,
    track_id: trackId,
    best_iou: 0,
    best_normalized_edge_proximity: rank / 10,
    comparison_count: 1,
    evidence: [{
      frame: 12,
      focal_row_index: 1,
      competitor_row_index: trackId,
      focal_raw_xywh: [100, 100, 100, 200],
      competitor_raw_xywh: geometry,
      iou: 0,
      edge_distance_pixels: 0,
      focal_box_height: 200,
      normalized_edge_proximity: 0,
    }],
  };
}

const focal = {
  frame: 12,
  display_geometry: { x1: 100, y1: 100, x2: 200, y2: 300 },
} as Observation;

describe("context overlay planning", () => {
  it("retains visible competitors across rank changes and caps the requested count", () => {
    const ranked = [competitor(4, 1), competitor(3, 2), competitor(2, 3), competitor(5, 4)];

    expect(stabilizeContextCompetitors([2, 3, 9], ranked, 3).map((item) => item.track_id)).toEqual([
      2,
      3,
      4,
    ]);
    expect(stabilizeContextCompetitors([], ranked, 99)).toHaveLength(4);
    expect(stabilizeContextCompetitors([2], ranked, 0)).toEqual([]);
  });

  it("keeps dense context ink subordinate and outside the focal label", () => {
    const competitors = Array.from({ length: 8 }, (_, index) =>
      competitor(index + 2, index + 1, [92 + index * 8, 72 + index * 5, 80, 220]),
    );
    const labelRect = { x1: 100, y1: 74, x2: 150, y2: 100 };

    const plan = buildContextOverlayPlan(focal, competitors, 12, labelRect);

    expect(plan.contextInkArea).toBeLessThanOrEqual(plan.focalArea * 0.05);
    expect(plan.focalStrokeWidth).toBeGreaterThan(plan.contextStrokeWidth);
    expect(plan.labelIntersectionCount).toBe(0);
    expect(plan.commands).toHaveLength(8);
    plan.commands.flatMap((command) => command.segments).forEach((segment) => {
      const intersectsLabel =
        Math.max(Math.min(segment.from.x, segment.to.x), labelRect.x1) <=
          Math.min(Math.max(segment.from.x, segment.to.x), labelRect.x2) &&
        Math.max(Math.min(segment.from.y, segment.to.y), labelRect.y1) <=
          Math.min(Math.max(segment.from.y, segment.to.y), labelRect.y2);
      expect(intersectsLabel).toBe(false);
    });
  });
});