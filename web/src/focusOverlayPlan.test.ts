import { describe, expect, it } from "vitest";

import type { Observation } from "./api";
import { buildFocusOverlayPlan } from "./focusOverlayPlan";

function observation(frame: number, rowIndex: number): Observation {
  return {
    source_key: "mot20-01",
    sequence: "MOT20-01",
    frame,
    row_index: rowIndex,
    row_hash: `row-${rowIndex}`,
    source_hash: "hash-a",
    raw_track_id: 8,
    usable_track_id: 8,
    raw_geometry: { x: frame * 10, y: 20, width: 30, height: 40 },
    display_geometry: { x1: frame * 10, y1: 20, x2: frame * 10 + 30, y2: 60 },
    score: 0.8,
    ground_truth: null,
    opaque_result_fields: null,
    score_semantics: "tracker_score",
    ground_truth_semantics: "not_defined",
  };
}

describe("Focus overlay plan", () => {
  it("draws no current box at a gap and never includes unrelated observations", () => {
    const focal = [observation(1, 1), observation(3, 3), observation(6, 6)];
    const plan = buildFocusOverlayPlan(focal, 4, 8);

    expect(plan.commands.filter((command) => command.type === "focusBox")).toEqual([]);
    expect(plan.commands).toEqual([
      {
        type: "focusTrace",
        points: [
          { x: 25, y: 40 },
          { x: 45, y: 40 },
        ],
      },
    ]);
  });
});