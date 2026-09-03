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
    expect(plan.commands.filter((command) => command.type === "focusTrace").map((command) => command.frames)).toEqual([[1], [3]]);
  });

  it("keeps past evidence separate from future evidence and breaks every observation gap", () => {
    const plan = buildFocusOverlayPlan(
      [observation(1, 1), observation(2, 2), observation(4, 4), observation(5, 5), observation(7, 7)],
      4,
      { mode: "complete", maximumVertices: 512 },
    );

    const traces = plan.commands.filter((command) => command.type === "focusTrace");
    expect(traces).toHaveLength(4);
    expect(traces.map((trace) => ({ future: trace.future, frames: trace.frames }))).toEqual([
      { future: false, frames: [1, 2] },
      { future: false, frames: [4] },
      { future: true, frames: [5] },
      { future: true, frames: [7] },
    ]);
    expect(plan.commands.find((command) => command.type === "focusBox")).toMatchObject({ observation: { frame: 4 } });
  });

  it("bounds dense trajectories while retaining first, last, current, and gap boundaries", () => {
    const dense = Array.from({ length: 700 }, (_, index) => observation(index + 1, index + 1));
    dense.splice(300, 1);
    const plan = buildFocusOverlayPlan(dense, 400, { mode: "complete", maximumVertices: 512 });
    const frames = plan.commands.filter((command) => command.type === "focusTrace").flatMap((command) => command.frames);

    expect(frames).toHaveLength(512);
    expect(frames).toEqual(expect.arrayContaining([1, 300, 302, 700, 400]));
  });

  it("falls back to unconnected bounded markers when gap boundaries alone exceed the budget", () => {
    const sparse = Array.from({ length: 700 }, (_, index) => observation(index * 2 + 1, index + 1));
    const plan = buildFocusOverlayPlan(sparse, 401, { mode: "complete", maximumVertices: 512 });
    const traces = plan.commands.filter((command) => command.type === "focusTrace");

    expect(plan.simplified).toBe(true);
    expect(traces).toHaveLength(512);
    expect(traces.every((trace) => trace.markersOnly)).toBe(true);
  });

  it("retains global endpoints and an interior current observation in the boundary-overflow fallback", () => {
    const continuous = Array.from({ length: 1_000 }, (_, index) => observation(index + 1, index + 1));
    const gapped = Array.from({ length: 600 }, (_, index) => observation(1_002 + index * 2, 2_000 + index));
    const plan = buildFocusOverlayPlan([...continuous, ...gapped], 500, { mode: "complete", maximumVertices: 512 });
    const frames = plan.commands.filter((command) => command.type === "focusTrace").flatMap((command) => command.frames);

    expect(frames).toHaveLength(512);
    expect(frames).toEqual(expect.arrayContaining([1, 500, 2_200]));
    expect(plan.commands.filter((command) => command.type === "focusTrace").every((command) => command.markersOnly)).toBe(true);
  });
});
