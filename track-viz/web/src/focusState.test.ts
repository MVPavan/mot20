import { describe, expect, it } from "vitest";

import type { Observation } from "./api";
import { initialFocusState, reduceFocus } from "./focusState";

const SOURCE = { sourceKey: "mot20-01", sourceHash: "hash-a" };

function observation(usableTrackId: number | null): Observation {
  return {
    source_key: SOURCE.sourceKey,
    sequence: "MOT20-01",
    frame: 12,
    row_index: 4,
    row_hash: "row-hash",
    source_hash: SOURCE.sourceHash,
    raw_track_id: usableTrackId ?? -1,
    usable_track_id: usableTrackId,
    raw_geometry: { x: 10, y: 20, width: 30, height: 40 },
    display_geometry: { x1: 10, y1: 20, x2: 40, y2: 60 },
    score: 0.9,
    ground_truth: null,
    opaque_result_fields: null,
    score_semantics: "tracker_score",
    ground_truth_semantics: "not_defined",
  };
}

describe("focus state", () => {
  it("enters Focus only for a confirmed observation with a usable track ID", () => {
    const initial = initialFocusState(SOURCE);

    expect(reduceFocus(initial, { type: "confirm", observation: observation(null) })).toEqual(
      initial,
    );
    expect(reduceFocus(initial, { type: "confirm", observation: observation(8) })).toEqual({
      mode: "focus",
      sourceKey: SOURCE.sourceKey,
      sourceHash: SOURCE.sourceHash,
      trackId: 8,
      confirmedRowIndex: 4,
    });
  });

  it("resets by source key and hash and ignores stale confirmations", () => {
    const focused = reduceFocus(initialFocusState(SOURCE), {
      type: "confirm",
      observation: observation(8),
    });
    const switched = reduceFocus(focused, {
      type: "sourceChanged",
      source: { sourceKey: "mot20-06", sourceHash: "hash-b" },
    });

    expect(switched).toEqual({ mode: "explore", sourceKey: "mot20-06", sourceHash: "hash-b" });
    expect(reduceFocus(switched, { type: "confirm", observation: observation(8) })).toEqual(
      switched,
    );
    expect(reduceFocus(focused, { type: "escape" })).toEqual(initialFocusState(SOURCE));
  });
});