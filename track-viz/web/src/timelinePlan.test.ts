import { describe, expect, it } from "vitest";

import {
  aggregateTimelineGlyphs,
  observationRuns,
  pointerFrame,
  seekFrameForKey,
} from "./timelinePlan";

describe("timeline plan", () => {
  it("keeps continuous and gapped observation runs exact", () => {
    expect(observationRuns([1, 2, 3, 7, 9, 10])).toEqual([
      { startFrame: 1, endFrame: 3 },
      { startFrame: 7, endFrame: 7 },
      { startFrame: 9, endFrame: 10 },
    ]);
  });

  it("maps pointers to exact one-based frames by deterministic rounding", () => {
    expect(pointerFrame(100, { left: 0, width: 100 }, 11)).toBe(11);
    expect(pointerFrame(45, { left: 0, width: 100 }, 11)).toBe(6);
    expect(pointerFrame(-20, { left: 0, width: 100 }, 11)).toBe(1);
  });

  it("seeks intermediate frames with the full keyboard contract", () => {
    expect(seekFrameForKey(5, "ArrowRight", false, 20)).toBe(6);
    expect(seekFrameForKey(5, "ArrowLeft", true, 20)).toBe(1);
    expect(seekFrameForKey(5, "ArrowRight", true, 20)).toBe(15);
    expect(seekFrameForKey(5, "Home", false, 20)).toBe(1);
    expect(seekFrameForKey(5, "End", false, 20)).toBe(20);
  });

  it("aggregates dense glyphs deterministically without losing endpoint and current activity evidence", () => {
    const glyphs = Array.from({ length: 600 }, (_, index) => ({
      frame: index + 1,
      kind: (index % 2 === 0 ? "activity" : "low-confidence") as "activity" | "low-confidence",
      label: `Evidence ${index + 1}`,
    }));
    const plan = aggregateTimelineGlyphs(glyphs, 301, 600, 256);

    expect(plan).toHaveLength(256);
    expect(plan.map((glyph) => glyph.frame)).toContain(1);
    expect(plan.map((glyph) => glyph.frame)).toContain(600);
    expect(plan.some((glyph) => glyph.frames.includes(301))).toBe(true);
    expect(plan.some((glyph) => glyph.frames.includes(300))).toBe(true);
    expect(plan.some((glyph) => glyph.frames.includes(302))).toBe(true);
  });
});
