import { describe, expect, it } from "vitest";

import { adjacentFrame, gapStartFrames } from "./focusNavigation";

describe("Focus navigation", () => {
  it("navigates exact observation frames and exact gap starts", () => {
    const observationFrames = [1, 3, 6];
    const gaps = [
      { start_frame: 2, end_frame: 2, length: 1 },
      { start_frame: 4, end_frame: 5, length: 2 },
    ];

    expect(adjacentFrame(observationFrames, 4, -1)).toBe(3);
    expect(adjacentFrame(observationFrames, 4, 1)).toBe(6);
    expect(adjacentFrame(observationFrames, 6, 1)).toBeNull();
    expect(gapStartFrames(gaps)).toEqual([2, 4]);
    expect(adjacentFrame(gapStartFrames(gaps), 3, -1)).toBe(2);
    expect(adjacentFrame(gapStartFrames(gaps), 3, 1)).toBe(4);
  });
});
