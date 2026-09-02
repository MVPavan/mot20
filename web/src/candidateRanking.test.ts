import { describe, expect, it } from "vitest";

import { rankCandidates, type HitRegion } from "./candidateRanking";

describe("rankCandidates", () => {
  it("orders by containment, area, normalized pointer geometry, edge distance, then row index", () => {
    const regions: HitRegion[] = [
      { rowIndex: 90, x1: 30, y1: 30, x2: 32, y2: 32 },
      { rowIndex: 50, x1: 0, y1: 0, x2: 20, y2: 20 },
      { rowIndex: 40, x1: 5, y1: 5, x2: 15, y2: 15 },
      { rowIndex: 30, x1: 8, y1: 0, x2: 13, y2: 20 },
      { rowIndex: 20, x1: 7, y1: 4, x2: 17, y2: 14 },
      { rowIndex: 10, x1: 7, y1: 4, x2: 17, y2: 14 },
    ];

    expect(rankCandidates(regions, { x: 10, y: 10 }).map(({ region }) => region.rowIndex)).toEqual([
      40,
      30,
      10,
      20,
      50,
      90,
    ]);
  });

  it("includes boxes touching the pointer at an image edge", () => {
    const regions: HitRegion[] = [{ rowIndex: 1, x1: 0, y1: 0, x2: 12, y2: 20 }];

    expect(rankCandidates(regions, { x: 0, y: 0 })[0]?.contains).toBe(true);
  });

  it("uses image-space edge distance before row index when prior metrics tie", () => {
    const regions: HitRegion[] = [
      { rowIndex: 1, x1: -2.5, y1: -2.5, x2: 7.5, y2: 7.5 },
      { rowIndex: 2, x1: -5, y1: -1.25, x2: 15, y2: 3.75 },
    ];

    expect(rankCandidates(regions, { x: 0, y: 0 }).map(({ region }) => region.rowIndex)).toEqual([
      2,
      1,
    ]);
  });
});