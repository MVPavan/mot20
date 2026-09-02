import { describe, expect, it } from "vitest";

import { buildOverlayPlan } from "./overlayPlan";
import { initialSelectionState, reduceSelection } from "./selectionState";

const regions = [
  { rowIndex: 3, x1: 2, y1: 4, x2: 18, y2: 30 },
  { rowIndex: 8, x1: 7, y1: 6, x2: 22, y2: 32 },
];

describe("buildOverlayPlan", () => {
  it("is stroke-free and command-empty in Explore when the pointer is off-canvas", () => {
    const plan = buildOverlayPlan({
      selection: initialSelectionState(),
      regions,
      pointerImage: null,
      revealAll: false,
    });

    expect(plan.commands).toEqual([]);
    expect(plan.strokeCount).toBe(0);
  });

  it("draws only the current hover and bounded magnifier unless B reveal is held", () => {
    const selection = reduceSelection(initialSelectionState(), {
      type: "hover",
      rowIndexes: [3, 8],
      pointerCss: { x: 10, y: 12 },
    });
    const quiet = buildOverlayPlan({
      selection,
      regions,
      pointerImage: { x: 10, y: 12 },
      revealAll: false,
    });
    const revealed = buildOverlayPlan({
      selection,
      regions,
      pointerImage: { x: 10, y: 12 },
      revealAll: true,
    });

    expect(quiet.commands.map((command) => command.type)).toEqual(["box", "magnifier"]);
    expect(quiet.strokeCount).toBe(2);
    expect(revealed.commands.filter((command) => command.type === "box")).toHaveLength(2);
  });

  it("numbers every frozen pinned candidate and reduces confirmation to the exact active box", () => {
    const pinned = reduceSelection(initialSelectionState(), { type: "pin", rowIndexes: [3, 8] });
    const pinnedPlan = buildOverlayPlan({ selection: pinned, regions, pointerImage: null, revealAll: false });
    const confirmed = reduceSelection(pinned, { type: "confirm" });
    const confirmedPlan = buildOverlayPlan({
      selection: confirmed,
      regions,
      pointerImage: null,
      revealAll: false,
    });

    expect(pinnedPlan.commands).toMatchObject([
      { type: "box", rowIndex: 3, number: 1 },
      { type: "box", rowIndex: 8, number: 2 },
    ]);
    expect(confirmedPlan.commands).toMatchObject([{ type: "box", rowIndex: 3 }]);
  });
});