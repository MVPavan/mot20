import { describe, expect, it } from "vitest";

import {
  HOVER_HYSTERESIS_CSS_PX,
  initialSelectionState,
  reduceSelection,
} from "./selectionState";

describe("selection state", () => {
  it("preserves hover inside the explicit hysteresis threshold and reranks beyond it", () => {
    let state = reduceSelection(initialSelectionState(), {
      type: "hover",
      rowIndexes: [3, 5],
      pointerCss: { x: 100, y: 100 },
    });
    state = reduceSelection(state, { type: "cycle", direction: 1 });
    expect(state.activeRowIndex).toBe(5);

    state = reduceSelection(state, {
      type: "hover",
      rowIndexes: [3, 5],
      pointerCss: { x: 100 + HOVER_HYSTERESIS_CSS_PX, y: 100 },
    });
    expect(state.activeRowIndex).toBe(5);

    state = reduceSelection(state, {
      type: "hover",
      rowIndexes: [3, 5],
      pointerCss: { x: 100 + HOVER_HYSTERESIS_CSS_PX + 0.01, y: 100 },
    });
    expect(state.activeRowIndex).toBe(3);
  });

  it("pins, cycles, confirms, escapes, and starts a new selection elsewhere", () => {
    let state = reduceSelection(initialSelectionState(), {
      type: "hover",
      rowIndexes: [4, 7, 9],
      pointerCss: { x: 20, y: 30 },
    });
    state = reduceSelection(state, { type: "pin", rowIndexes: [4, 7, 9] });
    expect(state).toMatchObject({ mode: "pinned", activeRowIndex: 4 });

    state = reduceSelection(state, { type: "cycle", direction: -1 });
    expect(state.activeRowIndex).toBe(9);
    state = reduceSelection(state, { type: "canvasClick", rowIndexes: [9, 12] });
    expect(state).toEqual({ mode: "confirmed", activeRowIndex: 9, rowIndexes: [9] });

    state = reduceSelection(state, { type: "escape" });
    expect(state).toEqual(initialSelectionState());

    state = reduceSelection(state, { type: "pin", rowIndexes: [1, 2] });
    state = reduceSelection(state, { type: "canvasClick", rowIndexes: [20, 21] });
    expect(state).toMatchObject({ mode: "pinned", activeRowIndex: 20, rowIndexes: [20, 21] });
  });

  it("keeps empty clicks and source resets in clean Explore state", () => {
    const pinned = reduceSelection(initialSelectionState(), { type: "pin", rowIndexes: [2] });
    expect(reduceSelection(pinned, { type: "reset" })).toEqual(initialSelectionState());
    expect(
      reduceSelection(initialSelectionState(), { type: "pin", rowIndexes: [] }),
    ).toEqual(initialSelectionState());
  });
});