import type { Point } from "./viewport";

export const HOVER_HYSTERESIS_CSS_PX = 8;

interface ExploreSelection {
  mode: "explore";
  rowIndexes: number[];
  activeRowIndex: number | null;
  pointerCss: Point | null;
}

interface PinnedSelection {
  mode: "pinned";
  rowIndexes: number[];
  activeRowIndex: number;
}

interface ConfirmedSelection {
  mode: "confirmed";
  rowIndexes: number[];
  activeRowIndex: number;
}

export type SelectionState = ExploreSelection | PinnedSelection | ConfirmedSelection;

export type SelectionAction =
  | { type: "hover"; rowIndexes: number[]; pointerCss: Point }
  | { type: "cycle"; direction: -1 | 1 }
  | { type: "pin"; rowIndexes: number[] }
  | { type: "canvasClick"; rowIndexes: number[] }
  | { type: "activate"; rowIndex: number }
  | { type: "confirm" }
  | { type: "escape" }
  | { type: "reset" };

export function initialSelectionState(): ExploreSelection {
  return { mode: "explore", rowIndexes: [], activeRowIndex: null, pointerCss: null };
}

function pinnedSelection(rowIndexes: number[], preferredRowIndex: number | null): SelectionState {
  if (rowIndexes.length === 0) {
    return initialSelectionState();
  }
  return {
    mode: "pinned",
    rowIndexes: [...rowIndexes],
    activeRowIndex:
      preferredRowIndex !== null && rowIndexes.includes(preferredRowIndex)
        ? preferredRowIndex
        : rowIndexes[0],
  };
}

export function reduceSelection(state: SelectionState, action: SelectionAction): SelectionState {
  switch (action.type) {
    case "hover": {
      if (state.mode !== "explore") {
        return state;
      }
      if (action.rowIndexes.length === 0) {
        return { ...initialSelectionState(), pointerCss: action.pointerCss };
      }
      const movement =
        state.pointerCss === null
          ? Number.POSITIVE_INFINITY
          : Math.hypot(
              action.pointerCss.x - state.pointerCss.x,
              action.pointerCss.y - state.pointerCss.y,
            );
      const preserveActive =
        movement <= HOVER_HYSTERESIS_CSS_PX &&
        state.activeRowIndex !== null &&
        action.rowIndexes.includes(state.activeRowIndex);
      return {
        mode: "explore",
        rowIndexes: [...action.rowIndexes],
        activeRowIndex: preserveActive ? state.activeRowIndex : action.rowIndexes[0],
        pointerCss: preserveActive ? state.pointerCss : action.pointerCss,
      };
    }
    case "cycle": {
      if (state.rowIndexes.length === 0 || state.activeRowIndex === null) {
        return state;
      }
      const currentIndex = state.rowIndexes.indexOf(state.activeRowIndex);
      const nextIndex =
        (currentIndex + action.direction + state.rowIndexes.length) % state.rowIndexes.length;
      return { ...state, activeRowIndex: state.rowIndexes[nextIndex] };
    }
    case "pin":
      return pinnedSelection(action.rowIndexes, state.activeRowIndex);
    case "canvasClick":
      if (
        state.mode === "pinned" &&
        state.activeRowIndex !== null &&
        action.rowIndexes.includes(state.activeRowIndex)
      ) {
        return {
          mode: "confirmed",
          rowIndexes: [state.activeRowIndex],
          activeRowIndex: state.activeRowIndex,
        };
      }
      if (
        state.mode === "confirmed" &&
        action.rowIndexes.includes(state.activeRowIndex)
      ) {
        return state;
      }
      return pinnedSelection(action.rowIndexes, null);
    case "activate":
      return state.mode === "pinned" && state.rowIndexes.includes(action.rowIndex)
        ? { ...state, activeRowIndex: action.rowIndex }
        : state;
    case "confirm":
      return state.mode === "pinned"
        ? {
            mode: "confirmed",
            rowIndexes: [state.activeRowIndex],
            activeRowIndex: state.activeRowIndex,
          }
        : state;
    case "escape":
    case "reset":
      return initialSelectionState();
  }
}