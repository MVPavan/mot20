import type { Observation } from "./api";

export interface SourceIdentity {
  sourceKey: string;
  sourceHash: string;
}

interface ExploreFocusState extends SourceIdentity {
  mode: "explore";
}

interface ActiveFocusState extends SourceIdentity {
  mode: "focus";
  trackId: number;
  confirmedRowIndex: number;
}

export type FocusState = ExploreFocusState | ActiveFocusState;

export type FocusAction =
  | { type: "confirm"; observation: Observation }
  | { type: "sourceChanged"; source: SourceIdentity }
  | { type: "escape" };

export function initialFocusState(source: SourceIdentity): ExploreFocusState {
  return { sourceKey: source.sourceKey, sourceHash: source.sourceHash, mode: "explore" };
}

export function reduceFocus(state: FocusState, action: FocusAction): FocusState {
  if (action.type === "sourceChanged") {
    return initialFocusState(action.source);
  }
  if (action.type === "escape") {
    return initialFocusState(state);
  }
  if (
    action.observation.usable_track_id === null ||
    action.observation.source_key !== state.sourceKey ||
    action.observation.source_hash !== state.sourceHash
  ) {
    return state;
  }
  return {
    mode: "focus",
    sourceKey: state.sourceKey,
    sourceHash: state.sourceHash,
    trackId: action.observation.usable_track_id,
    confirmedRowIndex: action.observation.row_index,
  };
}