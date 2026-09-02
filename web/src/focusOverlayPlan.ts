import type { Observation } from "./api";
import type { Point } from "./viewport";

export interface FocusBoxCommand {
  type: "focusBox";
  observation: Observation;
  trackId: number;
}

export interface FocusTraceCommand {
  type: "focusTrace";
  points: Point[];
}

export type FocusOverlayCommand = FocusBoxCommand | FocusTraceCommand;

export interface FocusOverlayPlan {
  commands: FocusOverlayCommand[];
}

export function buildFocusOverlayPlan(
  observations: readonly Observation[],
  frame: number,
  maxTracePoints: number,
): FocusOverlayPlan {
  const ordered = observations
    .filter((observation) => observation.frame <= frame)
    .sort((left, right) => left.frame - right.frame || left.row_index - right.row_index);
  const traceObservations = ordered.slice(-Math.max(0, maxTracePoints));
  const commands: FocusOverlayCommand[] = [];
  if (traceObservations.length > 1) {
    commands.push({
      type: "focusTrace",
      points: traceObservations.map((observation) => ({
        x: (observation.display_geometry.x1 + observation.display_geometry.x2) / 2,
        y: (observation.display_geometry.y1 + observation.display_geometry.y2) / 2,
      })),
    });
  }
  const current = ordered.findLast((observation) => observation.frame === frame);
  if (current?.usable_track_id !== null && current?.usable_track_id !== undefined) {
    commands.push({ type: "focusBox", observation: current, trackId: current.usable_track_id });
  }
  return { commands };
}