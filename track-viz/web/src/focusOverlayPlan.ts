import type { Observation } from "./api";
import type { Point } from "./viewport";

export type TrajectoryMode = "past" | "complete";

export interface FocusBoxCommand { type: "focusBox"; observation: Observation; trackId: number; }
export interface FocusTraceCommand {
  type: "focusTrace";
  points: Point[];
  frames: number[];
  future: boolean;
  markersOnly: boolean;
}
export type FocusOverlayCommand = FocusBoxCommand | FocusTraceCommand;
export interface FocusOverlayPlan { commands: FocusOverlayCommand[]; simplified: boolean; }
export interface FocusOverlayOptions { mode: TrajectoryMode; maximumVertices: number; }

function pointFor(observation: Observation): Point {
  return { x: (observation.display_geometry.x1 + observation.display_geometry.x2) / 2, y: (observation.display_geometry.y1 + observation.display_geometry.y2) / 2 };
}

function splitRuns(observations: readonly Observation[]): Observation[][] {
  const runs: Observation[][] = [];
  for (const observation of observations) {
    const lastRun = runs.at(-1);
    const previous = lastRun?.at(-1);
    if (previous !== undefined && observation.frame === previous.frame + 1) lastRun!.push(observation);
    else runs.push([observation]);
  }
  return runs;
}

function sampledObservations(observations: readonly Observation[], frame: number, maximumVertices: number) {
  if (observations.length <= maximumVertices) return { observations: [...observations], simplified: false, markersOnly: false };
  const mandatory = new Set<Observation>();
  for (const run of splitRuns(observations)) {
    mandatory.add(run[0]);
    mandatory.add(run.at(-1)!);
  }
  const current = observations.find((observation) => observation.frame === frame);
  if (current !== undefined) mandatory.add(current);
  if (mandatory.size > maximumVertices) {
    const protectedObservations = [observations[0], observations.at(-1)!, current]
      .filter((observation): observation is Observation => observation !== undefined)
      .filter((observation, index, values) => values.indexOf(observation) === index);
    const protectedSet = new Set(protectedObservations);
    const boundaries = [...mandatory].filter((observation) => !protectedSet.has(observation));
    const remaining = Math.max(0, maximumVertices - protectedObservations.length);
    const selected = new Set(protectedObservations);
    for (let index = 0; index < remaining; index += 1) {
      selected.add(boundaries[Math.floor(index * boundaries.length / remaining)]);
    }
    return {
      observations: observations.filter((observation) => selected.has(observation)),
      simplified: true,
      markersOnly: true,
    };
  }
  const remainder = observations.filter((observation) => !mandatory.has(observation));
  const count = maximumVertices - mandatory.size;
  const sampled = new Set(mandatory);
  for (let index = 0; index < count; index += 1) sampled.add(remainder[Math.floor(index * remainder.length / count)]);
  return { observations: observations.filter((observation) => sampled.has(observation)), simplified: true, markersOnly: false };
}

export function buildFocusOverlayPlan(
  observations: readonly Observation[],
  frame: number,
  options: FocusOverlayOptions | number = { mode: "past", maximumVertices: 512 },
): FocusOverlayPlan {
  const resolved = typeof options === "number" ? { mode: "past" as const, maximumVertices: options } : options;
  const ordered = observations
    .filter((observation) => resolved.mode === "complete" || observation.frame <= frame)
    .sort((left, right) => left.frame - right.frame || left.row_index - right.row_index);
  const sampled = sampledObservations(ordered, frame, Math.max(1, resolved.maximumVertices));
  const selected = new Set(sampled.observations);
  const commands: FocusOverlayCommand[] = [];
  for (const run of splitRuns(ordered)) {
    const visible = run.filter((observation) => selected.has(observation));
    const chunks: Observation[][] = [];
    for (const observation of visible) {
      const previous = chunks.at(-1)?.at(-1);
      const future = resolved.mode === "complete" && observation.frame > frame;
      const previousFuture = previous !== undefined && resolved.mode === "complete" && previous.frame > frame;
      if (previous !== undefined && future === previousFuture) chunks.at(-1)!.push(observation);
      else chunks.push([observation]);
    }
    for (const chunk of chunks) {
      commands.push({
        type: "focusTrace",
        points: chunk.map(pointFor),
        frames: chunk.map((observation) => observation.frame),
        future: resolved.mode === "complete" && chunk[0].frame > frame,
        markersOnly: sampled.markersOnly || (sampled.simplified && chunk.length === 1),
      });
    }
  }
  const current = ordered.findLast((observation) => observation.frame === frame);
  if (current?.usable_track_id !== null && current?.usable_track_id !== undefined) commands.push({ type: "focusBox", observation: current, trackId: current.usable_track_id });
  return { commands, simplified: sampled.simplified };
}
