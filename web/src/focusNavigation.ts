import type { TrackGap } from "./api";

export function adjacentFrame(
  frames: readonly number[],
  currentFrame: number,
  direction: -1 | 1,
): number | null {
  const ordered = [...new Set(frames)].sort((left, right) => left - right);
  if (direction === 1) {
    return ordered.find((frame) => frame > currentFrame) ?? null;
  }
  return ordered.findLast((frame) => frame < currentFrame) ?? null;
}

export function gapStartFrames(gaps: readonly TrackGap[]): number[] {
  return gaps.map((gap) => gap.start_frame).sort((left, right) => left - right);
}

interface EventNavigationData {
  settings: {
    displacement_enabled: boolean;
    scale_change_enabled: boolean;
    close_interaction_enabled: boolean;
  };
  confidence: { meaningful: boolean };
  displacement_events: ReadonlyArray<{ to_frame: number }>;
  scale_change_events: ReadonlyArray<{ to_frame: number }>;
  close_interaction_events: ReadonlyArray<{ frame: number }>;
  low_confidence_observations: ReadonlyArray<{ frame: number }>;
}

export function enabledEventFrames(events: EventNavigationData): number[] {
  const frames = [
    ...(events.settings.displacement_enabled
      ? events.displacement_events.map((event) => event.to_frame)
      : []),
    ...(events.settings.scale_change_enabled
      ? events.scale_change_events.map((event) => event.to_frame)
      : []),
    ...(events.settings.close_interaction_enabled
      ? events.close_interaction_events.map((event) => event.frame)
      : []),
    ...(events.confidence.meaningful
      ? events.low_confidence_observations.map((observation) => observation.frame)
      : []),
  ];
  return [...new Set(frames)].sort((left, right) => left - right);
}