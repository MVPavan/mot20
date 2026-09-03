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
