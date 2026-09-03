export interface ObservationRun {
  startFrame: number;
  endFrame: number;
}

export interface TimelineEvidenceGlyph {
  frame: number;
  kind: "activity" | "low-confidence";
  label: string;
}

export interface AggregatedTimelineGlyph {
  frame: number;
  frames: number[];
  kinds: Array<TimelineEvidenceGlyph["kind"]>;
  label: string;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

export function observationRuns(frames: readonly number[]): ObservationRun[] {
  const ordered = [...new Set(frames)].sort((left, right) => left - right);
  const runs: ObservationRun[] = [];
  for (const frame of ordered) {
    const previous = runs.at(-1);
    if (previous !== undefined && frame === previous.endFrame + 1) {
      previous.endFrame = frame;
    } else {
      runs.push({ startFrame: frame, endFrame: frame });
    }
  }
  return runs;
}

export function framePercent(frame: number, frameCount: number): number {
  return ((clamp(frame, 1, Math.max(1, frameCount)) - 1) / Math.max(1, frameCount - 1)) * 100;
}

export function pointerFrame(clientX: number, rail: { left: number; width: number }, frameCount: number): number {
  if (frameCount <= 1 || rail.width <= 0) return 1;
  const fraction = clamp((clientX - rail.left) / rail.width, 0, 1);
  return Math.round(fraction * (frameCount - 1)) + 1;
}

export function seekFrameForKey(
  frame: number,
  key: string,
  shiftKey: boolean,
  frameCount: number,
): number | null {
  const step = shiftKey ? 10 : 1;
  if (key === "ArrowLeft" || key === "ArrowDown") return clamp(frame - step, 1, frameCount);
  if (key === "ArrowRight" || key === "ArrowUp") return clamp(frame + step, 1, frameCount);
  if (key === "Home") return 1;
  if (key === "End") return frameCount;
  return null;
}

export function aggregateTimelineGlyphs(
  glyphs: readonly TimelineEvidenceGlyph[],
  currentFrame: number,
  frameCount: number,
  maximumGlyphs = 256,
): AggregatedTimelineGlyph[] {
  const byFrame = new Map<number, TimelineEvidenceGlyph[]>();
  for (const glyph of glyphs) {
    const atFrame = byFrame.get(glyph.frame) ?? [];
    atFrame.push(glyph);
    byFrame.set(glyph.frame, atFrame);
  }
  const frames = [...byFrame.keys()].sort((left, right) => left - right);
  if (frames.length === 0 || maximumGlyphs <= 0) return [];
  const activityFrames = frames.filter((frame) => byFrame.get(frame)?.some((glyph) => glyph.kind === "activity"));
  const required = new Set<number>([frames[0], frames.at(-1)!]);
  const previousActivity = activityFrames.filter((frame) => frame < currentFrame).at(-1);
  const nextActivity = activityFrames.find((frame) => frame > currentFrame);
  if (previousActivity !== undefined) required.add(previousActivity);
  if (nextActivity !== undefined) required.add(nextActivity);
  const currentEvidence = frames.find((frame) => frame === currentFrame);
  if (currentEvidence !== undefined) required.add(currentEvidence);

  const bins = new Map<number, number[]>();
  for (const frame of frames) {
    const bin = Math.min(maximumGlyphs - 1, Math.floor(((frame - 1) / Math.max(1, frameCount)) * maximumGlyphs));
    const values = bins.get(bin) ?? [];
    values.push(frame);
    bins.set(bin, values);
  }
  return [...bins.entries()].sort(([left], [right]) => left - right).map(([, binnedFrames]) => {
    const representative = binnedFrames.find((frame) => required.has(frame)) ?? binnedFrames[0];
    const evidence = binnedFrames.flatMap((frame) => byFrame.get(frame) ?? []);
    const kinds = [...new Set(evidence.map((glyph) => glyph.kind))];
    const labels = evidence.map((glyph) => glyph.label);
    const label = labels.length === 1
      ? labels[0]
      : `${labels.length} evidence markers across frames ${binnedFrames[0]}-${binnedFrames.at(-1)}`;
    return { frame: representative, frames: binnedFrames, kinds, label };
  });
}
