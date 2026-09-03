export type EventFamily = "displacement" | "scaleChange" | "proximity";

export interface EventActivity {
  family: EventFamily;
  startFrame: number;
  endFrame: number;
  anchorFrame: number;
  severity: number;
  rawMatchCount: number;
  competitorTrackId?: number;
}

interface TransitionEvent {
  from_frame: number;
  to_frame: number;
  frame_delta: number;
}

interface ProximityEvent {
  frame: number;
  competitor_track_id: number;
  normalized_edge_proximity: number;
}

export interface EventActivityInput {
  displacement_events: ReadonlyArray<TransitionEvent & { normalized_displacement: number }>;
  scale_change_events: ReadonlyArray<TransitionEvent & { normalized_scale_change: number }>;
  close_interaction_events: ReadonlyArray<ProximityEvent>;
}

export interface EventActivities {
  displacement: EventActivity[];
  scaleChange: EventActivity[];
  proximity: EventActivity[];
}

function transitionActivities(
  family: "displacement" | "scaleChange",
  events: ReadonlyArray<TransitionEvent & { severity: number }>,
): EventActivity[] {
  const ordered = [...events].sort((left, right) => (
    left.from_frame - right.from_frame || left.to_frame - right.to_frame
  ));
  const activities: EventActivity[] = [];
  let run: Array<TransitionEvent & { severity: number }> = [];

  function finishRun(): void {
    if (run.length === 0) return;
    const anchor = run.reduce((best, event) => (
      event.severity > best.severity
      || (event.severity === best.severity && event.to_frame < best.to_frame)
        ? event
        : best
    ));
    activities.push({
      family,
      startFrame: run[0].from_frame,
      endFrame: run.at(-1)!.to_frame,
      anchorFrame: anchor.to_frame,
      severity: anchor.severity,
      rawMatchCount: run.length,
    });
    run = [];
  }

  for (const event of ordered) {
    const previous = run.at(-1);
    const contiguous = previous !== undefined
      && previous.frame_delta === 1
      && event.frame_delta === 1
      && previous.to_frame === event.from_frame;
    if (!contiguous) finishRun();
    run.push(event);
  }
  finishRun();
  return activities;
}

function proximityActivities(events: readonly ProximityEvent[]): EventActivity[] {
  const byCompetitor = new Map<number, ProximityEvent[]>();
  for (const event of events) {
    const competitor = byCompetitor.get(event.competitor_track_id) ?? [];
    competitor.push(event);
    byCompetitor.set(event.competitor_track_id, competitor);
  }
  const activities: EventActivity[] = [];
  for (const [competitorTrackId, competitorEvents] of byCompetitor) {
    const ordered = [...competitorEvents].sort((left, right) => left.frame - right.frame);
    let run: ProximityEvent[] = [];
    const finishRun = () => {
      if (run.length === 0) return;
      const anchor = run.reduce((best, event) => (
        event.normalized_edge_proximity < best.normalized_edge_proximity
        || (event.normalized_edge_proximity === best.normalized_edge_proximity && event.frame < best.frame)
          ? event
          : best
      ));
      activities.push({
        family: "proximity",
        startFrame: run[0].frame,
        endFrame: run.at(-1)!.frame,
        anchorFrame: anchor.frame,
        severity: anchor.normalized_edge_proximity,
        rawMatchCount: run.length,
        competitorTrackId,
      });
      run = [];
    };
    for (const event of ordered) {
      if (run.length > 0 && event.frame !== run.at(-1)!.frame + 1) finishRun();
      run.push(event);
    }
    finishRun();
  }
  return activities.sort((left, right) => (
    left.anchorFrame - right.anchorFrame
    || (left.competitorTrackId ?? 0) - (right.competitorTrackId ?? 0)
    || left.startFrame - right.startFrame
  ));
}

export function deriveEventActivities(events: EventActivityInput): EventActivities {
  return {
    displacement: transitionActivities("displacement", events.displacement_events.map((event) => ({
      ...event,
      severity: event.normalized_displacement,
    }))),
    scaleChange: transitionActivities("scaleChange", events.scale_change_events.map((event) => ({
      ...event,
      severity: event.normalized_scale_change,
    }))),
    proximity: proximityActivities(events.close_interaction_events),
  };
}

export function activityNavigationTarget(
  activities: readonly EventActivity[],
  currentFrame: number,
  direction: -1 | 1,
): number | null {
  const ordered = [...activities].sort((left, right) => (
    left.startFrame - right.startFrame || left.endFrame - right.endFrame || left.anchorFrame - right.anchorFrame
  ));
  if (direction === 1) return ordered.find((activity) => activity.startFrame > currentFrame)?.anchorFrame ?? null;
  return ordered.findLast((activity) => activity.endFrame < currentFrame)?.anchorFrame ?? null;
}

export function lowConfidenceNavigationTarget(
  frames: readonly number[],
  currentFrame: number,
  direction: -1 | 1,
): number | null {
  const ordered = [...new Set(frames)].sort((left, right) => left - right);
  return direction === 1
    ? ordered.find((frame) => frame > currentFrame) ?? null
    : ordered.findLast((frame) => frame < currentFrame) ?? null;
}

export function activitiesAtFrame(activities: readonly EventActivity[], frame: number): EventActivity[] {
  return activities.filter((activity) => activity.startFrame <= frame && frame <= activity.endFrame);
}
