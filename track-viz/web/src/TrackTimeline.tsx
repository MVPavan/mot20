import type { CSSProperties, KeyboardEvent, PointerEvent } from "react";

import type { TimelineEventsResponse, TrackEvidenceResponse } from "./api";
import { deriveEventActivities, type EventActivity } from "./eventEpisodes";
import {
  aggregateTimelineGlyphs,
  framePercent,
  observationRuns,
  pointerFrame,
  seekFrameForKey,
  type TimelineEvidenceGlyph,
} from "./timelinePlan";

interface TrackTimelineProps {
  evidence: TrackEvidenceResponse;
  events: TimelineEventsResponse;
  frame: number;
  frameCount: number;
  onSeek(frame: number): void;
}

function familyLabel(family: EventActivity["family"]): string {
  return family === "scaleChange" ? "Scale change" : family[0].toUpperCase() + family.slice(1);
}

function boundedSummary(items: readonly string[], limit = 8): string {
  if (items.length === 0) return "none";
  const visible = items.slice(0, limit).join("; ");
  return items.length > limit ? `${visible}; and ${items.length - limit} more` : visible;
}

export function TrackTimeline({ evidence, events, frame, frameCount, onSeek }: TrackTimelineProps) {
  const activities = deriveEventActivities(events);
  const enabledActivities = [
    ...(events.settings.displacement_enabled ? activities.displacement : []),
    ...(events.settings.scale_change_enabled ? activities.scaleChange : []),
    ...(events.settings.close_interaction_enabled ? activities.proximity : []),
  ];
  const activityGlyphs: TimelineEvidenceGlyph[] = enabledActivities.map((activity) => ({
    frame: activity.anchorFrame,
    kind: "activity",
    label: `${familyLabel(activity.family)} activity${activity.competitorTrackId === undefined ? "" : ` with competitor ${activity.competitorTrackId}`} frames ${activity.startFrame}-${activity.endFrame}, anchor ${activity.anchorFrame}, severity ${activity.severity}, ${activity.rawMatchCount} raw match${activity.rawMatchCount === 1 ? "" : "es"}`,
  }));
  const confidenceGlyphs: TimelineEvidenceGlyph[] = events.confidence.meaningful
    ? [...new Set(events.low_confidence_observations.map((observation) => observation.frame))].map((lowFrame) => ({
      frame: lowFrame,
      kind: "low-confidence",
      label: `Low-confidence observation at frame ${lowFrame}`,
    }))
    : [];
  const glyphs = aggregateTimelineGlyphs([...activityGlyphs, ...confidenceGlyphs], frame, frameCount);
  const runs = observationRuns(evidence.observation_frames);
  const enabledThresholds = [
    events.settings.displacement_enabled ? `Displacement >= ${events.settings.displacement_threshold}` : null,
    events.settings.scale_change_enabled ? `Scale change >= ${events.settings.scale_change_threshold}` : null,
    events.settings.close_interaction_enabled ? `Close interaction <= ${events.settings.close_interaction_threshold}` : null,
  ].filter((value): value is string => value !== null);
  const timelineSummary = [
    `Sequence frames 1-${frameCount}; current frame ${frame}.`,
    `Observed runs: ${boundedSummary(runs.map((run) => run.startFrame === run.endFrame ? String(run.startFrame) : `${run.startFrame}-${run.endFrame}`))}.`,
    `Missing ranges: ${boundedSummary(evidence.gaps.map((gap) => `${gap.start_frame}-${gap.end_frame}`))}.`,
    `Event and confidence evidence: ${boundedSummary(glyphs.map((glyph) => glyph.label))}.`,
    glyphs.length >= 256 ? "Dense evidence is aggregated into temporal bins." : "",
  ].join(" ");
  const seekFromPointer = (event: PointerEvent<HTMLDivElement>) => onSeek(pointerFrame(event.clientX, event.currentTarget.getBoundingClientRect(), frameCount));
  const seekFromKeyboard = (event: KeyboardEvent<HTMLDivElement>) => {
    const target = seekFrameForKey(frame, event.key, event.shiftKey, frameCount);
    if (target === null) return;
    event.preventDefault();
    onSeek(target);
  };
  return (
    <section className="track-timeline" aria-label="Track timeline">
      <div
        aria-describedby="track-timeline-summary"
        aria-label={`Sequence timeline, current frame ${frame}`}
        aria-valuemax={frameCount}
        aria-valuemin={1}
        aria-valuenow={frame}
        className="track-timeline__rail"
        onKeyDown={seekFromKeyboard}
        onPointerDown={seekFromPointer}
        role="slider"
        tabIndex={0}
      >
        <span className="track-timeline__lifetime" style={{ left: `${framePercent(evidence.first_observation.frame, frameCount)}%`, right: `${100 - framePercent(evidence.last_observation.frame, frameCount)}%` }} />
        {runs.map((run) => (
          <span
            aria-hidden="true"
            className={`track-timeline__run${run.startFrame === run.endFrame ? " track-timeline__run--singleton" : ""}`}
            key={`${run.startFrame}-${run.endFrame}`}
            style={run.startFrame === run.endFrame
              ? { left: `${framePercent(run.startFrame, frameCount)}%` }
              : { left: `${framePercent(run.startFrame, frameCount)}%`, right: `${100 - framePercent(run.endFrame, frameCount)}%` }}
          />
        ))}
        {evidence.gaps.map((gap, index) => (
          <span aria-hidden="true" className="track-timeline__gap" key={`${gap.start_frame}-${gap.end_frame}`} style={{ left: `${framePercent(gap.start_frame, frameCount)}%`, "--timeline-lane": index % 3 } as CSSProperties} title={`Gap ${gap.start_frame}-${gap.end_frame}`} />
        ))}
        <span aria-hidden="true" className="track-timeline__endpoint" style={{ left: `${framePercent(evidence.first_observation.frame, frameCount)}%` }} />
        <span aria-hidden="true" className="track-timeline__endpoint" style={{ left: `${framePercent(evidence.last_observation.frame, frameCount)}%` }} />
        {glyphs.map((glyph, index) => (
          <span
            aria-hidden="true"
            className={`track-timeline__glyph track-timeline__glyph--${glyph.kinds.length === 1 ? glyph.kinds[0] : "mixed"}`}
            key={`${glyph.frame}-${glyph.frames.join("-")}-${index}`}
            style={{ left: `${framePercent(glyph.frame, frameCount)}%`, "--timeline-lane": index % 4 } as CSSProperties}
            title={glyph.label}
          />
        ))}
        <span aria-hidden="true" className="track-timeline__playhead" style={{ left: `${framePercent(frame, frameCount)}%` }} />
      </div>
      <p className="visually-hidden" id="track-timeline-summary">{timelineSummary}</p>
      <p id="track-timeline-description">
        Frames 1-{frameCount}. Use arrow keys to seek one frame, Shift+Arrow for ten frames, and Home or End for sequence bounds. {events.confidence.meaningful
          ? `Low confidence at or below ${events.confidence.threshold}.`
          : `Low confidence unavailable: ${events.confidence.status}.`}
        {enabledThresholds.length > 0 ? ` ${enabledThresholds.join("; ")}.` : ""}
        {glyphs.length >= 256 ? " Dense evidence is aggregated into temporal bins." : ""}
      </p>
    </section>
  );
}
