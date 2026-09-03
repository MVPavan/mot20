import type { TimelineEventsResponse, TrackEvidenceResponse } from "./api";
import { enabledEventFrames } from "./focusNavigation";

interface TrackTimelineProps {
  evidence: TrackEvidenceResponse;
  events: TimelineEventsResponse;
  frameCount: number;
  onSeek(frame: number): void;
}

export function TrackTimeline({ evidence, events, frameCount, onSeek }: TrackTimelineProps) {
  const markers = [
    { frame: evidence.first_observation.frame, label: "First observation", kind: "lifespan" },
    { frame: evidence.last_observation.frame, label: "Last observation", kind: "lifespan" },
    ...evidence.gaps.map((gap) => ({ frame: gap.start_frame, label: `Gap ${gap.start_frame}-${gap.end_frame}`, kind: "gap" })),
    ...enabledEventFrames(events).map((frame) => ({ frame, label: `Enabled event at frame ${frame}`, kind: "heuristic" })),
  ];
  const enabledThresholds = [
    events.settings.displacement_enabled
      ? `Displacement >= ${events.settings.displacement_threshold}`
      : null,
    events.settings.scale_change_enabled
      ? `Scale change >= ${events.settings.scale_change_threshold}`
      : null,
    events.settings.close_interaction_enabled
      ? `Close interaction <= ${events.settings.close_interaction_threshold}`
      : null,
  ].filter((value): value is string => value !== null);
  return (
    <section className="track-timeline" aria-label="Track timeline">
      <div className="track-timeline__rail">
        {markers.map((marker, index) => (
          <button
            aria-label={`${marker.label}, seek frame ${marker.frame}`}
            className={`track-timeline__marker track-timeline__marker--${marker.kind}`}
            key={`${marker.label}-${marker.frame}-${index}`}
            onClick={() => onSeek(marker.frame)}
            style={{ left: `${((marker.frame - 1) / Math.max(1, frameCount - 1)) * 100}%` }}
            title={marker.label}
            type="button"
          />
        ))}
      </div>
      <p>
        Frames 1-{frameCount}. {events.confidence.meaningful
          ? `Low confidence at or below ${events.confidence.threshold}.`
          : `Low confidence unavailable: ${events.confidence.status}.`}
        {enabledThresholds.length > 0 ? ` ${enabledThresholds.join("; ")}.` : ""}
      </p>
    </section>
  );
}