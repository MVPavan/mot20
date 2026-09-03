import type {
  EventSettings,
  FilmstripResponse,
  SourceMetadata,
  TimelineEventsResponse,
  TrackEvidenceResponse,
} from "./api";
import { adjacentFrame, enabledEventFrames, gapStartFrames } from "./focusNavigation";
import { TrackFilmstrip } from "./TrackFilmstrip";
import { TrackTimeline } from "./TrackTimeline";

interface FocusReviewProps {
  source: SourceMetadata;
  frame: number;
  focusTarget: { trackId: number; confirmedRowIndex: number } | null;
  focusStatus: "idle" | "loading" | "ready" | "error";
  evidence: TrackEvidenceResponse | null;
  filmstrip: FilmstripResponse | null;
  events: TimelineEventsResponse | null;
  mode: "focus" | "context";
  contextCount: number;
  onSeek(frame: number): void;
  onExit(): void;
  onModeChange(mode: "focus" | "context"): void;
  onContextCountChange(count: number): void;
  onEventSettingsChange(settings: EventSettings): void;
}

interface EventControlProps {
  checked: boolean;
  label: string;
  threshold: number;
  thresholdLabel: string;
  onCheckedChange(checked: boolean): void;
  onThresholdChange(threshold: number): void;
}

function EventControl({
  checked,
  label,
  threshold,
  thresholdLabel,
  onCheckedChange,
  onThresholdChange,
}: EventControlProps) {
  const thresholdName = label === "Abrupt displacement" ? "Displacement" : label;
  return (
    <div className="event-control">
      <label>
        <input checked={checked} onChange={(event) => onCheckedChange(event.target.checked)} type="checkbox" />
        {label}
      </label>
      <label>
        <span>{thresholdLabel}</span>
        <input
          aria-label={`${thresholdName} threshold in box heights`}
          disabled={!checked}
          max="100"
          min="0"
          onChange={(event) => onThresholdChange(Math.min(100, Math.max(0, Number(event.target.value))))}
          step="0.05"
          type="number"
          value={threshold}
        />
      </label>
    </div>
  );
}

export function FocusReview({
  source,
  frame,
  focusTarget,
  focusStatus,
  evidence,
  filmstrip,
  events,
  mode,
  contextCount,
  onSeek,
  onExit,
  onModeChange,
  onContextCountChange,
  onEventSettingsChange,
}: FocusReviewProps) {
  if (focusTarget === null) return null;
  if (focusStatus === "loading") return <p className="focus-state" role="status">Loading track evidence</p>;
  if (focusStatus === "error" || evidence === null || filmstrip === null || events === null) {
    return <p className="focus-state focus-state--error" role="alert">Track evidence could not be loaded.</p>;
  }
  const current = evidence.observations.find((observation) => observation.frame === frame) ?? null;
  const previous = evidence.observations.findLast((observation) => observation.frame < frame) ?? null;
  const next = evidence.observations.find((observation) => observation.frame > frame) ?? null;
  const eventFrames = enabledEventFrames(events);
  const navigation = [
    ["Previous observation", adjacentFrame(evidence.observation_frames, frame, -1)],
    ["Next observation", adjacentFrame(evidence.observation_frames, frame, 1)],
    ["Previous gap", adjacentFrame(gapStartFrames(evidence.gaps), frame, -1)],
    ["Next gap", adjacentFrame(gapStartFrames(evidence.gaps), frame, 1)],
    ["Previous enabled event", adjacentFrame(eventFrames, frame, -1)],
    ["Next enabled event", adjacentFrame(eventFrames, frame, 1)],
  ] as const;

  return (
    <section className="focus-review" aria-label={`Focus review for track ${focusTarget.trackId}`}>
      <div className="focus-review__heading">
        <div>
          <p className="focus-review__label">Focus</p>
          <h2>Track {focusTarget.trackId}</h2>
        </div>
        <button onClick={onExit} type="button">Exit Focus</button>
      </div>
      <div className="review-settings">
        <fieldset className="mode-control">
          <legend>Overlay mode</legend>
          {(["focus", "context"] as const).map((option) => (
            <label key={option}>
              <input
                checked={mode === option}
                name="overlay-mode"
                onChange={() => onModeChange(option)}
                type="radio"
              />
              <span>{option === "focus" ? "Focus" : "Context"}</span>
            </label>
          ))}
        </fieldset>
        <label className="context-count">
          <span>Context tracks</span>
          <input
            max="8"
            min="0"
            onChange={(event) => onContextCountChange(Math.min(8, Math.max(0, Number(event.target.value))))}
            step="1"
            type="number"
            value={contextCount}
          />
        </label>
      </div>
      <fieldset className="event-controls">
        <legend>Optional review events</legend>
        <EventControl
          checked={events.settings.displacement_enabled}
          label="Abrupt displacement"
          onCheckedChange={(checked) => onEventSettingsChange({ ...events.settings, displacement_enabled: checked })}
          onThresholdChange={(threshold) => onEventSettingsChange({ ...events.settings, displacement_threshold: threshold })}
          threshold={events.settings.displacement_threshold}
          thresholdLabel="Center movement"
        />
        <EventControl
          checked={events.settings.scale_change_enabled}
          label="Scale change"
          onCheckedChange={(checked) => onEventSettingsChange({ ...events.settings, scale_change_enabled: checked })}
          onThresholdChange={(threshold) => onEventSettingsChange({ ...events.settings, scale_change_threshold: threshold })}
          threshold={events.settings.scale_change_threshold}
          thresholdLabel="Height change"
        />
        <EventControl
          checked={events.settings.close_interaction_enabled}
          label="Close interaction"
          onCheckedChange={(checked) => onEventSettingsChange({ ...events.settings, close_interaction_enabled: checked })}
          onThresholdChange={(threshold) => onEventSettingsChange({ ...events.settings, close_interaction_threshold: threshold })}
          threshold={events.settings.close_interaction_threshold}
          thresholdLabel="Edge proximity"
        />
      </fieldset>
      <div className="focus-navigation" role="group" aria-label="Track evidence navigation">
        {navigation.map(([label, target]) => (
          <button disabled={target === null} key={label} onClick={() => target !== null && onSeek(target)} type="button">
            {label}
          </button>
        ))}
      </div>
      {current === null ? (
        <p className="gap-evidence" role="status">
          Gap at frame {frame}. Previous observation {previous?.frame ?? "none"}; next observation {next?.frame ?? "none"}.
        </p>
      ) : (
        <p className="gap-evidence">Observed on exact frame {frame}, source row {current.row_index}.</p>
      )}
      <TrackTimeline evidence={evidence} events={events} frameCount={source.frame_count} onSeek={onSeek} />
      <TrackFilmstrip filmstrip={filmstrip} onSeek={onSeek} source={source} />
    </section>
  );
}