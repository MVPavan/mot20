import { useEffect, useRef, useState, type KeyboardEvent } from "react";

import type {
  EventSettings,
  FilmstripResponse,
  SourceMetadata,
  TimelineEventsResponse,
  TrackEvidenceResponse,
} from "./api";
import {
  activitiesAtFrame,
  activityNavigationTarget,
  deriveEventActivities,
  lowConfidenceNavigationTarget,
  type EventActivity,
} from "./eventEpisodes";
import { adjacentFrame, gapStartFrames } from "./focusNavigation";
import { TrackFilmstrip } from "./TrackFilmstrip";
import { TrackTimeline } from "./TrackTimeline";
import { buildFocusOverlayPlan, type TrajectoryMode } from "./focusOverlayPlan";

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
  eventSettings: EventSettings;
  eventStatus: "idle" | "loading" | "updating" | "error";
  eventError: string;
  onSeek(frame: number): void;
  onExit(): void;
  onModeChange(mode: "focus" | "context"): void;
  onContextCountChange(count: number): void;
  onEventSettingsChange(update: (settings: EventSettings) => EventSettings): void;
  onRetryEvents(): void;
  trajectoryMode: TrajectoryMode;
  onTrajectoryModeChange(mode: TrajectoryMode): void;
}

interface EventControlProps {
  checked: boolean;
  label: string;
  threshold: number;
  thresholdLabel: string;
  onCheckedChange(checked: boolean): void;
  onThresholdChange(threshold: number): void;
  onValidationError(message: string): void;
}

function validateThreshold(draft: string): { value: number } | { error: string } {
  const value = draft.trim();
  if (value === "") return { error: "Enter a threshold from 0 through 100." };
  if (!/^\d+(?:\.\d+)?$/.test(value)) return { error: "Enter a complete decimal threshold." };
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return { error: "Enter a finite threshold." };
  if (numeric < 0 || numeric > 100) return { error: "Threshold must be from 0 through 100." };
  return { value: numeric };
}

function EventControl({
  checked,
  label,
  threshold,
  thresholdLabel,
  onCheckedChange,
  onThresholdChange,
  onValidationError,
}: EventControlProps) {
  const thresholdName = label === "Abrupt displacement" ? "Displacement" : label;
  const [draft, setDraft] = useState(String(threshold));
  const debounceRef = useRef<number | null>(null);
  const lastCommittedRef = useRef(threshold);
  const draftValidation = validateThreshold(draft);

  useEffect(() => {
    setDraft(String(threshold));
    lastCommittedRef.current = threshold;
  }, [threshold]);

  useEffect(() => () => {
    if (debounceRef.current !== null) window.clearTimeout(debounceRef.current);
  }, []);

  function commit(nextDraft: string, revertInvalid: boolean): void {
    if (debounceRef.current !== null) {
      window.clearTimeout(debounceRef.current);
      debounceRef.current = null;
    }
    const result = validateThreshold(nextDraft);
    if ("error" in result) {
      onValidationError(`${thresholdName}: ${result.error}`);
      if (revertInvalid) setDraft(String(threshold));
      return;
    }
    if (result.value === lastCommittedRef.current) return;
    lastCommittedRef.current = result.value;
    onValidationError("");
    onThresholdChange(result.value);
  }

  function schedule(nextDraft: string): void {
    if (debounceRef.current !== null) window.clearTimeout(debounceRef.current);
    if ("error" in validateThreshold(nextDraft)) return;
    onValidationError("");
    debounceRef.current = window.setTimeout(() => commit(nextDraft, false), 300);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>): void {
    if (event.key === "Enter") {
      event.preventDefault();
      commit(draft, false);
    }
  }

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
          aria-valuemax={100}
          aria-valuemin={0}
          aria-valuenow={"value" in draftValidation ? draftValidation.value : undefined}
          aria-valuetext={draft}
          disabled={!checked}
          inputMode="decimal"
          onBlur={() => commit(draft, true)}
          onChange={(event) => {
            setDraft(event.target.value);
            schedule(event.target.value);
          }}
          onKeyDown={handleKeyDown}
          role="spinbutton"
          type="text"
          value={draft}
        />
      </label>
    </div>
  );
}

type ActivityFamily = "displacement" | "scaleChange" | "proximity";

interface ActivityControlProps {
  family: ActivityFamily;
  label: string;
  enabled: boolean;
  activities: readonly EventActivity[];
  rawMatchCount: number;
  densityWarning: string | null;
  frame: number;
  onSeek(frame: number): void;
}

function activityReason(
  label: string,
  enabled: boolean,
  activityCount: number,
  direction: -1 | 1,
  target: number | null,
): string {
  if (!enabled) return `${label} is off.`;
  if (activityCount === 0) return `No matching ${label.toLowerCase()} activity.`;
  if (target === null) return `No ${direction === -1 ? "previous" : "next"} ${label.toLowerCase()} activity.`;
  return "";
}

function ActivityControl({
  family,
  label,
  enabled,
  activities,
  rawMatchCount,
  densityWarning,
  frame,
  onSeek,
}: ActivityControlProps) {
  const previousTarget = enabled ? activityNavigationTarget(activities, frame, -1) : null;
  const nextTarget = enabled ? activityNavigationTarget(activities, frame, 1) : null;
  const previousReason = activityReason(label, enabled, activities.length, -1, previousTarget);
  const nextReason = activityReason(label, enabled, activities.length, 1, nextTarget);
  const previousReasonId = `${family}-previous-activity-reason`;
  const nextReasonId = `${family}-next-activity-reason`;
  return (
    <section className="event-activity" aria-label={`${label} activity controls`}>
      <p className="event-activity__count">
        <strong>{label}</strong>: {activities.length} activit{activities.length === 1 ? "y" : "ies"} / {rawMatchCount} raw match{rawMatchCount === 1 ? "" : "es"}.
      </p>
      <div className="event-activity__actions">
        <button
          aria-describedby={previousReason === "" ? undefined : previousReasonId}
          disabled={previousTarget === null}
          onClick={() => previousTarget !== null && onSeek(previousTarget)}
          type="button"
        >
          Previous {label} activity
        </button>
        <button
          aria-describedby={nextReason === "" ? undefined : nextReasonId}
          disabled={nextTarget === null}
          onClick={() => nextTarget !== null && onSeek(nextTarget)}
          type="button"
        >
          Next {label} activity
        </button>
      </div>
      <p className="event-activity__reason" id={previousReasonId}>{previousReason}</p>
      <p className="event-activity__reason" id={nextReasonId}>{nextReason}</p>
      {densityWarning !== null && <p className="event-activity__warning">{densityWarning}</p>}
    </section>
  );
}

function LowConfidenceControl({
  meaningful,
  status,
  diagnostic,
  frames,
  frame,
  onSeek,
}: {
  meaningful: boolean;
  status: "meaningful" | "absent" | "constant" | "sentinel";
  diagnostic: string | null;
  frames: readonly number[];
  frame: number;
  onSeek(frame: number): void;
}) {
  const previousTarget = meaningful ? lowConfidenceNavigationTarget(frames, frame, -1) : null;
  const nextTarget = meaningful ? lowConfidenceNavigationTarget(frames, frame, 1) : null;
  const unavailable = diagnostic ?? `Low confidence unavailable: ${status}.`;
  const reason = (direction: -1 | 1, target: number | null) => {
    if (!meaningful) return unavailable;
    if (frames.length === 0) return "No matching low-confidence observation.";
    if (target === null) return `No ${direction === -1 ? "previous" : "next"} low-confidence observation.`;
    return "";
  };
  const previousReason = reason(-1, previousTarget);
  const nextReason = reason(1, nextTarget);
  return (
    <section className="event-activity" aria-label="Low-confidence observation controls">
      <p className="event-activity__count"><strong>Low confidence</strong>: {frames.length} observation{frames.length === 1 ? "" : "s"}.</p>
      <div className="event-activity__actions">
        <button
          aria-describedby={previousReason === "" ? undefined : "low-confidence-previous-reason"}
          disabled={previousTarget === null}
          onClick={() => previousTarget !== null && onSeek(previousTarget)}
          type="button"
        >
          Previous low-confidence observation
        </button>
        <button
          aria-describedby={nextReason === "" ? undefined : "low-confidence-next-reason"}
          disabled={nextTarget === null}
          onClick={() => nextTarget !== null && onSeek(nextTarget)}
          type="button"
        >
          Next low-confidence observation
        </button>
      </div>
      <p className="event-activity__reason" id="low-confidence-previous-reason">{previousReason}</p>
      <p className="event-activity__reason" id="low-confidence-next-reason">{nextReason}</p>
    </section>
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
  eventSettings,
  eventStatus,
  eventError,
  onSeek,
  onExit,
  onModeChange,
  onContextCountChange,
  onEventSettingsChange,
  onRetryEvents,
  trajectoryMode,
  onTrajectoryModeChange,
}: FocusReviewProps) {
  const [validationError, setValidationError] = useState("");
  if (focusTarget === null) return null;
  if (focusStatus === "error") {
    return <p className="focus-state focus-state--error" role="alert">Track evidence could not be loaded.</p>;
  }
  if (focusStatus === "loading" || evidence === null || filmstrip === null || events === null) {
    if (eventStatus === "error") {
      return <p className="focus-state focus-state--error" role="alert">Event data could not be loaded.</p>;
    }
    return <p className="focus-state" role="status">Loading track evidence</p>;
  }
  const current = evidence.observations.find((observation) => observation.frame === frame) ?? null;
  const previous = evidence.observations.findLast((observation) => observation.frame < frame) ?? null;
  const next = evidence.observations.find((observation) => observation.frame > frame) ?? null;
  const activities = deriveEventActivities(events);
  const activeActivities = [
    ...(eventSettings.displacement_enabled ? activities.displacement : []),
    ...(eventSettings.scale_change_enabled ? activities.scaleChange : []),
    ...(eventSettings.close_interaction_enabled ? activities.proximity : []),
  ];
  const currentActivities = activitiesAtFrame(activeActivities, frame);
  const lowConfidenceFrames = events.confidence.meaningful
    ? events.low_confidence_observations.map((observation) => observation.frame)
    : [];
  const navigation = [
    ["Previous observation", adjacentFrame(evidence.observation_frames, frame, -1)],
    ["Next observation", adjacentFrame(evidence.observation_frames, frame, 1)],
    ["Previous gap", adjacentFrame(gapStartFrames(evidence.gaps), frame, -1)],
    ["Next gap", adjacentFrame(gapStartFrames(evidence.gaps), frame, 1)],
  ] as const;
  const trajectorySimplified = buildFocusOverlayPlan(evidence.observations, frame, { mode: trajectoryMode, maximumVertices: 512 }).simplified;

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
        <fieldset className="trajectory-control">
          <legend>Trajectory evidence</legend>
          <label>
            <input checked={trajectoryMode === "past"} name="trajectory-mode" onChange={() => onTrajectoryModeChange("past")} type="radio" />
            <span>Past through current</span>
          </label>
          <label>
            <input checked={trajectoryMode === "complete"} name="trajectory-mode" onChange={() => onTrajectoryModeChange("complete")} type="radio" />
            <span>Complete track (future dashed)</span>
          </label>
        </fieldset>
      </div>
      <fieldset className="event-controls">
        <legend>Optional review events</legend>
        <EventControl
          checked={eventSettings.displacement_enabled}
          label="Abrupt displacement"
          onCheckedChange={(checked) => onEventSettingsChange((settings) => ({ ...settings, displacement_enabled: checked }))}
          onThresholdChange={(threshold) => onEventSettingsChange((settings) => ({ ...settings, displacement_threshold: threshold }))}
          onValidationError={setValidationError}
          threshold={eventSettings.displacement_threshold}
          thresholdLabel="Center movement"
        />
        <EventControl
          checked={eventSettings.scale_change_enabled}
          label="Scale change"
          onCheckedChange={(checked) => onEventSettingsChange((settings) => ({ ...settings, scale_change_enabled: checked }))}
          onThresholdChange={(threshold) => onEventSettingsChange((settings) => ({ ...settings, scale_change_threshold: threshold }))}
          onValidationError={setValidationError}
          threshold={eventSettings.scale_change_threshold}
          thresholdLabel="Height change"
        />
        <EventControl
          checked={eventSettings.close_interaction_enabled}
          label="Close interaction"
          onCheckedChange={(checked) => onEventSettingsChange((settings) => ({ ...settings, close_interaction_enabled: checked }))}
          onThresholdChange={(threshold) => onEventSettingsChange((settings) => ({ ...settings, close_interaction_threshold: threshold }))}
          onValidationError={setValidationError}
          threshold={eventSettings.close_interaction_threshold}
          thresholdLabel="Edge proximity"
        />
      </fieldset>
      <div aria-live="polite" className="event-status" role="status">
        {eventStatus === "updating" && "Updating review events"}
        {eventStatus === "error" && (
          <>
            {eventError} <button onClick={onRetryEvents} type="button">Retry event refresh</button>
          </>
        )}
        {eventStatus !== "error" && validationError}
      </div>
      <section className="event-activities" aria-label="Review event activities">
        <ActivityControl
          activities={activities.displacement}
          densityWarning={eventSettings.displacement_enabled && eventSettings.displacement_threshold === 0
            ? "A zero displacement threshold matches every valid transition because the backend uses >=."
            : null}
          enabled={eventSettings.displacement_enabled}
          family="displacement"
          frame={frame}
          label="Displacement"
          onSeek={onSeek}
          rawMatchCount={events.displacement_events.length}
        />
        <ActivityControl
          activities={activities.scaleChange}
          densityWarning={eventSettings.scale_change_enabled && eventSettings.scale_change_threshold === 0
            ? "A zero scale threshold matches every valid transition because the backend uses >=."
            : null}
          enabled={eventSettings.scale_change_enabled}
          family="scaleChange"
          frame={frame}
          label="Scale change"
          onSeek={onSeek}
          rawMatchCount={events.scale_change_events.length}
        />
        <ActivityControl
          activities={activities.proximity}
          densityWarning={eventSettings.close_interaction_enabled && eventSettings.close_interaction_threshold === 0
            ? "A zero proximity threshold can match touching or overlapping boxes because edge distance can be zero."
            : null}
          enabled={eventSettings.close_interaction_enabled}
          family="proximity"
          frame={frame}
          label="Proximity"
          onSeek={onSeek}
          rawMatchCount={events.close_interaction_events.length}
        />
        <LowConfidenceControl
          diagnostic={events.confidence.diagnostic?.message ?? null}
          frame={frame}
          frames={lowConfidenceFrames}
          meaningful={events.confidence.meaningful}
          onSeek={onSeek}
          status={events.confidence.status}
        />
      </section>
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
      {currentActivities.length > 0 && (
        <section className="current-event-activities" aria-label="Current event activities">
          <h3>Current activities</h3>
          <ul>
            {currentActivities.map((activity) => (
              <li key={`${activity.family}-${activity.competitorTrackId ?? "track"}-${activity.startFrame}-${activity.endFrame}`}>
                {activity.family === "scaleChange" ? "Scale change" : activity.family[0].toUpperCase() + activity.family.slice(1)}
                {activity.competitorTrackId === undefined ? "" : ` with competitor ${activity.competitorTrackId}`}
                {`: frames ${activity.startFrame}-${activity.endFrame}; anchor ${activity.anchorFrame}; severity ${activity.severity}; ${activity.rawMatchCount} raw match${activity.rawMatchCount === 1 ? "" : "es"}.`}
              </li>
            ))}
          </ul>
        </section>
      )}
      {trajectorySimplified && <p className="trajectory-note" role="status">Trajectory is simplified to 512 evidence vertices; disconnected evidence markers preserve gaps.</p>}
      <TrackTimeline evidence={evidence} events={events} frame={frame} frameCount={source.frame_count} onSeek={onSeek} />
      <TrackFilmstrip filmstrip={filmstrip} onSeek={onSeek} source={source} />
    </section>
  );
}
