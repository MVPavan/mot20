import { useEffect, useMemo, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from "react";

import {
  fetchTrackContext,
  fetchTimelineEvents,
  fetchFrameObservations,
  fetchSequences,
  fetchTrackEvidence,
  fetchTrackFilmstrip,
  frameImageUrl,
  type ContextCompetitor,
  type EventSettings,
  type FilmstripResponse,
  type Observation,
  type SequenceListResponse,
  type SourceMetadata,
  type TimelineEventsResponse,
  type TrackEvidenceResponse,
} from "./api";
import { BitmapFrameLoader, BitmapLru } from "./bitmapCache";
import { FrameViewport } from "./FrameViewport";
import { FocusReview } from "./FocusReview";
import { SourceSetup } from "./SourceSetup";
import { TrackSearch } from "./TrackSearch";
import { stabilizeContextCompetitors } from "./contextOverlayPlan";

const TEST_POLICY_TEXT =
  "This source is local test-adapted development material and is not a held-out benchmark result.";
const PREFETCH_BATCH_SIZE = 12;
const DEFAULT_EVENT_SETTINGS: EventSettings = {
  displacement_enabled: false,
  displacement_threshold: 0.5,
  displacement_operator: "greater_than_or_equal",
  scale_change_enabled: false,
  scale_change_threshold: 0.5,
  scale_change_operator: "greater_than_or_equal",
  close_interaction_enabled: false,
  close_interaction_threshold: 0.25,
  close_interaction_operator: "less_than_or_equal",
};

function sameEventSettings(left: EventSettings, right: EventSettings): boolean {
  return (
    left.displacement_enabled === right.displacement_enabled &&
    left.displacement_threshold === right.displacement_threshold &&
    left.displacement_operator === right.displacement_operator &&
    left.scale_change_enabled === right.scale_change_enabled &&
    left.scale_change_threshold === right.scale_change_threshold &&
    left.scale_change_operator === right.scale_change_operator &&
    left.close_interaction_enabled === right.close_interaction_enabled &&
    left.close_interaction_threshold === right.close_interaction_threshold &&
    left.close_interaction_operator === right.close_interaction_operator
  );
}

function provenanceValue(value: string | number | null): string {
  return value === null ? "Not provided" : String(value);
}

export default function App() {
  const [catalog, setCatalog] = useState<SequenceListResponse | null>(null);
  const [catalogError, setCatalogError] = useState("");
  const [selectedKey, setSelectedKey] = useState("");
  const [frameDraft, setFrameDraft] = useState("1");

  useEffect(() => {
    const controller = new AbortController();
    fetchSequences(controller.signal)
      .then(setCatalog)
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setCatalogError(error instanceof Error ? error.message : "Source metadata request failed");
        }
      });
    return () => controller.abort();
  }, []);

  const source = catalog?.sources.find((candidate) => candidate.source_key === selectedKey) ?? null;
  const frame = Number(frameDraft);
  const frameIsValid =
    source !== null && Number.isInteger(frame) && frame >= 1 && frame <= source.frame_count;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="topbar__eyebrow">Dense sequence review</p>
          <h1>MOT20 Frame Review</h1>
        </div>
        <div className="source-control">
          <label htmlFor="source">Source</label>
          <select
            id="source"
            onChange={(event) => {
              setSelectedKey(event.target.value);
              setFrameDraft("1");
            }}
            value={selectedKey}
          >
            <option value="">Select a local source</option>
            {catalog?.sources.map((candidate) => (
              <option key={candidate.source_key} value={candidate.source_key}>
                {candidate.sequence} / {candidate.source_key}
              </option>
            ))}
          </select>
        </div>
      </header>

      <SourceSetup
        onLoaded={(nextCatalog) => {
          setCatalog(nextCatalog);
          setCatalogError("");
          setSelectedKey(nextCatalog.sources[0]?.source_key ?? "");
          setFrameDraft("1");
        }}
      />

      {catalog === null && catalogError === "" && (
        <p className="system-state" role="status">
          Loading local source metadata
        </p>
      )}
      {catalogError !== "" && (
        <p className="system-state system-state--error" role="alert">
          {catalogError}
        </p>
      )}
      {catalog?.sources.length === 0 && (
        <section className="empty-state" aria-labelledby="empty-title">
          <p className="empty-state__index">00</p>
          <div>
            <h2 id="empty-title">No local sources available</h2>
            <p>The viewer started correctly, but no configured source is currently available.</p>
          </div>
        </section>
      )}
      {catalog !== null && catalog.sources.length > 0 && source === null && (
        <section className="empty-state" aria-labelledby="select-title">
          <p className="empty-state__index">01</p>
          <div>
            <h2 id="select-title">Choose a source to inspect exact frames</h2>
            <p>{catalog.sources.length} local source{catalog.sources.length === 1 ? "" : "s"} ready.</p>
          </div>
        </section>
      )}
      {source !== null && (
        <Viewer
          frameDraft={frameDraft}
          key={`${source.source_key}:${source.source_hash}`}
          setFrameDraft={setFrameDraft}
          source={source}
        />
      )}
    </main>
  );
}

interface ViewerProps {
  source: SourceMetadata;
  frameDraft: string;
  setFrameDraft(value: string): void;
}

function Viewer({ source, frameDraft, setFrameDraft }: ViewerProps) {
  const frame = Number(frameDraft);
  const frameIsValid = Number.isInteger(frame) && frame >= 1 && frame <= source.frame_count;
  const cacheRef = useRef<BitmapLru | null>(null);
  if (cacheRef.current === null) {
    cacheRef.current = new BitmapLru();
  }
  const cache = cacheRef.current;
  const loaderRef = useRef<BitmapFrameLoader | null>(null);
  if (loaderRef.current === null) {
    loaderRef.current = new BitmapFrameLoader(cache);
  }
  const loader = loaderRef.current;
  const [observations, setObservations] = useState<Observation[]>([]);
  const [observationStatus, setObservationStatus] = useState<"loading" | "ready" | "error">(
    "loading",
  );
  const [playing, setPlaying] = useState(false);
  const [readyFrame, setReadyFrame] = useState<number | null>(null);
  const [focusTarget, setFocusTarget] = useState<{ trackId: number; confirmedRowIndex: number } | null>(null);
  const [trackEvidence, setTrackEvidence] = useState<TrackEvidenceResponse | null>(null);
  const [filmstrip, setFilmstrip] = useState<FilmstripResponse | null>(null);
  const [timelineEvents, setTimelineEvents] = useState<TimelineEventsResponse | null>(null);
  const [focusStatus, setFocusStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [reviewMode, setReviewMode] = useState<"focus" | "context">("focus");
  const [trajectoryMode, setTrajectoryMode] = useState<"past" | "complete">("past");
  const [contextCount, setContextCount] = useState(3);
  const [contextCompetitors, setContextCompetitors] = useState<ContextCompetitor[]>([]);
  const [contextStatus, setContextStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [eventSettings, setEventSettings] = useState<EventSettings>(DEFAULT_EVENT_SETTINGS);
  const [eventStatus, setEventStatus] = useState<"idle" | "loading" | "updating" | "error">("idle");
  const [eventError, setEventError] = useState("");
  const [eventRetry, setEventRetry] = useState(0);
  const contextTrackIdsRef = useRef<number[]>([]);
  const eventRequestIdRef = useRef(0);
  const skipEventSettingsRequestRef = useRef(false);
  const lastSuccessfulEventSettingsRef = useRef<EventSettings>(DEFAULT_EVENT_SETTINGS);
  const missingProvenance = Object.entries(source.provenance)
    .filter(([, value]) => value === null)
    .map(([field]) => field.replaceAll("_", " "));

  useEffect(() => () => {
    loader.dispose();
    cache.clear();
  }, [cache, loader]);

  useEffect(() => {
    contextTrackIdsRef.current = [];
    setContextCompetitors([]);
    setContextStatus("idle");
  }, [focusTarget?.trackId]);

  useEffect(() => {
    if (focusTarget === null) {
      setTrackEvidence(null);
      setFilmstrip(null);
      setTimelineEvents(null);
      setFocusStatus("idle");
      setEventStatus("idle");
      setEventError("");
      return;
    }
    const controller = new AbortController();
    let active = true;
    setFocusStatus("loading");
    setTrackEvidence(null);
    setFilmstrip(null);
    setTimelineEvents(null);
    Promise.all([
      fetchTrackEvidence(source, focusTarget.trackId, focusTarget.confirmedRowIndex, controller.signal),
      fetchTrackFilmstrip(source, focusTarget.trackId, focusTarget.confirmedRowIndex, controller.signal),
    ])
      .then(([evidence, nextFilmstrip]) => {
        const responses = [evidence, nextFilmstrip];
        if (!active) return;
        if (
          responses.some(
            (response) =>
              response.source_key !== source.source_key ||
              response.source_hash !== source.source_hash ||
              response.track_id !== focusTarget.trackId,
          )
        ) {
          throw new Error("Track evidence response identity did not match the active Focus source");
        }
        setTrackEvidence(evidence);
        setFilmstrip(nextFilmstrip);
        setFocusStatus("ready");
      })
      .catch((error: unknown) => {
        if (active && !(error instanceof DOMException && error.name === "AbortError")) {
          setTrackEvidence(null);
          setFilmstrip(null);
          setFocusStatus("error");
        }
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [focusTarget, source]);

  useEffect(() => {
    if (focusTarget === null) {
      eventRequestIdRef.current += 1;
      return;
    }
    if (skipEventSettingsRequestRef.current) {
      skipEventSettingsRequestRef.current = false;
      return;
    }
    const controller = new AbortController();
    const requestId = ++eventRequestIdRef.current;
    const requestedSettings = eventSettings;
    setEventStatus(timelineEvents === null ? "loading" : "updating");
    setEventError("");
    fetchTimelineEvents(source, focusTarget.trackId, requestedSettings, controller.signal)
      .then((events) => {
        if (requestId !== eventRequestIdRef.current) return;
        if (
          events.source_key !== source.source_key ||
          events.source_hash !== source.source_hash ||
          events.track_id !== focusTarget.trackId
        ) {
          throw new Error("Timeline event response identity did not match the active Focus source");
        }
        lastSuccessfulEventSettingsRef.current = events.settings;
        setTimelineEvents(events);
        setEventSettings((current) => (
          sameEventSettings(current, events.settings) ? current : events.settings
        ));
        setEventStatus("idle");
      })
      .catch((error: unknown) => {
        if (
          requestId !== eventRequestIdRef.current ||
          (error instanceof DOMException && error.name === "AbortError")
        ) {
          return;
        }
        if (!sameEventSettings(requestedSettings, lastSuccessfulEventSettingsRef.current)) {
          skipEventSettingsRequestRef.current = true;
        }
        setEventSettings((current) => (
          sameEventSettings(current, lastSuccessfulEventSettingsRef.current)
            ? current
            : lastSuccessfulEventSettingsRef.current
        ));
        setEventStatus("error");
        setEventError("Event refresh failed; showing the last successful settings.");
      });
    return () => controller.abort();
  }, [eventRetry, eventSettings, focusTarget, source]);

  useEffect(() => {
    if (focusTarget === null || reviewMode !== "context" || !frameIsValid) {
      setContextCompetitors([]);
      setContextStatus("idle");
      return;
    }
    const controller = new AbortController();
    let active = true;
    setContextStatus("loading");
    fetchTrackContext(source, focusTarget.trackId, frame, 8, controller.signal)
      .then((response) => {
        if (!active) return;
        if (
          response.source_key !== source.source_key ||
          response.source_hash !== source.source_hash ||
          response.track_id !== focusTarget.trackId ||
          response.window.center_frame !== frame
        ) {
          throw new Error("Track context response identity did not match the active Focus frame");
        }
        const stable = stabilizeContextCompetitors(
          contextTrackIdsRef.current,
          response.competitors,
          contextCount,
        );
        contextTrackIdsRef.current = stable.map((competitor) => competitor.track_id);
        setContextCompetitors(stable);
        setContextStatus("ready");
      })
      .catch((error: unknown) => {
        if (active && !(error instanceof DOMException && error.name === "AbortError")) {
          setContextCompetitors([]);
          setContextStatus("error");
        }
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [contextCount, focusTarget, frame, frameIsValid, reviewMode, source]);

  useEffect(() => {
    if (!frameIsValid) {
      setObservations([]);
      return;
    }
    const controller = new AbortController();
    let active = true;
    setObservationStatus("loading");
    fetchFrameObservations(source, frame, controller.signal)
      .then((response) => {
        if (!active) return;
        if (
          response.source_key !== source.source_key ||
          response.source_hash !== source.source_hash ||
          response.frame !== frame
        ) {
          throw new Error("Observation response identity did not match the requested frame");
        }
        setObservations(response.observations);
        setObservationStatus("ready");
      })
      .catch((error: unknown) => {
        if (active && !(error instanceof DOMException && error.name === "AbortError")) {
          setObservations([]);
          setObservationStatus("error");
        }
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [frame, frameIsValid, source]);

  useEffect(() => {
    if (!playing || !frameIsValid || readyFrame !== frame) {
      return;
    }
    const timer = window.setTimeout(() => {
      if (frame >= source.frame_count) {
        setPlaying(false);
      } else {
        setFrameDraft(String(frame + 1));
      }
    }, window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? 1000 : 200);
    return () => window.clearTimeout(timer);
  }, [frame, frameIsValid, playing, readyFrame, setFrameDraft, source.frame_count]);

  function seek(nextFrame: number): void {
    const bounded = Math.min(source.frame_count, Math.max(1, Math.trunc(nextFrame)));
    setFrameDraft(String(bounded));
  }

  function updateEventSettings(update: (settings: EventSettings) => EventSettings): void {
    eventRequestIdRef.current += 1;
    setEventSettings(update);
  }

  function handleWorkspaceKey(event: ReactKeyboardEvent<HTMLElement>): void {
    if (event.key === "Escape" && focusTarget !== null) {
      event.preventDefault();
      setFocusTarget(null);
      return;
    }
    if (
      event.target instanceof HTMLInputElement ||
      event.target instanceof HTMLSelectElement ||
      event.target instanceof HTMLButtonElement
    ) {
      return;
    }
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      const amount = event.shiftKey ? 10 : 1;
      seek(frame + (event.key === "ArrowLeft" ? -amount : amount));
    } else if (event.key === " ") {
      event.preventDefault();
      setPlaying((value) => !value);
    }
  }

  const prefetchBatch = frameIsValid ? Math.floor((frame - 1) / PREFETCH_BATCH_SIZE) : -1;
  const prefetch = useMemo(() => {
    if (prefetchBatch < 0) return [];
    const firstFrame = prefetchBatch * PREFETCH_BATCH_SIZE + 2;
    return Array.from({ length: PREFETCH_BATCH_SIZE }, (_, index) => firstFrame + index)
      .filter((candidate) => candidate <= source.frame_count)
      .map((candidate) => ({ frame: candidate, url: frameImageUrl(source, candidate) }));
  }, [prefetchBatch, source.frame_count, source.source_hash, source.source_key]);

  return (
    <section
      className="review-workspace"
      aria-label={`${source.sequence} exact frame viewer`}
      onKeyDown={handleWorkspaceKey}
    >
      {source.policy_classification === "local_test_adapted_development_material" && (
        <p className="policy-banner" role="note">
          {TEST_POLICY_TEXT}
        </p>
      )}

      <div className="transport-band">
        <div>
          <p className="sequence-name">{source.sequence}</p>
          <p className="sequence-dimensions">
            {source.width} x {source.height} / {source.frame_rate} fps
          </p>
        </div>
        <div className="transport-controls" aria-label="Frame transport controls" role="group">
          <button aria-label="Previous 10 frames" onClick={() => seek(frame - 10)} type="button">-10</button>
          <button aria-label="Previous frame" onClick={() => seek(frame - 1)} type="button">-1</button>
          <button
            aria-label={playing ? "Pause playback" : "Start playback"}
            onClick={() => setPlaying((value) => !value)}
            type="button"
          >
            {playing ? "Pause" : "Play"}
          </button>
          <button aria-label="Next frame" onClick={() => seek(frame + 1)} type="button">+1</button>
          <button aria-label="Next 10 frames" onClick={() => seek(frame + 10)} type="button">+10</button>
        </div>
        <div className="frame-control">
          <label htmlFor="frame-number">Frame number</label>
          <div className="frame-control__input">
            <input
              aria-describedby="frame-range"
              id="frame-number"
              inputMode="numeric"
              max={source.frame_count}
              min="1"
              onChange={(event) => {
                setFrameDraft(event.target.value);
              }}
              step="1"
              type="number"
              value={frameDraft}
            />
            <span id="frame-range">of {source.frame_count}</span>
          </div>
        </div>
      </div>

      <input
        aria-label="Frame scrubber"
        className="frame-scrubber"
        max={source.frame_count}
        min="1"
        onChange={(event) => seek(Number(event.target.value))}
        step="1"
        type="range"
        value={frameIsValid ? frame : 1}
      />

      <TrackSearch
        onSelect={(evidence) => {
          const confirmed = evidence.observations.find((observation) => observation.frame === frame)
            ?? evidence.first_observation;
          setPlaying(false);
          seek(confirmed.frame);
          setFocusTarget({ trackId: evidence.track_id, confirmedRowIndex: confirmed.row_index });
        }}
        source={source}
      />

      {!frameIsValid && (
        <p className="frame-error" role="alert">
          Enter an exact one-based frame from 1 through {source.frame_count}.
        </p>
      )}
      {frameIsValid && (
        <FrameViewport
          cache={cache}
          contextCompetitors={contextCompetitors}
          frame={frame}
          imageUrl={frameImageUrl(source, frame)}
          loader={loader}
          observationStatus={observationStatus}
          observations={observations}
          contextStatus={contextStatus}
          onFrameReady={setReadyFrame}
          onPin={() => setPlaying(false)}
          focusEvidence={trackEvidence}
          focusTrackId={focusTarget?.trackId ?? null}
          trajectoryMode={trajectoryMode}
          onExitFocus={() => setFocusTarget(null)}
          onFocus={(observation) => {
            if (observation.usable_track_id !== null) {
              setPlaying(false);
              setFocusTarget({
                trackId: observation.usable_track_id,
                confirmedRowIndex: observation.row_index,
              });
            }
          }}
          prefetch={prefetch}
          reviewMode={reviewMode}
          source={source}
        />
      )}

      <FocusReview
        contextCount={contextCount}
        evidence={trackEvidence}
        eventError={eventError}
        eventSettings={eventSettings}
        eventStatus={eventStatus}
        events={timelineEvents}
        filmstrip={filmstrip}
        focusStatus={focusStatus}
        focusTarget={focusTarget}
        frame={frame}
        mode={reviewMode}
        onContextCountChange={setContextCount}
        onEventSettingsChange={updateEventSettings}
        onExit={() => setFocusTarget(null)}
        onModeChange={setReviewMode}
        onRetryEvents={() => setEventRetry((value) => value + 1)}
        onSeek={seek}
        onTrajectoryModeChange={setTrajectoryMode}
        source={source}
        trajectoryMode={trajectoryMode}
      />

      <dl className="source-status">
        <div>
          <dt>Source hash</dt>
          <dd className="hash-value">{source.source_hash}</dd>
        </div>
        <div>
          <dt>Capability</dt>
          <dd>{source.capability.track_features ? "Track tools available" : "Track tools unavailable"}</dd>
        </div>
        <div>
          <dt>ID status</dt>
          <dd>{source.capability.id_status.replaceAll("_", " ")}</dd>
        </div>
        <div>
          <dt>Producer</dt>
          <dd>{provenanceValue(source.provenance.producer)}</dd>
        </div>
        <div>
          <dt>Detector</dt>
          <dd>{provenanceValue(source.provenance.detector)}</dd>
        </div>
        <div>
          <dt>Adaptation iterations</dt>
          <dd>{provenanceValue(source.provenance.adaptation_iterations)}</dd>
        </div>
      </dl>
      {missingProvenance.length > 0 && (
        <p className="diagnostic-status">
          Provenance incomplete: {missingProvenance.join(", ")}.
        </p>
      )}
      {source.diagnostics.map((diagnostic) => (
        <p className="diagnostic-status" key={`${diagnostic.code}-${diagnostic.message}`}>
          {diagnostic.message}
        </p>
      ))}
    </section>
  );
}
