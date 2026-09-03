import { useEffect, useState, type KeyboardEvent as ReactKeyboardEvent } from "react";

import { fetchObservationCrop, fetchTrackEvidence, type Observation, type SourceMetadata } from "./api";

interface CandidateChooserProps {
  source: SourceMetadata;
  candidates: Observation[];
  activeRowIndex: number;
  onActivate(rowIndex: number): void;
  onConfirm(): void;
  onFocus(observation: Observation): void;
}

interface CropState {
  status: "loading" | "ready" | "error";
  dataUrl?: string;
}

function CandidateCrop({ source, observation }: { source: SourceMetadata; observation: Observation }) {
  const [state, setState] = useState<CropState>({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setState({ status: "loading" });
    fetchObservationCrop(source, observation.row_index, controller.signal)
      .then((crop) => {
        if (!active) return;
        if (
          crop.source_key !== source.source_key ||
          crop.source_hash !== source.source_hash ||
          crop.frame !== observation.frame ||
          crop.row_index !== observation.row_index ||
          crop.row_hash !== observation.row_hash
        ) {
          throw new Error("Crop identity did not match the pinned observation");
        }
        setState({ status: "ready", dataUrl: `data:${crop.media_type};base64,${crop.image_base64}` });
      })
      .catch((error: unknown) => {
        if (active && !(error instanceof DOMException && error.name === "AbortError")) {
          setState({ status: "error" });
        }
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [observation.row_hash, observation.row_index, source]);

  if (state.status === "loading") {
    return <span className="candidate-card__crop-state">Loading crop</span>;
  }
  if (state.status === "error" || state.dataUrl === undefined) {
    return <span className="candidate-card__crop-state candidate-card__crop-state--error">Crop unavailable</span>;
  }
  return (
    <img
      alt={`Current frame crop for observation row ${observation.row_index}`}
      className="candidate-card__crop"
      src={state.dataUrl}
    />
  );
}

function TrackedCandidateEvidence({ source, observation }: { source: SourceMetadata; observation: Observation }) {
  const [temporal, setTemporal] = useState<{ earlier: Observation | null; later: Observation | null } | null>(null);
  useEffect(() => {
    if (observation.usable_track_id === null) return;
    const controller = new AbortController();
    let active = true;
    setTemporal(null);
    fetchTrackEvidence(source, observation.usable_track_id, observation.row_index, controller.signal)
      .then((evidence) => {
        if (
          !active ||
          evidence.source_key !== source.source_key ||
          evidence.source_hash !== source.source_hash ||
          evidence.track_id !== observation.usable_track_id
        ) return;
        const earlier = evidence.observations.filter((item) => item.frame < observation.frame);
        const later = evidence.observations.filter((item) => item.frame > observation.frame);
        setTemporal({
          earlier: earlier.length === 0 ? null : earlier[Math.floor((earlier.length - 1) / 2)],
          later: later.length === 0 ? null : later[Math.floor((later.length - 1) / 2)],
        });
      })
      .catch((error: unknown) => {
        if (active && !(error instanceof DOMException && error.name === "AbortError")) {
          setTemporal({ earlier: null, later: null });
        }
      });
    return () => { active = false; controller.abort(); };
  }, [observation, source]);

  if (temporal === null) return <span className="candidate-card__temporal-state">Loading track evidence</span>;
  return (
    <span className="candidate-card__temporal">
      {temporal.earlier === null ? <span>Earlier unavailable</span> : <span><CandidateCrop observation={temporal.earlier} source={source} />Earlier / {temporal.earlier.frame}</span>}
      {temporal.later === null ? <span>Later unavailable</span> : <span><CandidateCrop observation={temporal.later} source={source} />Later / {temporal.later.frame}</span>}
    </span>
  );
}

export function CandidateChooser({
  source,
  candidates,
  activeRowIndex,
  onActivate,
  onConfirm,
  onFocus,
}: CandidateChooserProps) {
  function handleListKeyDown(event: ReactKeyboardEvent<HTMLDivElement>): void {
    if (event.target !== event.currentTarget) return;
    const activeIndex = candidates.findIndex((candidate) => candidate.row_index === activeRowIndex);
    let nextIndex: number | null = null;
    if (event.key === "ArrowDown" || event.key === "ArrowRight") {
      nextIndex = (activeIndex + 1) % candidates.length;
    } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
      nextIndex = (activeIndex - 1 + candidates.length) % candidates.length;
    } else if (event.key === "Home") {
      nextIndex = 0;
    } else if (event.key === "End") {
      nextIndex = candidates.length - 1;
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      const active = candidates[activeIndex];
      if (active !== undefined) {
        onConfirm();
        onFocus(active);
      }
      return;
    }
    if (nextIndex !== null) {
      event.preventDefault();
      const next = candidates[nextIndex];
      if (next !== undefined) {
        onActivate(next.row_index);
        document.getElementById(`candidate-${next.row_index}`)?.scrollIntoView?.({ block: "nearest" });
      }
    }
  }

  return (
    <section className="candidate-chooser" aria-label="Pinned observation candidates">
      <div className="candidate-chooser__heading">
        <div>
          <p className="candidate-chooser__label">Pinned overlap</p>
          <h2>{candidates.length} observation candidates</h2>
        </div>
        <p aria-live="polite" className="candidate-chooser__position">
          {candidates.findIndex((candidate) => candidate.row_index === activeRowIndex) + 1} / {candidates.length}
        </p>
      </div>
      <div
        aria-activedescendant={`candidate-${activeRowIndex}`}
        aria-label="Observation candidates"
        className="candidate-chooser__list"
        onKeyDown={handleListKeyDown}
        role="listbox"
        tabIndex={0}
      >
        {candidates.map((observation, index) => {
          const active = observation.row_index === activeRowIndex;
          return (
            <button
              aria-label={`Candidate ${index + 1}, observation row ${observation.row_index}`}
              aria-selected={active}
              className="candidate-card"
              id={`candidate-${observation.row_index}`}
              key={observation.row_index}
              onClick={() => {
                onActivate(observation.row_index);
                onConfirm();
                onFocus(observation);
              }}
              onFocus={() => onActivate(observation.row_index)}
              onMouseEnter={() => onActivate(observation.row_index)}
              role="option"
              type="button"
            >
              <span className="candidate-card__number">{index + 1}</span>
              <CandidateCrop observation={observation} source={source} />
              <span className="candidate-card__detail">
                <strong>Observation row {observation.row_index}</strong>
                <span>
                  {observation.score === null ? "Score not defined" : `Score ${observation.score.toFixed(3)}`}
                </span>
                <span>{observation.usable_track_id === null ? "Current frame only" : `Current / ${observation.frame}`}</span>
              </span>
              {observation.usable_track_id !== null && <TrackedCandidateEvidence observation={observation} source={source} />}
            </button>
          );
        })}
      </div>
      <p aria-live="polite" className="visually-hidden">{candidates.length} candidates pinned.</p>
    </section>
  );
}