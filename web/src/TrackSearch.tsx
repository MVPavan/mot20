import { useEffect, useRef, useState, type FormEvent } from "react";

import { ApiRequestError, fetchTrackSearch, type SourceMetadata, type TrackEvidenceResponse } from "./api";

export function TrackSearch({
  source,
  onSelect,
}: {
  source: SourceMetadata;
  onSelect(evidence: TrackEvidenceResponse): void;
}) {
  const [draft, setDraft] = useState("");
  const [status, setStatus] = useState<"idle" | "invalid" | "loading" | "missing" | "error">("idle");
  const controllerRef = useRef<AbortController | null>(null);
  useEffect(() => () => controllerRef.current?.abort(), []);

  if (!source.capability.track_features) {
    return <p className="track-search track-search--unavailable">Track ID search unavailable for this source.</p>;
  }

  function submit(event: FormEvent): void {
    event.preventDefault();
    const trackId = Number(draft);
    if (!Number.isInteger(trackId) || trackId < 1 || String(trackId) !== draft.trim()) {
      setStatus("invalid");
      return;
    }
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setStatus("loading");
    fetchTrackSearch(source, trackId, controller.signal)
      .then((evidence) => {
        if (
          controller.signal.aborted ||
          evidence.source_key !== source.source_key ||
          evidence.source_hash !== source.source_hash ||
          evidence.track_id !== trackId
        ) return;
        setStatus("idle");
        onSelect(evidence);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setStatus(error instanceof ApiRequestError && error.status === 404 ? "missing" : "error");
      });
  }

  return (
    <form className="track-search" onSubmit={submit}>
      <label htmlFor="track-id-search">Exact track ID</label>
      <input
        id="track-id-search"
        inputMode="numeric"
        min="1"
        onChange={(event) => {
          controllerRef.current?.abort();
          controllerRef.current = null;
          setDraft(event.target.value);
          setStatus("idle");
        }}
        type="text"
        value={draft}
      />
      <button disabled={status === "loading"} type="submit">Find track</button>
      <span aria-live="polite">
        {status === "invalid" && "Enter a positive integer track ID."}
        {status === "loading" && "Searching this sequence."}
        {status === "missing" && `Track ${draft.trim()} is not present in this sequence.`}
        {status === "error" && "Track search unavailable."}
      </span>
    </form>
  );
}