import { useEffect, useRef, useState, type FormEvent } from "react";

import {
  fetchSourcePathSuggestions,
  fetchSourceSelection,
  replaceSourceSelection,
  type SequenceListResponse,
  type SourcePathKind,
  type SourcePathSuggestions,
} from "./api";

export function SourceSetup({ onLoaded }: { onLoaded(catalog: SequenceListResponse): void }) {
  const [images, setImages] = useState("");
  const [annotations, setAnnotations] = useState("");
  const [browserKind, setBrowserKind] = useState<SourcePathKind | null>(null);
  const [browserResult, setBrowserResult] = useState<SourcePathSuggestions | null>(null);
  const [browserLoading, setBrowserLoading] = useState(false);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [error, setError] = useState("");
  const suggestionControllers = useRef<Record<SourcePathKind, AbortController | null>>({
    images: null,
    annotations: null,
  });

  useEffect(() => {
    const controller = new AbortController();
    fetchSourceSelection(controller.signal)
      .then((selection) => {
        setImages(selection.images ?? "");
        setAnnotations(selection.annotations ?? "");
      })
      .catch(() => undefined);
    return () => {
      controller.abort();
      suggestionControllers.current.images?.abort();
      suggestionControllers.current.annotations?.abort();
    };
  }, []);

  function updateSuggestions(kind: SourcePathKind, value: string): void {
    const controller = new AbortController();
    suggestionControllers.current[kind]?.abort();
    suggestionControllers.current[kind] = controller;
    setBrowserKind(kind);
    setBrowserLoading(true);
    fetchSourcePathSuggestions(kind, value, controller.signal)
      .then((result) => {
        setBrowserResult(result);
        setBrowserLoading(false);
      })
      .catch((requestError: unknown) => {
        if (!(requestError instanceof DOMException && requestError.name === "AbortError")) {
          setBrowserResult(null);
          setBrowserLoading(false);
        }
      });
  }

  function selectEntry(kind: SourcePathKind, path: string, entryType: "directory" | "file"): void {
    if (kind === "images") setImages(path);
    else setAnnotations(path);
    if (entryType === "directory") {
      updateSuggestions(kind, path);
    } else {
      setBrowserKind(null);
    }
  }

  function pathBrowser(kind: SourcePathKind) {
    if (browserKind !== kind) return null;
    return (
      <div
        aria-label={`Server ${kind === "images" ? "image folder" : "annotation file"} browser`}
        className="source-path-browser"
        role="dialog"
      >
        <div className="source-path-browser__header">
          <code title={browserResult?.directory}>{browserResult?.directory ?? "Server filesystem"}</code>
          <div>
            <button
              aria-label="Go to parent directory"
              disabled={browserResult?.parent == null}
              onClick={() => {
                if (browserResult?.parent != null) updateSuggestions(kind, browserResult.parent);
              }}
              title="Parent directory"
              type="button"
            >
              Up
            </button>
            <button
              aria-label="Close server browser"
              onClick={() => setBrowserKind(null)}
              title="Close"
              type="button"
            >
              Close
            </button>
          </div>
        </div>
        {kind === "images" && browserResult !== null && (
          <button
            className="source-path-browser__use"
            onClick={() => {
              setImages(browserResult.directory);
              setBrowserKind(null);
            }}
            type="button"
          >
            Use this folder
          </button>
        )}
        {browserLoading && <p role="status">Reading server directory</p>}
        {!browserLoading && browserResult !== null && browserResult.entries.length === 0 && (
          <p>No matching server paths</p>
        )}
        {!browserLoading && browserResult !== null && browserResult.entries.length > 0 && (
          <ul className="source-path-browser__entries">
            {browserResult.entries.map((entry) => (
              <li key={entry.path}>
                <button
                  onClick={() => selectEntry(kind, entry.path, entry.entry_type)}
                  title={entry.path}
                  type="button"
                >
                  <span aria-hidden="true">{entry.entry_type === "directory" ? "DIR" : "FILE"}</span>
                  <code>{entry.path}</code>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  }

  function submit(event: FormEvent): void {
    event.preventDefault();
    const selectedImages = images.trim();
    const selectedAnnotations = annotations.trim();
    if (selectedImages === "" || selectedAnnotations === "") {
      setStatus("error");
      setError("Both server paths are required.");
      return;
    }
    setStatus("loading");
    setError("");
    replaceSourceSelection(selectedImages, selectedAnnotations)
      .then((catalog) => {
        setStatus("idle");
        onLoaded(catalog);
      })
      .catch((requestError: unknown) => {
        setStatus("error");
        setError(requestError instanceof Error ? requestError.message : "Source selection failed.");
      });
  }

  return (
    <form className="source-setup" onSubmit={submit}>
      <div className="source-setup__field">
        <label htmlFor="source-images">Images folder on server</label>
        <div className="source-setup__input-row">
          <input
            autoComplete="off"
            id="source-images"
            onChange={(event) => {
              setImages(event.target.value);
              updateSuggestions("images", event.target.value);
            }}
            type="text"
            value={images}
          />
          <button
            aria-label="Browse server images"
            onClick={() => updateSuggestions("images", "/")}
            title="Browse server filesystem"
            type="button"
          >
            Browse
          </button>
        </div>
        {pathBrowser("images")}
      </div>
      <div className="source-setup__field">
        <label htmlFor="source-annotations">Predictions or ground truth file on server</label>
        <div className="source-setup__input-row">
          <input
            autoComplete="off"
            id="source-annotations"
            onChange={(event) => {
              setAnnotations(event.target.value);
              updateSuggestions("annotations", event.target.value);
            }}
            type="text"
            value={annotations}
          />
          <button
            aria-label="Browse server annotations"
            onClick={() => updateSuggestions("annotations", "/")}
            title="Browse server filesystem"
            type="button"
          >
            Browse
          </button>
        </div>
        {pathBrowser("annotations")}
      </div>
      <button disabled={status === "loading"} type="submit">
        {status === "loading" ? "Loading source" : "Load source"}
      </button>
      <span aria-live="polite" className={status === "error" ? "source-setup__error" : ""}>
        {error}
      </span>
    </form>
  );
}