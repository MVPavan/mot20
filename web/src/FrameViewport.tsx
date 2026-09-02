import {
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";

import { type ContextCompetitor, type Observation, type SourceMetadata } from "./api";
import { BitmapLru, decodeFrame, type DecodedFrame } from "./bitmapCache";
import { containingCandidates, type HitRegion } from "./candidateRanking";
import { CandidateChooser } from "./CandidateChooser";
import {
  buildContextOverlayPlan,
  CONTEXT_STROKE_WIDTH,
  type ContextOverlayPlan,
  type Rectangle as ContextRectangle,
} from "./contextOverlayPlan";
import { buildOverlayPlan, type OverlayPlan } from "./overlayPlan";
import { buildFocusOverlayPlan, type FocusOverlayPlan } from "./focusOverlayPlan";
import { initialSelectionState, reduceSelection } from "./selectionState";
import { trackColor } from "./trackColor";
import { createViewportTransform, type Point, type ViewportTransform } from "./viewport";

interface FrameViewportProps {
  source: SourceMetadata;
  frame: number;
  imageUrl: string;
  observations: Observation[];
  observationStatus: "loading" | "ready" | "error";
  cache: BitmapLru;
  prefetch: Array<{ frame: number; url: string }>;
  onPin(): void;
  focusTrackId: number | null;
  focusEvidence: import("./api").TrackEvidenceResponse | null;
  contextCompetitors: ContextCompetitor[];
  reviewMode: "focus" | "context";
  onFocus(observation: Observation): void;
  onExitFocus(): void;
}

interface ViewportSize {
  width: number;
  height: number;
}

type ImageState =
  | { status: "loading" }
  | { status: "ready"; image: DecodedFrame }
  | { status: "error" };

type BoxCommand = Extract<OverlayPlan["commands"][number], { type: "box" }>;
type MagnifierCommand = Extract<OverlayPlan["commands"][number], { type: "magnifier" }>;

function canvasDpr(transform: ViewportTransform, cssWidth: number): number {
  return transform.canvasSize.width / cssWidth;
}

function performanceTimestamp(eventTimestamp: number): number {
  const relativeTimestamp = eventTimestamp > performance.timeOrigin
    ? eventTimestamp - performance.timeOrigin
    : eventTimestamp;
  return Math.max(0, Math.min(performance.now(), relativeTimestamp));
}

function drawBox(
  context: CanvasRenderingContext2D,
  transform: ViewportTransform,
  cssWidth: number,
  command: BoxCommand,
): void {
  const topLeft = transform.imageToScreen({ x: command.region.x1, y: command.region.y1 });
  const bottomRight = transform.imageToScreen({ x: command.region.x2, y: command.region.y2 });
  const dpr = canvasDpr(transform, cssWidth);
  const colors = {
    audit: "rgba(255, 255, 255, 0.42)",
    candidate: "#f2c94c",
    active: "#ff6b3d",
    confirmed: "#4dd09c",
  } as const;
  context.strokeStyle = colors[command.emphasis];
  context.lineWidth = (command.emphasis === "active" || command.emphasis === "confirmed" ? 3 : 1.5) * dpr;
  context.strokeRect(
    topLeft.x * dpr,
    topLeft.y * dpr,
    (bottomRight.x - topLeft.x) * dpr,
    (bottomRight.y - topLeft.y) * dpr,
  );
  if (command.number !== undefined) {
    const labelSize = 22 * dpr;
    context.fillStyle = command.emphasis === "active" ? "#ff6b3d" : "#f2c94c";
    context.fillRect(topLeft.x * dpr, topLeft.y * dpr, labelSize, labelSize);
    context.fillStyle = "#111719";
    context.font = `700 ${13 * dpr}px sans-serif`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(String(command.number), topLeft.x * dpr + labelSize / 2, topLeft.y * dpr + labelSize / 2);
  }
}

function drawMagnifier(
  context: CanvasRenderingContext2D,
  transform: ViewportTransform,
  cssWidth: number,
  cssHeight: number,
  image: DecodedFrame,
  command: MagnifierCommand,
): void {
  const dpr = canvasDpr(transform, cssWidth);
  const pointerScreen = transform.imageToScreen(command.pointerImage);
  const size = Math.max(72, Math.min(116, cssWidth - 16, cssHeight - 16));
  const left = Math.min(Math.max(8, pointerScreen.x + 18), cssWidth - size - 8);
  const top = Math.min(Math.max(8, pointerScreen.y - size - 18), cssHeight - size - 8);
  const sampleSize = Math.max(
    24,
    Math.min(96, command.region.x2 - command.region.x1, command.region.y2 - command.region.y1),
  );
  const sourceX = Math.max(0, Math.min(command.pointerImage.x - sampleSize / 2, image.width - sampleSize));
  const sourceY = Math.max(0, Math.min(command.pointerImage.y - sampleSize / 2, image.height - sampleSize));

  context.save();
  context.beginPath();
  context.rect(left * dpr, top * dpr, size * dpr, size * dpr);
  context.clip();
  context.imageSmoothingEnabled = false;
  context.drawImage(
    image,
    sourceX,
    sourceY,
    Math.min(sampleSize, image.width),
    Math.min(sampleSize, image.height),
    left * dpr,
    top * dpr,
    size * dpr,
    size * dpr,
  );
  context.restore();
  context.strokeStyle = "#ff6b3d";
  context.lineWidth = 2 * dpr;
  context.strokeRect(left * dpr, top * dpr, size * dpr, size * dpr);
}

function drawOverlay(
  context: CanvasRenderingContext2D,
  plan: OverlayPlan,
  transform: ViewportTransform,
  viewportSize: ViewportSize,
  image: DecodedFrame,
): void {
  plan.commands.forEach((command) => {
    if (command.type === "box") {
      drawBox(context, transform, viewportSize.width, command);
    } else {
      drawMagnifier(context, transform, viewportSize.width, viewportSize.height, image, command);
    }
  });
}

function drawFocusOverlay(
  context: CanvasRenderingContext2D,
  plan: FocusOverlayPlan,
  transform: ViewportTransform,
  cssWidth: number,
  color: string,
): void {
  const dpr = canvasDpr(transform, cssWidth);
  context.strokeStyle = color;
  context.fillStyle = color;
  plan.commands.forEach((command) => {
    if (command.type === "focusTrace") {
      context.beginPath();
      command.points.forEach((point, index) => {
        const screen = transform.imageToScreen(point);
        if (index === 0) context.moveTo(screen.x * dpr, screen.y * dpr);
        else context.lineTo(screen.x * dpr, screen.y * dpr);
      });
      context.lineWidth = 2 * dpr;
      context.stroke();
      return;
    }
    const geometry = command.observation.display_geometry;
    const topLeft = transform.imageToScreen({ x: geometry.x1, y: geometry.y1 });
    const bottomRight = transform.imageToScreen({ x: geometry.x2, y: geometry.y2 });
    context.lineWidth = 4 * dpr;
    context.strokeRect(
      topLeft.x * dpr,
      topLeft.y * dpr,
      (bottomRight.x - topLeft.x) * dpr,
      (bottomRight.y - topLeft.y) * dpr,
    );
    const label = `ID ${command.trackId}`;
    context.font = `700 ${14 * dpr}px sans-serif`;
    const labelWidth = context.measureText(label).width + 12 * dpr;
    const labelHeight = 24 * dpr;
    const labelY = Math.max(0, topLeft.y * dpr - labelHeight);
    context.fillRect(topLeft.x * dpr, labelY, labelWidth, labelHeight);
    context.fillStyle = "#111719";
    context.textAlign = "left";
    context.textBaseline = "middle";
    context.fillText(label, topLeft.x * dpr + 6 * dpr, labelY + labelHeight / 2);
    context.fillStyle = color;
  });
}

function focalLabelRectangle(
  context: CanvasRenderingContext2D,
  transform: ViewportTransform,
  cssWidth: number,
  observation: Observation,
  trackId: number,
  imageWidth: number,
): ContextRectangle {
  const dpr = canvasDpr(transform, cssWidth);
  const topLeft = transform.imageToScreen({
    x: observation.display_geometry.x1,
    y: observation.display_geometry.y1,
  });
  context.font = `700 ${14 * dpr}px sans-serif`;
  const labelWidth = context.measureText(`ID ${trackId}`).width / dpr + 12;
  const labelY = Math.max(0, topLeft.y - 24);
  const imageTopLeft = transform.screenToImage({ x: topLeft.x, y: labelY });
  const imageBottomRight = transform.screenToImage({ x: topLeft.x + labelWidth, y: labelY + 24 });
  const scale = transform.imageRectCss.width / imageWidth;
  const padding = CONTEXT_STROKE_WIDTH / scale / 2;
  return {
    x1: imageTopLeft.x - padding,
    y1: imageTopLeft.y - padding,
    x2: imageBottomRight.x + padding,
    y2: imageBottomRight.y + padding,
  };
}

function drawContextOverlay(
  context: CanvasRenderingContext2D,
  plan: ContextOverlayPlan,
  transform: ViewportTransform,
  cssWidth: number,
  sequence: string,
): void {
  const dpr = canvasDpr(transform, cssWidth);
  plan.commands.forEach((command) => {
    context.beginPath();
    command.segments.forEach((segment) => {
      const from = transform.imageToScreen(segment.from);
      const to = transform.imageToScreen(segment.to);
      context.moveTo(from.x * dpr, from.y * dpr);
      context.lineTo(to.x * dpr, to.y * dpr);
    });
    context.strokeStyle = trackColor(sequence, command.trackId);
    context.lineWidth = command.strokeWidth * dpr;
    context.stroke();
  });
}

export function FrameViewport({
  source,
  frame,
  imageUrl,
  observations,
  observationStatus,
  cache,
  prefetch,
  onPin,
  focusTrackId,
  focusEvidence,
  contextCompetitors,
  reviewMode,
  onFocus,
  onExitFocus,
}: FrameViewportProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const [viewportSize, setViewportSize] = useState<ViewportSize | null>(null);
  const [imageState, setImageState] = useState<ImageState>({ status: "loading" });
  const [transform, setTransform] = useState<ViewportTransform | null>(null);
  const [imageRect, setImageRect] = useState("");
  const [imageDrawCount, setImageDrawCount] = useState(0);
  const [pointerImage, setPointerImage] = useState<Point | null>(null);
  const [revealAll, setRevealAll] = useState(false);
  const [selection, dispatchSelection] = useReducer(reduceSelection, undefined, initialSelectionState);
  const selectionRef = useRef(selection);
  const pendingPointerTimestampRef = useRef<number | null>(null);
  selectionRef.current = selection;

  const regions = useMemo<HitRegion[]>(
    () =>
      observations.map((observation) => ({
        rowIndex: observation.row_index,
        ...observation.display_geometry,
      })),
    [observations],
  );
  const observationByRow = useMemo(
    () => new Map(observations.map((observation) => [observation.row_index, observation])),
    [observations],
  );

  useEffect(() => {
    dispatchSelection({ type: "reset" });
    setPointerImage(null);
    setRevealAll(false);
  }, [frame, source.source_key]);

  useEffect(() => {
    if (focusTrackId === null) {
      dispatchSelection({ type: "reset" });
    }
  }, [focusTrackId]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver(([entry]) => {
      if (entry.contentRect.width > 0 && entry.contentRect.height > 0) {
        setViewportSize({ width: entry.contentRect.width, height: entry.contentRect.height });
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const cached = cache.get(frame);
    if (cached !== undefined) {
      setImageState({ status: "ready", image: cached });
      return;
    }
    const controller = new AbortController();
    let active = true;
    setImageState({ status: "loading" });
    decodeFrame(imageUrl, controller.signal)
      .then((image) => {
        if (!active) {
          image.close?.();
          return;
        }
        cache.set(frame, image);
        setImageState({ status: "ready", image });
      })
      .catch((error: unknown) => {
        if (active && !(error instanceof DOMException && error.name === "AbortError")) {
          setImageState({ status: "error" });
        }
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [cache, frame, imageUrl]);

  useEffect(() => {
    const controllers = prefetch.flatMap(({ frame: prefetchFrame, url }) => {
      if (cache.get(prefetchFrame) !== undefined) {
        return [];
      }
      const controller = new AbortController();
      decodeFrame(url, controller.signal)
        .then((image) => cache.set(prefetchFrame, image))
        .catch(() => undefined);
      return [controller];
    });
    return () => controllers.forEach((controller) => controller.abort());
  }, [cache, prefetch]);

  useEffect(() => {
    if (imageState.status !== "ready" || viewportSize === null) {
      return;
    }
    const imageCanvas = imageCanvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;
    if (!imageCanvas || !overlayCanvas) {
      return;
    }
    const nextTransform = createViewportTransform({
      imageWidth: source.width,
      imageHeight: source.height,
      cssWidth: viewportSize.width,
      cssHeight: viewportSize.height,
      devicePixelRatio: window.devicePixelRatio || 1,
    });
    imageCanvas.width = nextTransform.canvasSize.width;
    imageCanvas.height = nextTransform.canvasSize.height;
    overlayCanvas.width = nextTransform.canvasSize.width;
    overlayCanvas.height = nextTransform.canvasSize.height;
    const context = imageCanvas.getContext("2d");
    if (context === null) {
      setImageState({ status: "error" });
      return;
    }
    context.fillStyle = "#111719";
    context.fillRect(0, 0, nextTransform.canvasSize.width, nextTransform.canvasSize.height);
    const rectangle = nextTransform.imageRectPixels;
    context.drawImage(imageState.image, rectangle.x, rectangle.y, rectangle.width, rectangle.height);
    setTransform(nextTransform);
    setImageRect([rectangle.x, rectangle.y, rectangle.width, rectangle.height].join(","));
    setImageDrawCount((count) => count + 1);
  }, [imageState, source.height, source.width, viewportSize]);

  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    if (canvas === null || transform === null || imageState.status !== "ready" || viewportSize === null) {
      return;
    }
    const animationFrame = requestAnimationFrame(() => {
      const context = canvas.getContext("2d");
      if (context === null) {
        return;
      }
      context.clearRect(0, 0, canvas.width, canvas.height);
      if (focusTrackId !== null && focusEvidence !== null) {
        const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
        const plan = buildFocusOverlayPlan(focusEvidence.observations, frame, reducedMotion ? 1 : 8);
        const focalBox = plan.commands.find((command) => command.type === "focusBox");
        const contextPlan = reviewMode === "context" && focalBox !== undefined
          ? buildContextOverlayPlan(
              focalBox.observation,
              contextCompetitors,
              frame,
              focalLabelRectangle(
                context,
                transform,
                viewportSize.width,
                focalBox.observation,
                focusTrackId,
                source.width,
              ),
            )
          : null;
        canvas.dataset.contextCommands = String(contextPlan?.commands.length ?? 0);
        canvas.dataset.contextInkArea = String(contextPlan?.contextInkArea ?? 0);
        canvas.dataset.focalArea = String(contextPlan?.focalArea ?? 0);
        canvas.dataset.labelIntersectionCount = String(contextPlan?.labelIntersectionCount ?? 0);
        canvas.dataset.focalStrokeWidth = String(contextPlan?.focalStrokeWidth ?? 4);
        canvas.dataset.contextStrokeWidth = String(contextPlan?.contextStrokeWidth ?? CONTEXT_STROKE_WIDTH);
        if (contextPlan !== null) {
          drawContextOverlay(context, contextPlan, transform, viewportSize.width, source.sequence);
        }
        canvas.dataset.overlayStrokes = String(plan.commands.length + (contextPlan?.commands.length ?? 0));
        canvas.dataset.overlayCommands = String(plan.commands.length + (contextPlan?.commands.length ?? 0));
        drawFocusOverlay(context, plan, transform, viewportSize.width, trackColor(source.sequence, focusTrackId));
      } else {
        const plan = buildOverlayPlan({ selection, regions, pointerImage, revealAll });
        canvas.dataset.overlayStrokes = String(plan.strokeCount);
        canvas.dataset.overlayCommands = String(plan.commands.length);
        drawOverlay(context, plan, transform, viewportSize, imageState.image);
        const pointerTimestamp = pendingPointerTimestampRef.current;
        if (pointerTimestamp !== null) {
          const durationMs = Math.max(0, performance.now() - pointerTimestamp);
          pendingPointerTimestampRef.current = null;
          canvas.dataset.lastPointerLatencyMs = durationMs.toFixed(3);
          window.dispatchEvent(new CustomEvent("mot20-viewer:pointer-latency", {
            detail: { durationMs, frame, candidateCount: selection.rowIndexes.length },
          }));
        }
      }
    });
    return () => cancelAnimationFrame(animationFrame);
  }, [contextCompetitors, focusEvidence, focusTrackId, frame, imageState, pointerImage, regions, revealAll, reviewMode, selection, source.sequence, transform, viewportSize]);

  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    if (canvas === null) {
      return;
    }
    const handleWheel = (event: WheelEvent) => {
      const currentSelection = selectionRef.current;
      if (currentSelection.mode === "explore" && currentSelection.rowIndexes.length > 0) {
        event.preventDefault();
        dispatchSelection({ type: "cycle", direction: event.deltaY < 0 ? -1 : 1 });
      }
    };
    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", handleWheel);
  }, []);

  function pointerCoordinates(clientX: number, clientY: number): { css: Point; image: Point } | null {
    const canvas = overlayCanvasRef.current;
    if (canvas === null || transform === null) {
      return null;
    }
    const bounds = canvas.getBoundingClientRect();
    const css = { x: clientX - bounds.left, y: clientY - bounds.top };
    const image = transform.screenToImage(css);
    if (image.x < 0 || image.x > source.width || image.y < 0 || image.y > source.height) {
      return null;
    }
    return { css, image };
  }

  function rankedRowsAt(image: Point): number[] {
    return containingCandidates(regions, image).map((candidate) => candidate.region.rowIndex);
  }

  function handlePointerMove(event: ReactPointerEvent<HTMLCanvasElement>): void {
    if (focusTrackId !== null) return;
    pendingPointerTimestampRef.current = performanceTimestamp(event.timeStamp);
    const pointer = pointerCoordinates(event.clientX, event.clientY);
    if (pointer === null) {
      setPointerImage(null);
      dispatchSelection({ type: "hover", rowIndexes: [], pointerCss: { x: -1, y: -1 } });
      return;
    }
    setPointerImage(pointer.image);
    dispatchSelection({ type: "hover", rowIndexes: rankedRowsAt(pointer.image), pointerCss: pointer.css });
  }

  function handleCanvasClick(event: ReactPointerEvent<HTMLCanvasElement>): void {
    if (focusTrackId !== null) return;
    event.currentTarget.focus();
    const pointer = pointerCoordinates(event.clientX, event.clientY);
    const rowIndexes = pointer === null ? [] : rankedRowsAt(pointer.image);
    if (selection.mode === "explore") {
      if (rowIndexes.length > 0) {
        onPin();
      }
      dispatchSelection({ type: "pin", rowIndexes });
    } else {
      if (selection.mode === "pinned" && rowIndexes.includes(selection.activeRowIndex)) {
        const observation = observationByRow.get(selection.activeRowIndex);
        if (observation !== undefined) onFocus(observation);
      }
      dispatchSelection({ type: "canvasClick", rowIndexes });
    }
  }

  function handleKeyDown(event: ReactKeyboardEvent): void {
    if (event.key.toLowerCase() === "b" && !event.repeat) {
      event.preventDefault();
      setRevealAll(true);
    } else if (event.key === "Enter" && selection.mode === "pinned") {
      event.preventDefault();
      const observation = observationByRow.get(selection.activeRowIndex);
      dispatchSelection({ type: "confirm" });
      if (observation !== undefined) onFocus(observation);
    } else if (event.key === "Escape" && focusTrackId !== null) {
      event.preventDefault();
      dispatchSelection({ type: "reset" });
      onExitFocus();
    } else if (event.key === "Escape" && selection.mode !== "explore") {
      event.preventDefault();
      dispatchSelection({ type: "escape" });
    }
  }

  const pinnedCandidates =
    selection.mode === "pinned"
      ? selection.rowIndexes.flatMap((rowIndex) => {
          const observation = observationByRow.get(rowIndex);
          return observation === undefined ? [] : [observation];
        })
      : [];
  const followReason = source.capability.track_features
    ? "Follow is unavailable until tracked review is enabled."
    : "Follow disabled: this source has sentinel-only IDs and no stable track identity.";

  return (
    <div
      className="frame-stage"
      data-active-row={selection.activeRowIndex ?? ""}
      data-candidate-count={selection.rowIndexes.length}
      data-selection-mode={selection.mode}
      onKeyDown={handleKeyDown}
      onKeyUp={(event) => {
        if (event.key.toLowerCase() === "b") setRevealAll(false);
      }}
    >
      <div className="selection-toolbar" title="Hold B while the canvas is focused to reveal all current-frame boxes">
        <span>{focusTrackId === null ? "Explore" : `${reviewMode === "context" ? "Context" : "Focus"} / ID ${focusTrackId}`}</span>
        {focusTrackId === null && <span><kbd>B</kbd> boxes</span>}
      </div>
      <div
        className="viewport"
        data-bitmap-cache-capacity={cache.capacity}
        data-bitmap-cache-closed-count={cache.closedCount}
        data-bitmap-cache-size={cache.size}
        data-frame-url={imageUrl}
        data-image-draw-count={imageDrawCount}
        data-image-rect={imageRect}
        data-testid="frame-viewport"
        ref={containerRef}
      >
        <canvas
          aria-label={`${source.sequence} frame ${frame} image`}
          className="viewport__canvas"
          data-layer="image"
          ref={imageCanvasRef}
        />
        <canvas
          aria-label={`${source.sequence} frame ${frame} observation selection canvas`}
          className="viewport__canvas viewport__canvas--overlay"
          data-layer="overlay"
          data-overlay-commands="0"
          data-overlay-strokes="0"
          onBlur={() => setRevealAll(false)}
          onClick={handleCanvasClick}
          onPointerLeave={() => {
            if (focusTrackId !== null) return;
            setPointerImage(null);
            dispatchSelection({ type: "hover", rowIndexes: [], pointerCss: { x: -1, y: -1 } });
          }}
          onPointerMove={handlePointerMove}
          ref={overlayCanvasRef}
          tabIndex={0}
        />
        {imageState.status === "loading" && (
          <p className="viewport__message" role="status">
            Loading exact frame {frame}
          </p>
        )}
        {imageState.status === "error" && (
          <p className="viewport__message viewport__message--error" role="alert">
            Exact frame {frame} could not be loaded.
          </p>
        )}
        {imageState.status === "ready" && observationStatus === "loading" && (
          <p className="viewport__observation-state" role="status">Loading observations</p>
        )}
        {observationStatus === "error" && (
          <p className="viewport__observation-state viewport__observation-state--error" role="alert">
            Frame observations could not be loaded.
          </p>
        )}
        {observationStatus === "ready" && observations.length === 0 && (
          <p className="viewport__observation-state">No observations on frame {frame}</p>
        )}
      </div>

      {selection.mode === "pinned" && pinnedCandidates.length > 0 && (
        <CandidateChooser
          activeRowIndex={selection.activeRowIndex}
          candidates={pinnedCandidates}
          onActivate={(rowIndex) => dispatchSelection({ type: "activate", rowIndex })}
          onConfirm={() => dispatchSelection({ type: "confirm" })}
          onFocus={onFocus}
          source={source}
        />
      )}
      {selection.mode === "confirmed" && focusTrackId === null && (
        <section className="confirmation-strip" aria-live="polite">
          <div>
            <p className="confirmation-strip__label">Confirmed observation</p>
            <strong>Source row {selection.activeRowIndex}</strong>
          </div>
          <button aria-describedby="follow-disabled-reason" disabled type="button">Follow</button>
          <p id="follow-disabled-reason">{followReason}</p>
        </section>
      )}
      <p aria-live="polite" className="visually-hidden">
        {selection.mode === "pinned" ? `${selection.rowIndexes.length} candidates pinned.` : ""}
        {selection.mode === "confirmed" ? followReason : ""}
      </p>
    </div>
  );
}