# Track Viewer Handoff

## What It Is

`track-viz/` is a self-contained, local MOT20 inspection application. It shows
exact sequence JPEGs with MOT ground-truth or tracker-result boxes and supports
ambiguous candidate selection, track search, focused temporal review, context
tracks, event markers, crops, and bounded video export.

The viewer is evidence-preserving: frames and annotation rows are read from the
selected source and are never rewritten. Generated caches, exports, browser
reports, and verification records stay under `track-viz/artifacts/` or the
ignored frontend output directories.

## Implementation Shape

- Backend: Python 3.12, FastAPI, Pydantic, Pillow, NumPy, Supervision, Uvicorn.
- Frontend: React 19, TypeScript, Vite, canvas overlays.
- Tests: pytest, Vitest/Testing Library, and Playwright.
- Packaging: `track-viz/pyproject.toml`; frontend dependencies are in
  `track-viz/web/package.json`.
- Entry points: root `make` targets delegate to `track-viz/Makefile`;
  `track-viz/scripts/run_viewer.py` starts the server.

The normal runtime flow is:

1. The page accepts an image directory and annotation file on the server.
2. `config.py` validates the selection and `loaders.py` builds a source registry.
3. `contracts.py` parses MOT rows and `indexes.py` creates frame/track indexes.
4. FastAPI serves metadata, exact JPEG bytes, track evidence, crops, and exports.
5. React fetches that evidence, decodes frames into a bounded bitmap cache, and
   draws normalized overlays on the canvas.

## Code Map

Backend code is under `track-viz/src/mot20/viewer/`:

| File | Responsibility |
| --- | --- |
| `server.py` | CLI arguments, loopback/trusted-host policy, app/router composition |
| `api.py` | Core FastAPI app, source selection/browser, metadata, observations, exact frame serving, SPA mount |
| `config.py` | Typed source configuration, direct-path inference, provenance, safe path resolution |
| `contracts.py` | MOT sequence and annotation contracts; 9-column GT and 10-column result parsing |
| `loaders.py` | Sequence/JPEG validation, source hashing, immutable registry construction |
| `indexes.py` | Sequence-local frame and track indexes |
| `tracks.py`, `filmstrip.py` | Track evidence/navigation and deterministic temporal samples |
| `context.py`, `events.py` | Competitor ranking and optional timeline event calculations |
| `crops.py` | Bounded observation crops and write-once cache entries |
| `exports.py` | Hash-keyed, bounded, atomic video exports and metadata |
| `colors.py`, `supervision_adapter.py` | Stable track colors and export-rendering conversion |

Frontend code is under `track-viz/web/src/`:

| File | Responsibility |
| --- | --- |
| `App.tsx` | Main source/frame/review state and data-loading orchestration |
| `SourceSetup.tsx` | Server filesystem browser and dynamic source loading |
| `FrameViewport.tsx` | Exact image rendering, canvas overlays, hit testing, keyboard/pointer interaction |
| `api.ts` | Typed HTTP client and response contracts |
| `bitmapCache.ts` | 150-entry LRU, decode deduplication, and batched prefetch |
| `FocusReview.tsx`, `TrackFilmstrip.tsx`, `TrackTimeline.tsx` | Stable event settings, family activity navigation, temporal track review UI |
| `CandidateChooser.tsx`, `TrackSearch.tsx` | Ambiguity resolution and direct identity lookup |
| `eventEpisodes.ts`, `timelinePlan.ts`, `focusOverlayPlan.ts` | Pure activity grouping, range-rail, and gap-aware trajectory calculations |
| `*State.ts`, `focusNavigation.ts`, `viewport.ts` | Other pure, unit-tested UI calculations |

Backend tests mirror the package under `track-viz/tests/viewer/`. Frontend unit
tests sit beside their modules; deterministic and opt-in real-data journeys are
under `track-viz/web/e2e/`. Configuration is in
`track-viz/configs/viewer.toml`, and operator/algorithm notes are in
`track-viz/README.md` and `track-viz/docs/`.

## Contracts To Preserve

- Frames are one-based and map to the exact enumerated JPEG names in `seqinfo.ini`.
- Raw MOT `xywh`, row index/hash, source hash, IDs, scores, and opaque fields are
  retained; only browser/export display geometry is clamped.
- Track IDs are sequence-local. Sentinel-only IDs do not gain track capability.
- Relative dataset paths resolve from the repository root, not from `track-viz/`.
- Frame responses use content-derived immutable ETags; playback waits for decode
  readiness and does not intentionally skip slow frames.
- Focus threshold edits stay local for 300 ms and commit immediately on Enter or
  blur. Event refreshes retain the last successful event evidence and leave the
  timeline, filmstrip, and review controls mounted.
- Focus defaults to zero nearby tracks. A positive **Number of nearby tracks**
  value enables bounded Context; returning it to zero restores focal-only review.
- Event-family navigation uses UI-derived contiguous activities while preserving
  backend raw matches. Low-confidence observations have separate exact-frame
  navigation and must not be folded into family activities.
- The range-style timeline is sequence-wide and one-based; pointer rounding,
  Arrow/Shift+Arrow/Home/End seeking, observed runs, gaps, and the playhead must
  keep their exact source-frame semantics.
- Past trajectory never includes future observations. **Show future trajectory**
  makes future evidence distinct; both states break at every missing-observation
  gap.
- Source-changing and export requests enforce trusted-host and same-origin rules.
- Dataset, annotation, cache, and export paths must not escape their allowed roots.

## Start Here

For a new coding session, read in this order:

1. `track-viz/README.md` for operation, source formats, and limitations.
2. This file for ownership and architecture.
3. `server.py` and `api.py` for backend composition, or `App.tsx` and
   `FrameViewport.tsx` for the UI control flow.
4. The nearest matching test before changing behavior.

From the repository root, use `make run`, `make test`, and `make acceptance`.
The focused commands and release evidence requirements are maintained in
`.claude/project/verification.md`. The full acceptance gate last passed on
2026-09-04 for the dark visual and simplified-control foundation.

For Focus review, start with the deterministic gapped/activity fixture in
`web/e2e/tracer.spec.ts`; use `web/e2e/real.spec.ts` track 72 only for a
read-only usability observation. Current visual-acceptance screenshots are
ignored under `artifacts/verification/visual-control-foundation/`, while the
canonical performance report and before/after source manifests are under
`artifacts/verification/`.
