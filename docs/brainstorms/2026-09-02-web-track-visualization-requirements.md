# Web-Based Dense Track Visualization Requirements

Date: 2026-09-02

Status: Brainstorm in progress; architecture direction is reviewed, but implementation is not yet approved.

## Context

The project needs an efficient way to inspect tracker output on crowded MOT20
sequences without drawing every track at once. The initial target is the
`joco_v1` result family for `MOT20-06` and `MOT20-08`. Reviewers must be able to
select a visible person, determine whether the tracker has an observation for
that person, and follow the selected track through time.

The experience must be web-based. A desktop OpenCV window is not the intended
operator surface.

## Verified local evidence

### Target `joco_v1` files

The current files under
`datasets/MOT20_TEST_TRACK_DEEPAK/joco_v1/dets/` use ten-field MOT-style rows,
but every sampled and counted row has `-1` in the second column. The sibling
`tracks/` directory is a distinct result surface and is not the Stage 0 input:

- `MOT20-06.txt`: 136,267 rows, frames 1 through 1008, up to 161 rows per frame
- `MOT20-08.txt`: 92,213 rows, frames 1 through 806, up to 137 rows per frame
- valid tracker-ID rows across both files: zero

These files can exercise detection-only frame visualization, hit testing, and
crowd interaction. They cannot exercise persistent track selection, track
traces, track timelines, or track-ID search until corrected exports provide
real identities in column 2.

### Tracked development fixture

`datasets/MOT20/train/MOT20-01/gt/gt.txt` is the ground-proven tracked fixture
for development. It contains nine-field MOT ground-truth rows with real track
identities. For included pedestrian rows (`mark = 1`, `class = 1`), it has:

- 429 frames and 74 track identities
- 19,870 pedestrian observations
- 36 to 55 pedestrians per frame, with a mean of 46.3
- overlapping pedestrian boxes in every frame
- 67.3% of box-center test clicks contained by more than one pedestrian box
- as many as seven candidate boxes at one tested point

This fixture is suitable for developing and validating tracked interactions,
but the loader must not pretend that ground truth and tracker-result rows have
the same trailing-field semantics.

## Goal

Build a local web application that keeps dense MOT20 frames visually quiet,
lets a reviewer deliberately select the intended person despite overlapping
boxes, and then presents only the temporal track evidence needed to inspect
that identity.

The application should optimize for both cognitive load and review throughput.
It should not require a reviewer to watch every frame or read dozens of labels
simultaneously.

## Product principles

- Start with an unannotated source frame. Reveal tracking information only in
  response to reviewer intent.
- Treat selection as an evidence-guided interaction, not as an infallible
  geometric guess.
- Preserve exact frame, sequence, track, and source-row identity throughout.
- Keep interaction frame-accurate and keyboard-friendly.
- Keep source images and result files immutable.
- Add complexity only after the simpler stage is validated against real data.

## Recommended architecture

Use a thin FastAPI backend with a React and TypeScript browser client.

The browser should use two stacked HTML canvases:

1. A lower canvas displays the source JPEG frame.
2. An upper canvas owns hover feedback, hit testing, candidate outlines, and
   selected-track overlays.

All boxes for the current frame remain in browser memory as hit regions, but
they are not all drawn by default. A single image-to-screen transform and its
inverse must be shared by drawing and hit testing. Frames are letterboxed and
never stretched.

The FastAPI backend should:

- validate and normalize MOT rows at startup
- build sequence-scoped frame and track indexes in memory
- serve only enumerated source frames and bounded crop requests
- expose frame metadata, track observations, and event summaries through REST
- record a content hash for each loaded result file
- reject path traversal and source/result contract mismatches

No database, WebSocket, or spatial index is required initially. Linear hit
testing over at most a few hundred boxes is inexpensive. The main interaction
bottleneck is JPEG decoding and browser memory, not box lookup.

## Deliberate person selection

Bounding boxes do not encode which visible pixels belong to which person.
During occlusion, no box-ranking rule can guarantee that a click maps to the
person intended by the reviewer. The interface must therefore expose ambiguity
rather than silently choosing.

### Hover: preview without commitment

- Moving over a person highlights the currently ranked candidate immediately.
- A small magnified crop helps with distant or partially occluded people.
- The highlight remains stable under small pointer movements to avoid flicker.
- The mouse wheel or `Tab` cycles through all boxes containing the pointer.
- Hover does not start track-following mode or mutate review state.

### First click: pin the evidence

- The first click pauses playback and freezes the candidate set at that point.
- If one box contains the point, its outline and crop are pinned for
  confirmation.
- If several boxes contain the point, show deterministic numbered outlines and
  a compact chooser with at most five visible candidate cards at a time.
- Each candidate card shows its track ID when available and crops from the
  current, an earlier, and a later observation.
- Hovering or focusing a candidate card highlights its exact box in the frame.
- Additional candidates remain reachable by scrolling or keyboard navigation;
  they must not be discarded by the visible-card cap.

### Second click or Enter: confirm and follow

- A second click on the pinned candidate, a click on its candidate card, or
  `Enter` confirms the selection.
- Confirmation enters focus mode and follows the selected track.
- `Escape` cancels the pinned selection or exits focus mode.
- Clicking elsewhere while pinned starts a new candidate selection.

This is a two-stage sequence of normal clicks, not the browser's literal
double-click event. Literal double-click delays single-click behavior, is less
discoverable, and does not resolve which overlapping candidate was intended.

### Candidate ranking

Ranking chooses the initial preview only; reviewer confirmation remains
authoritative. Use this order:

1. Prefer boxes that contain the pointer.
2. Prefer the smallest containing area, which usually identifies the most
   specific box.
3. Use normalized pointer-to-box geometry and distance to the box edge as
   tie-breakers.
4. Preserve the current candidate with hover hysteresis until the pointer moves
   materially or the reviewer cycles candidates.

If no box contains the pointer, report that there is no tracker observation at
that location. Optionally offer the nearest bounded set using distance to the
box edge, not distance to the box center.

## Visualization modes

### Explore mode

- Show the clean frame with no persistent boxes or labels.
- Reveal only the hover candidate and optional cursor magnifier.
- Holding a dedicated key temporarily reveals all current-frame boxes so the
  reviewer can audit what the tracker emitted.

### Focus mode

- Draw only the selected track with a strong stable color, box, ID label, and
  short motion trace.
- Keep labels off unrelated tracks.
- Show previous and next observations when the selected track has a gap.
- Provide direct previous/next observation navigation.

### Context mode

- Add only tracks that plausibly compete with the selected track near the
  current time.
- Use thin corner markers for context tracks and retain the stronger focal
  style for the selected track.
- Rank context using overlap and box-height-normalized proximity over a small
  temporal window.
- Apply hysteresis so context markers do not flicker between frames.
- Use a user-adjustable count with a conservative default and hard cap rather
  than assuming that exactly three context tracks is always correct.

## Track evidence

### Timeline

For a selected track, show:

- first and last observation
- missing-frame gaps
- low-confidence observations when confidence is meaningful
- optional abrupt displacement or scale-change events
- optional close-interaction intervals

Gap and lifespan information are objective. Motion and interaction events are
threshold-dependent, so their thresholds must be visible and those event types
must remain optional until validated. Displacement should be normalized by box
height to account for perspective and scale.

Clicking any timeline marker seeks to the exact source JPEG frame.

### Crop filmstrip

- Show a bounded temporal sample rather than every observation.
- Include the current observation and representative earlier/later crops.
- Cap the initial strip at 64 crops and document the temporal sampling rule.
- Cache crops under a derived, git-ignored artifact path keyed by source hash
  and crop parameters.
- Preserve raw box coordinates in metadata even when display crops clamp boxes
  to image boundaries.

### Search and navigation

- Search by sequence-local track ID.
- Support frame navigation by 1 and 10 frames.
- Support previous/next observation, gap, and enabled event navigation.
- Namespace all selection and index state by sequence so switching sequences
  cannot leak a track selection.

## Supervision usage

Use the Python `supervision` library for normalized detection collections and
offline rendering, not for latency-sensitive interactive frame rendering.

Expected uses include:

- NumPy-style `Detections` filtering by tracker ID
- `BoxAnnotator` for focal tracks in exported media
- `BoxCornerAnnotator` for context tracks
- `LabelAnnotator` with smart positioning for selected labels
- `TraceAnnotator` with `ColorLookup.TRACK` and bounded trace length
- `VideoSink` for short focused clips or per-track videos

The browser canvas is the single source of truth for live overlays and hit
testing. Server-rendering each interactive frame would add JPEG encode/decode
latency and could allow rendered boxes to disagree with browser hit regions.

Pin the Supervision version when implementation begins. Define a deterministic
track-to-color rule shared by browser rendering and offline exports, and verify
the two render paths with a representative golden frame.

## Data contracts

Normalize both source formats into one internal observation model while
preserving raw rows:

- Ground truth adapter: nine fields
  `frame,id,x,y,width,height,mark,class,visibility`
- Tracker result adapter: ten fields
  `frame,id,x,y,width,height,confidence,field_8,field_9,field_10`

The meanings of tracker-result fields 8 through 10 are not established for
`joco_v1`; all three are currently `-1`. Preserve them as opaque source values
until the producing export contract defines their semantics.

The normalized model must retain at least sequence, one-based frame, track ID,
raw and display-clamped box coordinates, score or visibility where meaningful,
source-row index, and source-file hash.

Startup validation must check:

- expected field count and numeric parsing
- one-based frame range and exact image identity
- sequence frame count against `seqinfo.ini` and source JPEGs
- finite, non-degenerate box geometry
- per-column distributions and constant sentinel columns
- valid tracker-ID availability before enabling track features
- empty frames and result rows without matching images

Negative or overflowing coordinates may be valid border observations. Preserve
them in source metadata, clamp only for rendering and crops, and reject only
materially invalid or degenerate geometry according to an explicit contract.

Features driven by a missing or constant column must be disabled with a clear
diagnostic rather than silently producing misleading output.

## Staged delivery

### Stage 0: detection-only viewer

Buildable with the current `joco_v1` files:

- strict input validation and feature diagnostics
- exact frame loading and scrubbing
- clean-frame canvas rendering
- hover highlighting and overlap candidate cycling
- first-click pinning and candidate confirmation behavior
- keyboard frame navigation
- bounded directional JPEG prefetch

Track-following actions remain visibly disabled when all IDs are `-1`.

### Stage 1: tracked interaction

Develop against `MOT20-01` ground truth and validate again when corrected
`joco_v1` tracker exports arrive:

- sequence-scoped track index
- confirmed focus and context modes
- trace, lifespan, gaps, filmstrip, and track-ID search
- previous/next track-observation navigation
- tracker/checkpoint and adaptation provenance for each corrected export

Validation on sequences 06 and 08 is local test-adapted workflow validation,
not held-out tracker evaluation or evidence for parameter selection.

### Stage 2: review acceleration and export

- validated optional event detection
- short focused clip export or offline per-track videos
- shared browser/export colors and representative render comparison
- append-only review decisions only after their schema and workflow are
  separately approved

Before implementing asynchronous clip jobs, test whether synchronous bounded
exports of at most 300 frames or offline per-track videos satisfy the actual
review workflow.

## Performance requirements

- Use `createImageBitmap` for browser JPEG decoding where supported.
- Maintain a bounded least-recently-used bitmap cache, initially around 100 to
  200 frames, and explicitly release evicted bitmaps.
- Prefetch in the current playback or scrub direction.
- Serve immutable frame responses with cache headers and ETags.
- Redraw the overlay canvas through `requestAnimationFrame` without re-blitting
  the image canvas for pointer-only changes.
- Keep all indexes in memory for the current data scale.
- Do not add a spatial index unless measured pointer performance requires it.

## Safety and provenance

- Open source images and result files read-only.
- Write crops, caches, clips, and future review decisions only to separate
  git-ignored derived paths.
- Enumerate allowed sequences and frames at startup; never construct a file path
  directly from untrusted client input.
- Display and retain the source result hash so a changed export cannot silently
  inherit stale selections or decisions.
- A future decision anchor should contain sequence, frame, track ID, source row
  hash, result-file hash, action, reviewer, timestamp, and reason.
- Do not describe `MOT20-01` ground-truth behavior as measured tracker quality.
- Work involving adapted MOT20 test data remains subject to `docs/MOTPolicy.md`.

## Alternatives considered

- OpenCV desktop UI: capable of mouse callbacks but rejected because the
  required operator surface is web-based.
- Gradio custom component: suitable for a disposable detection-only prototype,
  but the required overlap chooser, timeline, keyboard workflow, and canvas
  state approach the cost of a dedicated client.
- Streamlit: its rerun interaction model is poorly suited to continuous frame
  scrubbing and pointer feedback.
- FiftyOne: useful later for embedding exploration, but less direct for this
  specialized selection and review state.
- CVAT: valuable for annotation creation and already present in this repository,
  but too annotation-heavy for quiet, read-only track exploration.
- Offline per-track videos: an important low-cost baseline once valid IDs are
  available. Generate and test these before assuming every review requires the
  full interactive client.

## Success criteria

- A reviewer can seek to an exact frame without frame-number ambiguity.
- The default scene remains readable even with more than 100 result boxes.
- Hover feedback identifies the current candidate without committing state.
- Every ambiguous click exposes all candidates and allows deliberate keyboard
  or pointer selection.
- Pointer-to-highlight latency is measured at the densest available frame and
  remains below 50 ms at the 95th percentile on the target review machine.
- A confirmed track remains selected and correctly rendered across its
  observations and gaps.
- Missing track IDs disable track features explicitly.
- Browser and offline-export colors identify the same track consistently.
- No source dataset, result, or ground-truth file is modified.

## Independent architecture review

The proposed FastAPI and browser-canvas architecture was reviewed through
GitHub Copilot CLI 1.0.79 using `claude-opus-5` with medium reasoning effort.
The verdict was "approve with changes." The changes incorporated here include:

- staged delivery gated on real track IDs
- browser-only interactive rendering
- one shared render/hit-test transform
- explicit box and frame validation
- deterministic ambiguity handling
- keyboard-first navigation and temporary all-box reveal
- bounded browser and filmstrip caches
- deferral of databases, WebSockets, spatial indexes, heuristic events, and
  asynchronous export jobs
- source hashing and immutable derived artifacts
- offline per-track videos as a lower-cost baseline

## Open decisions

- What corrected `joco_v1` export will provide stable track IDs for sequences
  06 and 08, and what tracker/checkpoint produced it?
- Should Stage 0 be implemented before corrected exports arrive, or should work
  begin directly with Stage 1 against the `MOT20-01` fixture?
- Should every selection require pin then confirm, or may an optional fast mode
  immediately confirm a point with exactly one candidate?
- Which pointer and keyboard gestures should cycle candidate boxes without
  conflicting with frame scrubbing or browser behavior?
- What maximum context-track count remains readable on the target display?
- Which event types and thresholds prove useful after observation on tracked
  data?
- Are offline per-track videos sufficient for a meaningful portion of review,
  and which cases still require the interactive application?
- What review-decision actions and schema will be approved after exploration is
  validated?

## Current non-goals

- Do not infer or invent tracker identities for the existing `joco_v1` rows.
- Do not evaluate tracker accuracy from MOT20 test sequences without ground
  truth.
- Do not add ReID similarity search in the initial viewer.
- Do not implement track correction, split, merge, or relabeling decisions in
  this brainstorm phase.
- Do not overwrite source images, ground truth, detections, or tracker results.