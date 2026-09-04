# Annotation Platform UI Direction

Date: 2026-09-04

Status: Proposed direction; pending user approval

Related requirements:
`track-viz/docs/brainstorms/2026-09-04-focus-controls-and-visual-direction-requirements.md`

Reference: `https://www.beautifului.dev/`

## Product Direction

Track-Viz should grow from a read-only tracker viewer into an annotation and
track-correction workspace. Expected future operations include bounding-box
correction, track merge, track split, track swap, review of proposed changes,
undo/redo, and export of corrected derived results.

The intended product is a specialized tracker-correction workbench, not a
general replacement for CVAT. CVAT remains useful for broad manual annotation;
Track-Viz should differentiate itself through track history, identity evidence,
ReID/event-assisted proposals, interval corrections, MOT-native provenance, and
fast corrected-result review.

The original images and tracker or ground-truth files remain immutable. Editing
must produce versioned derived decisions and corrected outputs with source hashes
and review provenance.

The initial product assumption is a local, single-reviewer workflow. The edit
model should retain revision identifiers and explicit operations so correction
semantics survive a later multi-reviewer design. Revisions alone do not solve
future collaboration, which would still require identity, permissions,
conflict/rebase policy, and locking decisions.

## Options Considered

### 1. Keep the current UI and add tools inline

This has the lowest immediate cost, but the current long single-column Focus
panel is already dense. Adding box tools, track operations, reasons, pending
changes, and undo history there would make discovery and keyboard order worse.
This option defers rather than avoids layout work.

### 2. Copy Beautiful UI or migrate to its apparent stack

This could create a polished surface quickly for generic controls, but most of
the gallery targets AI-agent interfaces rather than frame annotation. Moving the
current React/Vite/plain-CSS application to Next.js or Tailwind would add carrying
cost without improving canvas rendering, source safety, or edit correctness.

### 3. Build an annotation-ready shell using selected visual principles

This preserves the existing application and fast canvas while introducing a
small reusable design system. The workspace structure should be introduced with
the first real editing workflow rather than as empty placeholder panels. It is
the recommended direction.

## Recommended Workspace

Move toward four logical regions on desktop:

1. **Command bar** — source, frame transport, mode, save/dirty state, undo, and
   redo. Source-path setup is secondary and can expand when needed.
2. **Viewport** — the dominant image and overlay surface. Review and edit
   gestures stay here, with no decorative UI over the image unless it conveys
   selection or edit state.
3. **Inspector** — a collapsible, context-sensitive side panel for the current
   observation, track identity, nearby tracks, event evidence, and correction
   actions.
4. **Temporal workspace** — a collapsible lower area for the sequence rail,
   track segments, gaps, filmstrip, proposals, and pending edits.

These are interaction regions, not a requirement for four permanently visible
panes. Empty panels should not reduce the viewport. Asynchronous loading and
status changes must not resize the viewport or move precision controls.

At narrow widths the inspector and temporal workspace may stack below the
viewport, but the same controls and accessible order should remain. The product
is desktop-first for annotation; narrow layout remains review-capable rather
than attempting to make precision box editing equally productive on a phone.

Use explicit **Review** and **Edit** modes. Review mode remains safe and quiet.
Edit mode reveals correction tools and persistent unsaved-change state. This
reduces accidental edits and prevents every future control from appearing at
once.

Do not finalize this shell before the first editing vertical slice defines what
the inspector, history, and save states actually need.

## Beautiful UI Reuse

Reuse the visual language, not the framework:

- neutral surface hierarchy and compact typography
- hairline borders, moderate radii, and restrained elevation
- compact segmented controls, chips, status indicators, and loading states
- selection-action patterns for operations on one or more tracks
- diff and record-table patterns for pending corrections and history
- one strong accent for active state, with separate warning and destructive
  semantics

Do not import chat, prompt, thinking, or agent-specific components. Do not add
animation, blur, large shadows, or ornamental effects to the viewport path.
Reimplement the small applicable primitives in the existing React and CSS stack.
If source code is copied substantially, retain the required MIT attribution.

## Performance Contract

The current accepted evidence records 16.8 ms pointer-to-overlay p95 on a dense
161-observation frame and 14.7 ms warm seek p95. This covers Explore hover, not
continuous edit dragging or revision application. A visual or annotation change
must not silently weaken those baselines, and editing requires separate budgets.

- Keep image and overlay rendering canvas-based and driven by
  `requestAnimationFrame`.
- Use an imperative transient edit layer, or equivalent isolated canvas path,
  so drag previews do not require React renders or rebuild stable overlays.
- Keep transient drag geometry outside React render state; publish committed
  geometry only at bounded interaction points.
- Do not fetch or persist on every pointer movement.
- Keep frame decode, caching, and prefetch independent from inspector rendering.
- Virtualize long track, proposal, and history collections when they are added.
- Move expensive propagation, similarity, or bulk-diff work off the interactive
  path, using bounded background work only when measurement justifies it.
- Add a spatial index only if measured edit hit-testing exceeds the latency
  budget; the present linear scan is already adequate at the tested density.
- Retain the existing pointer and warm-seek baselines as regression evidence.
- Measure edit-drag event-to-next-paint latency, coalesced/dropped updates,
  single-command commit and undo latency, seek latency with a representative
  large ledger, and corrected-export materialization time.
- Target drag previews within one displayed frame at 60 Hz under normal load;
  any p95 above 25 ms requires investigation, and 50 ms remains a hard ceiling.

## Correction Model

Corrections should be explicit operations over an immutable source rather than
in-place mutations. A local SQLite correction ledger is the recommended first
persistence layer because transactions and recovery matter more than avoiding a
small embedded database. Portable JSON export can remain the interchange and
audit format.

The backend should own the authoritative effective state:

```text
immutable source + ordered correction ledger = effective revision
```

Every derived API request and response must identify the source hash and
effective revision or effective-state hash. A response calculated before an edit
must not be accepted after the active revision changes.

The eventual operation vocabulary is:

- update, add, or remove a bounding box
- split a track at a frame or interval
- merge track fragments
- swap track assignments across a selected interval
- exclude an observation or track from a derived use

Every operation should retain the source hash, sequence, original track and row
identities, affected frames, before/after values, reviewer, timestamp, reason,
and revision. Multi-frame identity operations must be atomic. Undo/redo should
operate on these commands, and export should materialize a new corrected result
without overwriting the source.

The first slice should only update an existing box. Added observations require
generated identities and field defaults; identity operations require exact
collision, interval, gap, and destination-ID semantics. Those decisions should
be made before their controls are designed.

The editing lifecycle must distinguish transient preview, pending command,
durable draft revision, and published corrected export. It must define crash
recovery, source switching with unsaved changes, optimistic revision checks, and
undo followed by a new branch of edits.

The UI should show a reviewable pending-change list and before/after diff before
publishing a corrected result. This is where Beautiful UI's selection-action,
record-table, and diff-table patterns are most relevant.

## Delivery Sequence

### Stage 1: reusable visual foundation and accepted simplifications

- replace the ruled-paper appearance with a calm neutral workspace
- replace and expand the existing design tokens and compact reusable
  control/panel styles
- implement the already accepted nearby-track, future-trajectory, and banner
  simplifications
- preserve existing application structure and behavior where possible

### Stage 2: correction-state contract

- define revisioned effective state, the local correction ledger, stale-revision
  rejection, recovery, and corrected export contracts
- define exact update-existing-box geometry semantics using raw MOT geometry and
  display-clamped rendering without conflating the two
- define minimum box size, off-frame behavior, commit/cancel, keyboard/numeric
  editing, and source-switch behavior

### Stage 3: first end-to-end editing vertical slice

- add precise zoom and pan plus **Review** and **Edit** modes
- implement update-existing-box with transient preview, commit, cancel, durable
  draft, undo/redo, pending-change review, and corrected export
- introduce only the command-bar, collapsible-inspector, and temporal-workspace
  structure needed by this real workflow
- define keyboard shortcuts without conflicting with current frame seeking,
  playback, or all-box reveal
- validate export round-trip, crash recovery, stale revision rejection,
  high-DPI resize/zoom, accessibility, and measured edit performance

### Stage 4: identity operations

- add split, merge, and interval swap after their exact semantics and conflict
  behavior are validated on synthetic and ground-truth-backed cases

### Stage 5: assisted review

- add ranked ReID/event proposals as suggestions, never silent source mutations
- record accept, reject, and uncertain decisions with model and threshold
  provenance

## Non-Goals For The First Pass

- no full Beautiful UI clone
- no Next.js, Tailwind, or broad component-library migration
- no simultaneous implementation of all annotation operations
- no external database service, authentication, or multi-user locking until
  collaboration is an approved near-term requirement
- no source-file overwrite
- no reduction of deterministic browser, accessibility, source-integrity, or
  performance verification

## Approval Questions

- Is local single-reviewer use the correct first annotation target?
- Is a single dark neutral theme acceptable for the first refresh? Supporting
  both light and dark would double visual and accessibility verification before
  it improves annotation correctness.
- Is update-existing-box, including zoom/pan, revision persistence, undo/redo,
  and corrected export, the right first vertical slice before add/remove and
  track split, merge, and swap?
