# Track-Viz Annotation Platform Foundation Plan

Date: 2026-09-04

Status: Approved for execution; dark neutral theme selected for the first task

Execution: Epic 1 Tasks 1 and 2 are complete. Task 3 automated acceptance,
real-data performance, source-integrity comparison, axe scans, and screenshots
are complete; user visual acceptance is pending.

Origins:

- `track-viz/docs/brainstorms/2026-09-04-focus-controls-and-visual-direction-requirements.md`
- `track-viz/docs/brainstorms/2026-09-04-annotation-platform-ui-direction.md`

## Goal

Evolve Track-Viz from a read-only tracker viewer into a fast, specialized
tracker-correction workbench without replacing CVAT, weakening source
immutability, or regressing current frame/overlay interaction performance.

The immediate deliverable is a dark neutral visual foundation. Later work is
split into independently reviewable epics for control simplification, event
semantics, revisioned box correction, and track identity correction.

## Invariants

- Source images, MOT annotations, and tracker results remain read-only.
- Exact sequence, frame, source-row, source-hash, and track identities remain
  preserved until an explicit derived correction operation says otherwise.
- Image decode, bitmap caching, prefetch, and canvas overlay rendering stay off
  ordinary inspector/control render paths.
- Existing pointer and warm-seek performance evidence remains the regression
  baseline; new editing paths receive separate measurements.
- Desktop is the precision-annotation target. Narrow layout remains usable for
  review and must not overflow or lose accessible operation order.
- No Next.js, Tailwind, broad component library, or external database service is
  introduced by this plan.
- Deterministic synthetic cases prove behavior. Real MOT20 data remains
  read-only observational evidence with source-manifest comparison.

## Epic 1: Modernize Visual And Control Foundation

### Task 1: Establish The Dark Visual Foundation

Goal: replace the ruled-paper visual language with a calm, dark-neutral review
workspace while preserving the current DOM, control behavior, canvas rendering,
responsive order, accessibility, spacing, font metrics, grid structure, and
viewport geometry. This task changes tokens, colors, backgrounds, borders, and
restrained visual effects only.

Modify:

- `track-viz/web/index.html` only for dark color-scheme/theme metadata
- `track-viz/web/src/styles.css`
- `track-viz/web/e2e/tracer.spec.ts`
- `track-viz/README.md` only if operation or theme behavior needs documenting

Test-first tracer bullet:

- add one deterministic browser assertion for the public visual contract:
  dark page surface, no ruled background image, dark semantic panel surfaces,
  measurable text/control contrast, no overflow, minimum viewport area,
  unchanged image/overlay alignment, and desktop/narrow coverage
- run it against the current UI and confirm the expected failure
- replace and expand the existing CSS tokens, then restyle existing components
  without moving their semantic structure

Verification:

- focused Playwright visual-contract test in desktop and narrow projects
- frontend unit tests, TypeScript, and automated axe scans
- deterministic Playwright suite, including axe and layout stability
- production frontend build
- fresh desktop and narrow screenshots for user review

Risks:

- insufficient contrast in disabled, warning, provenance, and timeline states
- CSS changes accidentally altering geometry or canvas size
- form controls retaining light user-agent styling
- broad selector changes breaking dense/narrow layouts

### Task 2: Simplify Focus Controls And Visible Provenance

Goal: implement the accepted product changes without changing source metadata or
provenance contracts.

Modify:

- `track-viz/web/src/App.tsx`
- `track-viz/web/src/FocusReview.tsx`
- `track-viz/web/src/styles.css`
- nearest frontend tests
- `track-viz/web/e2e/tracer.spec.ts`
- `track-viz/README.md`
- `track-viz/docs/HANDOFF.md`

Behavior:

- replace Focus/Context plus Context-count controls with one **Number of nearby
  tracks** field
- default to `0`; positive values enable bounded context
- replace the trajectory radio fieldset with **Show future trajectory**
- remove the visible local-test-adapted banner while preserving provenance in
  APIs, documentation, decisions, and exports

Depends on: Task 1, because the simplified controls consume the approved visual
tokens and Task 1 owns the shared stylesheet first.

Verification:

- test-first frontend state and integration coverage
- exact context request/count behavior for `0`, positive values, and cap
- past-only default and visibly distinct future trajectory when enabled
- no policy banner in the viewer; provenance remains present in source/export
  contracts

### Task 3: Validate The Visual And Control Foundation

Goal: integrate Tasks 1 and 2 and obtain visual acceptance before annotation UI
begins to depend on the new surface hierarchy.

Depends on: Tasks 1 and 2.

Verification:

- full `make acceptance`
- opt-in `make e2e-real`
- desktop and narrow screenshots across Explore, chooser, Focus, and context
- zero serious/critical axe violations
- pointer and warm-seek comparison against the current accepted evidence
- byte-identical configured source manifests
- user visual acceptance

## Epic 2: Validate Review-Event Semantics

### Task 1: Characterize Raw Event Algorithms

Read and test `events.py`, API contracts, and synthetic fixtures for abrupt
displacement, scale change, and close interaction. Cover normalization, frame
gaps, zero-size boxes, missing observations, competitors, threshold operators,
and boundary equality without changing production behavior.

### Task 2: Define Product-Level Event Semantics

Use deterministic corruptions and read-only real-track observations to decide
what each event means to a reviewer, how contiguous matches group, how anchors
are chosen, what thresholds communicate, and how previous/next acts inside an
activity. Record approved semantics before changing algorithms.

Depends on: Task 1.

### Task 3: Implement Approved Event Corrections

Change only behavior approved by Task 2. Preserve raw auditable measurements,
test every family independently and in combination, and keep event refreshes
isolated from track, filmstrip, and image loading.

Depends on: Task 2.

### Task 4: Integrate And Validate Event Review

Run deterministic and real-data browser journeys, accessibility, layout,
performance, documentation, and source-integrity checks. Obtain user acceptance
for event meaning and navigation.

Depends on: Task 3.

## Epic 3: Deliver The First Box-Correction Vertical Slice

### Task 1: Specify Revisioned Correction State

Define the authoritative equation `immutable source + ordered correction ledger
= effective revision`; SQLite transaction/recovery behavior; stable identities;
revision/effective-hash API contracts; preview, pending, durable draft, undo,
redo, branch, source-switch, and corrected-export states; and exact existing-box
geometry semantics.

### Task 2: Implement The Correction Ledger And Reducer

Add transactional local persistence and the authoritative backend reducer without
overwriting source files. Cover replay, atomic command append, revision advance,
branch behavior after undo, and crash recovery.

Depends on: Task 1.

### Task 3: Add Revision-Aware Effective Read APIs

Return source and effective-revision identity from affected frame, observation,
track, context, and export APIs. Reject stale writes and ensure stale responses
cannot replace the active revision. Event evidence is explicitly unavailable for
edited drafts until the separate event-semantics epic is approved and a later
revision-aware event integration is implemented.

Depends on: Task 2.

### Task 4: Add Precision Zoom And Pan

Add high-DPI-correct zoom/pan primitives and non-conflicting navigation controls.
Preserve exact image/overlay alignment, frame seeking, playback, and all-box
reveal shortcuts.

Depends on: Task 1 and Epic 1 visual acceptance.

### Task 5: Add Transient Existing-Box Edit Rendering

Add Review/Edit modes, handle hit-testing, pointer capture, escape/cancel, and an
imperative transient edit layer that does not write or trigger React renders per
pointer movement.

Depends on: Tasks 1 and 4, and Epic 1 visual acceptance.

### Task 6: Commit Existing-Box Corrections

Connect the edit preview to validated durable commands with minimum-size,
off-frame, raw-versus-display geometry, conflict, cancel, and optimistic revision
behavior.

Depends on: Tasks 3 and 5.

### Task 7: Add Undo, Redo, And Draft Recovery

Implement command-granular undo/redo, branch-after-undo behavior, durable draft
recovery, source-switch protection, and visible dirty/save state.

Depends on: Task 6.

### Task 8: Add Pending Diff And Numeric Adjustment

Provide before/after inspection, pending-change history, keyboard adjustment,
and exact numeric `xywh` editing without changing committed state until the user
confirms.

Depends on: Tasks 6 and 7.

### Task 9: Materialize Corrected MOT Results

Export a new corrected result with deterministic row ordering, numeric
precision, source and revision provenance, and round-trip validation. Never
overwrite the configured source.

Depends on: Task 8.

### Task 10: Validate The Box-Correction Vertical Slice

Exercise reducer replay, stale revisions, crash recovery, high-DPI zoom/edit,
mouse and keyboard paths, source switching, undo branching, export round-trip,
source integrity, edit-drag latency, commit/undo latency, large-ledger seek, and
desktop/narrow layout.

Depends on: Task 9.

## Epic 4: Add Track Identity Corrections

### Task 1: Define Observation And Identity Operation Semantics

Approve generated observation identities, added-row defaults, track-ID
allocation/export mapping, interval inclusivity, gaps, overlaps, collisions,
operation composition, and behavior when earlier revisions change referenced
tracks.

### Task 2: Add And Remove Observations

Implement explicit observation creation/removal using the approved field,
identity, revision, undo, and export contracts.

Depends on: Task 1 and Epic 3 completion.

### Task 3: Split Tracks

Implement exact interval-boundary split behavior, preview, collision handling,
undo, provenance, and export tests.

Depends on: Task 1 and Epic 3 completion.

### Task 4: Merge Track Fragments

Implement reviewed fragment selection, overlap and duplicate handling,
destination identity, atomic commit, undo, and export. Candidate ranking and
similarity assistance are outside this task.

Depends on: Task 1 and Epic 3 completion.

### Task 5: Swap Track Assignments Across An Interval

Implement two-track interval swap with explicit one-sided presence, gaps,
collisions, preview, atomic commit, undo, and export semantics.

Depends on: Task 1 and Epic 3 completion.

### Task 6: Integrate And Validate Identity Correction

Run deterministic split/merge/swap/duplicate/gap cases, ground-truth-backed
diagnostics, large-ledger and proposal-list performance, accessibility, crash
recovery, corrected-export round trips, and source-integrity checks.

Depends on: Tasks 2, 3, 4, and 5.

## Epic 5: Add Assisted Correction Proposals

This epic is deferred backlog until a ReID implementation and proposal-quality
evaluation contract exist. It is not executable work under the current plan.

### Task 1: Define Proposal Inputs And Evaluation

Specify ReID/event inputs, model and checkpoint provenance, candidate generation,
ground-truth-backed diagnostic evaluation, thresholds, ranking, and uncertainty.

### Task 2: Implement Proposal Generation

Generate bounded, revision-aware proposals without applying corrections.

Depends on: Task 1, Epic 2 completion, Epic 4 completion, and an established ReID
workstream output.

### Task 3: Add Proposal Review And Decision Capture

Present evidence and accept/reject/uncertain actions. Record source, revision,
model, threshold, reviewer, reason, and decision provenance.

Depends on: Task 2.

### Task 4: Validate Assisted Review

Measure proposal quality, reviewer workload, large-list performance, revision
staleness, accessibility, source integrity, and corrected-export provenance.

Depends on: Task 3.

## Current Execution Boundary

Create Epics 1-4 and their direct child tasks in Beads with this plan as the
specification. Record Epic 5 and its children as deferred backlog. Claim and
execute only Epic 1, Task 1. Do not begin event behavior changes or annotation
persistence in the current task.
