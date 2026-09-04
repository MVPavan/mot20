# Track-Viz Prediction Problem Discovery And Review Plan

Date: 2026-09-05

Status: Proposed for product review; not authorized for implementation

Origins:

- user discussion on identifying tracking failures before correction
- `track-viz/docs/brainstorms/2026-09-04-annotation-platform-ui-direction.md`
- `track-viz/docs/plans/2026-09-04-annotation-platform-foundation.md`
- `docs/MOTPolicy.md`

## Goal

Add an explainable review workflow that helps a human find, inspect, and label
likely prediction problems before any correction is applied. The first useful
version should work on tracker predictions without ground truth, including local
MOT20 test-style sources. A later ground-truth audit mode should provide
reproducible evaluator-defined attribution where authorized annotations exist.

This plan covers problem discovery, evidence presentation, and review decisions.
It does not replace the correction-ledger, box-editing, or identity-operation
work already defined by the annotation-platform foundation plan.

## Product Assumption To Confirm

Prediction-only review is the primary workflow. Ground truth is optional and
must never be required to open the review inbox or inspect a candidate. Without
ground truth, the system reports **candidates**, never confirmed tracking errors.
Only a reviewer decision or an authorized ground-truth comparison may confirm
an error.

## Failure Taxonomy

Separate what the software can observe from what a reviewer may conclude. One
candidate has one observable type, while a confirmed incident may receive
multiple failure labels because real failures overlap.

### Observable candidate types

| Candidate type | Prediction-only observation |
| --- | --- |
| Same-ID gap risk | One predicted ID is absent between two of its own observations. Visibility and correctness remain unknown. |
| Premature-endpoint risk | A predicted track begins or ends away from an obvious sequence or image-boundary exit. The person may still be occluded or legitimately absent. |
| Cross-ID continuation hypothesis | One tracklet ends and another begins with compatible time, motion, position, and scale. Whether they depict the same person remains unknown. |
| Within-ID takeover risk | One predicted ID has a large approved continuity discontinuity, optionally near competing tracks. |
| Paired-swap hypothesis | Two predicted identities interact and their most plausible pre/post continuations invert as a pair. |
| Concurrent-duplicate risk | Two predicted identities have sustained overlap and compatible motion during the same frames. |

Do not generate a missing-observation claim from a gap alone. Evidence from an
unmatched detector output is unavailable in the current single-source viewer and
must wait for a separate versioned detection-input contract.

### Reviewer-confirmed failure labels

| Failure label | Definition | Possible later correction |
| --- | --- | --- |
| Missing observation / track drop | The reviewer confirms that a visible person lacks a predicted box for one or more frames. | Add an observation or bridge a reviewed interval |
| Broken identity continuity / fragmentation | The reviewer confirms that one person is represented by disconnected predicted tracklets. | Merge track fragments |
| Identity switch | The reviewer confirms that a person continues under another predicted identity or that an ID starts following another person. | Split, merge, or reassign an interval |
| Two-way identity swap | Two identities exchange people around the same interaction; this is a specific identity-switch subtype. | Atomic two-track interval swap |
| Identity takeover or merge | One predicted identity represents different people across time. | Split the track and reassign fragments |
| Duplicate track | Multiple predicted identities follow the same person at the same time. | Remove or merge reviewed observations |
| Localization error | The identity is plausible but the predicted box is materially misplaced or badly sized. | Update the existing box |
| False-positive observation or track | A predicted box does not correspond to a person. | Remove an observation or track from the derived result |

A confirmed incident may carry more than one failure label. Store one primary
correction target separately so overlapping descriptions do not create competing
actions. Treat **track switching** as ordinary language for identity switching.
Treat **identity drop** as broken identity continuity when a successor is
confirmed; use missing observation only for frames where visibility without a
box is confirmed.

MOTChallenge fragmentation and identity-switch metrics have evaluator-specific
definitions. Ground-truth audit mode must use the selected evaluator's exact
contracts rather than silently equating metric events with the product labels
above.

## Existing Evidence And Its Limits

Track-Viz already exposes useful raw evidence:

- `track-viz/src/mot20/viewer/tracks.py` returns observation frames, internal
  gaps, endpoints, and complete per-track observations.
- `track-viz/src/mot20/viewer/context.py` ranks nearby competing identities using
  overlap and edge proximity in a bounded temporal window.
- `track-viz/src/mot20/viewer/events.py` reports abrupt center displacement,
  scale change, close interaction, and meaningful low-confidence observations.
- the Focus timeline, trajectory, filmstrip, and nearby-track overlay provide a
  starting point for temporal review.

These signals are not error classifiers. A gap may be a valid occlusion, an
endpoint may be an exit, displacement may come from camera or box motion, and a
close interaction is only a risk region. The existing event-semantics epic must
characterize and approve those signals before this workflow promotes them as
candidate reasons.

## Review Candidate Contract

Create one source-scoped, versioned contract shared by every candidate
generator. A candidate should contain:

- source key, source hash, sequence, generator name, generator version, and
  effective revision when correction state exists
- candidate category and optional subtype
- focal track IDs, competitor or successor track IDs, source-row identities,
  and exact affected frame interval
- a bounded review interval with before, during, and after context
- individual signal values and thresholds, not only an opaque aggregate score
- a deterministic ranking score plus a human-readable list of reasons
- feature availability, so missing confidence, detections, embeddings, or
  ground truth are explicit rather than synthesized
- review state plus reviewer-decision provenance when a decision is persisted

Candidate identity must be stable for the same source hash, generator version,
configuration, and evidence rows. Stale candidates must not be applied after the
effective revision changes.

Tunable windows, gates, weights, limits, and thresholds belong in versioned
configuration under `track-viz/configs/`, not Python or TypeScript literals.

Keep review decisions orthogonal:

- verdict: unreviewed, confirmed, false alarm, or uncertain
- zero or more confirmed failure labels
- optional explanation such as occlusion, entry/exit, out-of-frame, camera
  motion, localization noise, or other
- one optional primary correction target
- reviewer, timestamp, note, source/config/generator identity, and effective
  correction revision reviewed

Occlusion and out-of-frame are explanations, not verdicts. A decision write must
be idempotent and auditable.

## Source And Feature Capabilities

Prediction discovery requires a source explicitly classified as a tracker result
with usable track identities. Ground-truth-only sources, detection-only sources,
and sentinel/unusable identities must return disabled capability diagnostics
rather than partial candidates.

Additional evidence requires separate contracts:

- a detection input identifies its detector/checkpoint, source image hash,
  sequence geometry, frame mapping, row identities, and score semantics
- a ReID input identifies its model/checkpoint, preprocessing, embedding
  dimension and normalization, source rows, coverage, and artifact hash
- a prediction-plus-ground-truth audit pair validates sequence name, exact frame
  mapping, image geometry, source hashes, GT classes/marks/ignore regions, and
  mismatch behavior before evaluation

The current source selector loads one annotation source. Paired audit and
auxiliary detection/ReID evidence therefore require explicit APIs and must not be
inferred from similarly named local paths.

## Candidate Computation Lifecycle

Generation runs as a bounded, cancellable job keyed by source hash, candidate
configuration hash, generator name/version, and effective correction revision.
The initial implementation may keep completed results in process memory or an
ignored derived cache, but must never write beside or modify the source.

The API reports pending, ready, failed, canceled, and stale states. Pagination is
against one immutable completed snapshot with a deterministic tie-breaker, so
pages contain no duplicates or omissions. Exact totals are required only after
generation completes; a partial or capped job reports that its total is unknown.
Source replacement, configuration change, or correction-revision change cancels
or invalidates in-flight work and stale responses cannot replace the active job.
Define memory, result-count, temporal-window, and concurrency limits before
running on dense sources.

## Candidate Generation Strategy

### Geometry-first baseline

Deliver a useful baseline without inventing a ReID implementation:

1. **Gap and endpoint candidates**
   - expose internal same-ID gaps
   - rank a mid-sequence endpoint differently from an image-boundary or
     sequence-end exit candidate without claiming visibility
   - use motion and box-size continuity only as ranking evidence
2. **Fragment/restart candidates**
   - compare tracklet endpoints and starts within a bounded frame window
   - gate by predicted position, time separation, scale compatibility, and
     direction
   - keep multiple plausible successors visible when the scene is ambiguous
3. **Takeover and switch-risk candidates**
   - combine approved within-track discontinuity signals with nearby competing
     identities and interaction intervals
   - never call the result a confirmed identity switch without review or ground
     truth
4. **Two-way swap candidates**
   - require a paired hypothesis: two identities are present before and after an
     interaction and their most plausible continuations invert
   - group repeated frame-level signals into one bounded incident
5. **Duplicate candidates**
   - detect sustained, not merely momentary, high overlap with compatible motion
   - preserve both source identities and every supporting frame

Use spatial and temporal indexes to avoid unbounded all-track-pair comparison in
dense MOT20 scenes. Limit and paginate results, but never silently discard them;
return snapshot state and explicit cap/truncation diagnostics. Return an exact
total only for a completed enumeration.

### Appearance-assisted expansion

Add appearance evidence only after a ReID model, checkpoint, embedding contract,
and experiment provenance are established elsewhere in the repository. ReID
should rerank or strengthen candidates, not silently mutate identities. Report
raw distances, normalization, model identity, threshold, and missing-embedding
coverage. Similar-looking people in crowded scenes must remain reviewable as
ambiguous alternatives.

### Ground-truth audit mode

Where authorized ground truth exists, add a separate audit generator that:

- uses an explicitly selected matching/evaluation contract
- identifies misses, false positives, identity switches, and fragmentations
  using exact ground-truth and prediction identities
- distinguishes evaluator metric events from product review categories
- records the dataset split, source hashes, matching parameters, evaluator
  version, and artifact path
- never describes manually adapted MOT20 test results as held-out benchmark
  evidence

## Reviewer Experience

Add a bounded **Review inbox** rather than placing every signal directly in the
Focus controls.

The inbox should provide:

- category, sequence, interval, identity, review state, and reason filters
- deterministic ranking with visible reason chips
- counts for completed-snapshot total, displayed, reviewed, confirmed, uncertain,
  and false alarm
- keyboard navigation that opens the selected incident without losing list state
- explicit empty, unavailable-feature, pending, capped, loading, canceled, and
  stale states

Opening a candidate should show:

- a short before/during/after playback window centered on the incident
- focal and competing trajectories with stable colors and identity labels
- predecessor and successor crop strips aligned by frame
- the raw signal values that produced the candidate
- alternate continuations when more than one tracklet is plausible
- a verdict, any confirmed failure labels, an optional explanation, and one
  optional correction target
- an optional note and, later, a handoff to an applicable correction operation

The tracer bullet uses the existing Explore and Focus surfaces: the inbox opens
the candidate's first affected frame and Focus evidence, while the existing
transport remains responsible for playback. Candidate-aligned bounded playback
and crop strips are later features, not assumed current capabilities. Do not
introduce the future Review/Edit shell early merely to host the inbox. Editing
controls appear only when the correction plan's explicit Edit mode and revision
ledger exist. Reviewing a candidate must never modify the source prediction.

## Relationship To Existing Work

After product approval, create a separate geometry-first discovery epic using
this document as its specification. This plan refines the review-candidate
foundation that deferred assisted-proposal epic `mot-po1` may later consume; it
does not supersede or prematurely unblock that ReID-dependent epic.

Record these durable dependencies when the new epic is created:

- signal characterization depends on completion of review-event epic `mot-dev`
- durable review decisions depend on the correction-state contract in
  `mot-t1t.1`, but remain separate from correction commands
- correction handoffs depend on the applicable box-correction children under
  `mot-t1t` and identity-correction children under `mot-bm3`
- ReID-assisted ranking depends on an established ReID artifact contract and the
  proposal-input decisions under `mot-po1`
- ground-truth audit depends on approval of the paired-source and evaluator
  contract defined in this plan

Do not create or rewrite Beads work merely because this proposed plan is
committed. Create the epic and dependency links only after the approval
checkpoints below are resolved.

## Execution Plan

### Task 1: Approve Terminology And Review Decisions

Confirm the prediction-first assumption, observable candidate types, overlapping
confirmed labels, mapping of identity drop, verdict/explanation fields, and
primary correction target. Record examples and non-examples before changing
code.

### Task 2: Characterize Reusable Existing Signals

Complete the existing review-event characterization work for displacement,
scale, proximity, grouping, anchors, and navigation. Add no new candidate
behavior until each reused signal has deterministic boundary coverage and
approved user-facing meaning. This task depends on `mot-dev`.

### Task 3: Define Candidate Contracts And Synthetic Incidents

Create the versioned candidate/API contract and compact synthetic fixtures for:

- a valid occlusion gap
- a true missing-observation gap
- one person restarting under a new ID
- one ID taking over another person
- a two-person swap
- duplicate identities
- same-ID duplicate observations in one frame
- tied and multiple plausible successors
- an ordinary close interaction with no switch
- legitimate entry, exit, scale change, and camera-relative motion controls
- tracked-result, ground-truth, detection-only, and sentinel-ID capability cases

Likely create or modify:

- `track-viz/src/mot20/viewer/problem_candidates.py`
- `track-viz/src/mot20/viewer/api.py`
- `track-viz/src/mot20/viewer/server.py`
- `track-viz/tests/viewer/test_problem_candidates.py`
- `track-viz/tests/viewer/fixtures.py`
- `track-viz/configs/viewer.toml`

### Task 4: Geometry-First Fragment Tracer Bullet

Implement the smallest end-to-end path: a deterministic synthetic person ends
under one track ID, resumes under another after a short gap, and produces one
explainable cross-ID continuation hypothesis. Expose it through a bounded API
and a minimal read-only inbox row that opens the existing Focus evidence
interval.

This tracer bullet proves candidate identity, backend-to-frontend transport,
ranking reasons, bounded evidence, and navigation without requiring ReID,
persistence, or correction operations.

Likely create or modify:

- `track-viz/web/src/ReviewInbox.tsx`
- `track-viz/web/src/ReviewInbox.test.tsx`
- `track-viz/web/src/api.ts`
- `track-viz/web/src/App.tsx`
- `track-viz/web/src/styles.css`
- `track-viz/web/e2e/tracer.spec.ts`

### Task 5: Add Remaining Geometry-First Generators

Add gap/endpoint, takeover-risk, two-way-swap, and concurrent-duplicate
candidates one family at a time. Each family needs positive, ambiguous, and
hard-negative synthetic cases. Collapse repeated raw signals into one incident
while keeping every raw measurement available for audit. Localization and
false-positive generation remain deferred until a detection-input or
ground-truth contract supplies evidence beyond prediction geometry; reviewers
may still apply those labels manually once decision persistence exists.

### Task 6: Persist Review Decisions

Extend the approved local revision/ledger design with source- and
candidate-version-scoped reviewer decisions. Define deduplication, stale
candidate behavior, idempotency, transaction/crash recovery, schema migration,
correction relationships, notes, and portable export before adding persistent
controls. Store decisions in the same SQLite transaction discipline, but do not
encode them as correction commands or advance the effective correction revision.
Maintain an independent review-decision revision while recording which effective
correction revision was reviewed. This task depends on `mot-t1t.1`; it must not
introduce a competing persistence model.

### Task 7: Connect Confirmed Problems To Corrections

After the relevant correction operations exist, offer explicit handoffs:

- missing observation to add-observation review
- broken continuity to merge-fragment review
- takeover or identity merge to split/reassign review
- two-way swap to atomic interval-swap review
- duplicate or false positive to remove/exclude review
- localization error to update-box review

The reviewer must preview and confirm a correction separately. Confirming a
problem never applies a correction automatically. Each handoff depends on the
corresponding implemented operation under `mot-t1t` or `mot-bm3`.

### Task 8: Add ReID-Assisted Ranking

Once the ReID workstream supplies versioned embeddings and model provenance,
add appearance-distance evidence to fragment, takeover, swap, and duplicate
candidates. Evaluate its incremental value against the geometry-only baseline
before changing default ranking weights. This task also depends on the approved
proposal-input contract under `mot-po1`.

### Task 9: Add Ground-Truth Audit Mode

Integrate an approved evaluator or evaluation contract and map reproducible
evaluator-defined events into the same review evidence shape while retaining
evaluator-specific details.
Keep ground-truth audit results visually and semantically distinct from
prediction-only suggestions. First implement and approve the synchronized
prediction-plus-GT pairing contract, including hard failures for sequence, frame,
geometry, identity-capability, class/mark, and ignore-region mismatches.

### Task 10: Validate The Integrated Review Workflow

Run the applicable deterministic correctness, API, browser, accessibility,
performance, source-integrity, stale-revision, and provenance checks at each
milestone below. Use read-only real MOT20 observations to study candidate volume
and reviewer usability; do not convert those observations into claimed accuracy
without maintained labels and reproducible experiment records.

## Milestone Acceptance Gates

Do not hold the useful geometry baseline open for optional ReID or ground-truth
work that has no implementation yet.

1. **Contracts and characterization:** terminology, capabilities, computation
   lifecycle, synthetic incidents, and reused event semantics are approved.
2. **Fragment tracer bullet:** one cross-ID continuation hypothesis travels from
   deterministic fixture through API and inbox into existing Focus evidence;
   source state remains unchanged.
3. **Geometry discovery baseline:** all approved geometry families pass positive,
   ambiguous, hard-negative, dense-scene, pagination, cancellation, and
   source-replacement checks; real-data candidate volume is reported as an
   observation.
4. **Persistent review decisions:** idempotent verdict/label/explanation writes,
   crash recovery, migrations, export provenance, and correction-revision
   independence pass before persistent review is enabled by default.
5. **Correction handoffs:** each available operation has a separate preview and
   confirmation path; confirming an incident still performs no correction.
6. **ReID expansion:** geometry-only and appearance-assisted ranking are compared
   on a maintained labeled set with complete model and artifact provenance. This
   gate may remain deferred without blocking milestones 1-5.
7. **Ground-truth audit:** paired-source validation and evaluator-defined events
   reproduce exact fixtures and retained evaluator output. This gate may remain
   deferred without blocking prediction-only review.

## Verification

Use characterization-first and test-first execution because incorrect candidate
semantics can create misleading pseudo-labels.

- backend unit tests cover every calculation, equality boundary, gap, endpoint,
  same-frame duplicate observation, tied/multiple successors, exact one-based
  interval boundary, missing feature, deterministic ordering, cap, and
  truncation
- API tests prove source-hash and effective-revision scoping, stable candidate
  IDs, stable pagination without duplicates or omissions, configuration-version
  invalidation, explicit feature availability, and stale rejection
- capability tests distinguish tracked prediction results from ground truth,
  detection-only sources, and sentinel/unusable identities with exact diagnostics
- computation tests cover cancellation, memory/result caps, source replacement
  during generation or loading, and stale job/response rejection
- frontend unit tests cover filtering, list-state preservation, decisions,
  ambiguity, empty/error/stale states, and accessible names
- deterministic Playwright tests exercise each incident and hard-negative case
  on desktop and narrow review layouts
- axe scans report zero serious or critical violations for the inbox and incident
  review state
- performance reports include generation time, candidate count, memory, API
  latency, inbox rendering, and navigation latency on a declared dense source
- source manifests remain byte-identical before and after prediction-only review
- GT audit validation reproduces the selected evaluator's event attribution on
  small exact fixtures and covers sequence/geometry mismatch, GT mark/class
  filtering, and ignore-region behavior before any MOT20 metric report
- decision persistence tests cover idempotency, transaction failure, crash
  recovery, schema migration, review-revision advance, correction-revision
  independence, and same-origin/trusted-host enforcement
- review-decision exports record source variant and hash, candidate generator and
  configuration hash, thresholds, reviewer, timestamp, artifact path, and local
  or test-adapted policy classification
- candidate precision, recall, or ranking quality is reported only against a
  maintained labeled set with dataset split, source, generator configuration,
  model/checkpoint, command, and artifact provenance

Run the established focused and aggregate viewer gates from
`.claude/project/verification.md`; add stable root commands only when the new
implementation establishes them.

## Risks And Mitigations

- **Candidate explosion in crowds:** use bounded windows, spatial/temporal
  indexes, incident grouping, pagination, totals, and explicit truncation.
- **False certainty without ground truth:** say candidate or risk, expose every
  reason, and require a reviewer or GT audit to confirm.
- **Legitimate occlusion or exit mislabeled as a drop:** include occlusion and
  out-of-frame explanations plus negative fixtures.
- **Camera motion and perspective distort geometry:** avoid one global threshold;
  preserve raw values and support sequence-calibrated configuration only with
  recorded provenance.
- **Similar clothing weakens ReID:** present alternate matches and uncertainty;
  never auto-merge identities.
- **Threshold tuning leaks MOT20 test knowledge:** label local test-adapted
  decisions and keep any future held-out benchmark workflow separate.
- **Metric-definition drift:** keep evaluator events distinct from product labels
  and version the matching contract.
- **Stale evidence after corrections:** scope candidates to source and effective
  revision, reject stale actions, and regenerate explicitly.
- **Competing persistence models:** depend on the approved correction ledger for
  durable decisions instead of introducing an unrelated store.
- **UI density and performance regressions:** keep the viewport dominant,
  virtualize long lists when measured, and preserve the current canvas path.

## Non-Goals

- no automatic correction or silent identity reassignment
- no claim that abrupt displacement, scale, proximity, or confidence alone is a
  tracking error
- no ReID placeholder values or model claims before that workstream exists
- no requirement for ground truth in the default workflow
- no overwrite of images, predictions, annotations, embeddings, or exports
- no official benchmark claim from manually reviewed or test-adapted data
- no attempt to replace general-purpose annotation tools such as CVAT

## Approval Checkpoints

Before implementation, approve:

1. prediction-only review as the primary workflow
2. the failure taxonomy and mapping of track switching and identity drop
3. reviewer decision states and whether notes are required
4. the geometry-first fragment tracer bullet
5. whether review-decision persistence waits for the shared correction ledger
