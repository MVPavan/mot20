# ReID-Guided Track Cleaning Requirements

Date: 2026-08-31

Status: Brainstorm in progress; requirements are not yet approved for implementation.

## Context

The available MOT20 test predictions were produced by BoostTrack++ with ReID enabled. They cover test sequences `MOT20-04`, `MOT20-06`, `MOT20-07`, and `MOT20-08` under `datasets/test_tracks/`.

The objective is not to use the predicted track IDs directly as ReID training labels. The tracks must first be audited, cleaned, split, or merged so that the resulting pseudo-identities are more reliable.

## Goal

Build a reviewable process that uses per-crop ReID embeddings to find identity errors within predicted tracks and fragmented identities across predicted tracks. Suspicious cases must be inspectable by a human. Whether every correction or only uncertain/high-risk corrections require explicit approval remains an open policy decision.

## Proposed workflow

1. Read the MOT-format test predictions and corresponding MOT20 test images.
2. Extract person crops from a selected, quality-controlled track variant; the source variant and row-selection policy remain unresolved.
3. Compute a ReID embedding for every crop.
4. Analyze embedding consistency within each track.
5. Compare track-level embedding distributions across tracks in the same sequence.
6. Rank suspicious crops, possible track splits, and possible track merges.
7. Present the evidence for manual review.
8. Save identity corrections, quality decisions, uncertainty, and any required review outcomes without changing the source predictions.
9. Generate ReID training examples only from the resulting cleaned identities.

## Within-track analysis

The process should detect isolated crop outliers and sustained temporal changes between identity modes within a track. It must not assume that one identity always dominates: a corrupted track may contain two or more similarly sized identity segments. Possible causes include:

- an identity switch
- another person temporarily assigned to the track
- an incorrect or drifting bounding box
- occlusion or a low-quality crop
- a legitimate appearance change, viewpoint change, or scale change

A plain arithmetic centroid may be distorted by the same outliers it is intended to detect. The analysis should therefore consider robust representations such as a medoid, trimmed centroid, local temporal prototypes, or multiple appearance modes.

The review evidence should retain temporal order so that a reviewer can distinguish an isolated bad crop from a sustained identity switch and identify a possible split frame.

## Cross-track analysis

Tracks within the same MOT20 sequence should be compared to detect:

- one person fragmented into multiple track IDs
- track fragments separated by an occlusion or missed detections
- duplicate tracks following the same person at overlapping frames
- visually similar people who must remain separate identities

Track comparison should use more than the distance between two mean centroids. Useful evidence includes representative crops, closest crop pairs, robust prototypes, distance distributions, frame ranges, temporal overlap, spatial continuity, and full-frame context.

Numeric track IDs are local to a sequence. Tracks from different MOT20 sequences must not be merged merely because their numeric IDs or embeddings are similar.

Temporal constraints must remain visible during review:

- non-overlapping tracks separated by a plausible temporal and spatial gap are merge candidates
- tracks present at the same time are normally different people
- simultaneous tracks may still represent duplicate predictions, so this is a strong constraint rather than an unconditional rule

## Manual-review decisions

The process should support at least these outcomes:

- keep the track unchanged
- remove individual low-quality or incorrect crops
- split a track at an identified frame or interval
- merge two track fragments into one pseudo-identity
- mark two overlapping tracks as the same identity while deduplicating their overlapping observations
- classify a cross-track candidate as same person, different person, or uncertain
- mark a case as uncertain
- exclude a track or crop from ReID training

Every saved decision should preserve its source sequence, track variant, original track ID, frame range, original bounding boxes, action, uncertainty, and review reason.

## Track variants

The available result families are raw, linearly interpolated `post`, and smoothed `post_gbi` tracks.

One hypothesis is that raw tracker observations may be the safest initial crop source because they correspond to actual detector/tracker observations. Interpolated rows may cover occlusion gaps without a clearly visible person, while GBI changes the bounding-box geometry even for retained frame/ID entries. This has not been selected as a requirement.

The final choice must be validated rather than assumed. Comparing raw, post, and post-GBI boxes on ground-truth data for localization quality, crop quality, identity purity, and downstream ReID performance is a candidate experiment, not yet an approved mandatory benchmark.

## Ground-truth validation

Ground-truth MOT20 tracks can validate embedding consistency, crop handling, and same/different identity scoring. However, ground-truth tracks alone do not contain the tracker errors that the cleanup process is meant to find.

The stronger diagnostic validation is to obtain BoostTrack++ predictions on a MOT20 split with ground truth, such as `val_half/`, and then:

1. Run the proposed cleanup process without using ground-truth identities as inputs.
2. Match predicted boxes to ground-truth identities per frame using a documented assignment protocol.
3. Measure whether proposed crop removals, track splits, and track merges agree with ground truth.
4. Report track purity, false-positive and false-negative identity pairs, outlier-detection quality, and merge/split decision quality.

The matching protocol must eventually define IoU thresholds or costs, assignment behavior, ignored regions, visibility filtering, unmatched predictions, and unmatched ground-truth boxes. Controlled synthetic identity switches, fragmentation, duplicate tracks, and crop outliers may also be injected into ground-truth tracks to test known failure cases.

This validation may be diagnostic rather than an unbiased performance estimate. If the detector, tracker, or ReID model was trained on the complete MOT20 training set, then `val_half/` is not held out from that component. Validation must record complete model-training provenance and distinguish diagnostic error analysis from held-out generalization claims.

## Risks and safeguards

- A predicted track can contain an identity switch, producing false same-person pairs.
- One person can be fragmented across tracks, producing false different-person pairs.
- Different track IDs visible in the same frame are useful negative evidence, but duplicate predictions remain possible.
- Appearance can be multi-modal within one true identity, so centroid distance alone must not automatically remove crops or split tracks.
- BoostTrack++ already used ReID. Auditing it with the same embedding model may reproduce or reinforce its original similarity errors.
- The original images, tracks, detections, and delivered artifacts must remain immutable. Derived crops, embeddings, proposed corrections, and reviewed labels must use separate output paths.
- Every derived artifact must retain sequence, track variant, frame, original bounding box, embedding model/checkpoint, preprocessing configuration, and decision/reviewer provenance.
- Whether automatic low-risk corrections are permitted or all corrections require human approval remains unresolved until ground-truth validation is available.
- Fine-tuning on manually cleaned MOT20 test identities is test-set adaptation. Its acceptability for MOTChallenge submissions and comparability with published results must be established before such a model is trained or submitted.

## Confirmed decisions

- The source predictions are BoostTrack++ tracks created with ReID enabled.
- The purpose is track and identity cleaning before ReID training, not direct crop extraction for training.
- Both within-track outliers and cross-track identity matches must be investigated.
- Suspicious cases must be manually viewable.
- Ground-truth data will be used to validate the method before applying its decisions to MOT20 test pseudo-labels.
- Source datasets and predictions will not be overwritten; all crops, embeddings, decisions, and cleaned identities will be derived artifacts with provenance.
- Implementation is intentionally deferred while the remaining design choices are discussed.

## Open questions

- Which ReID model or ensemble will produce the audit embeddings?
- Which track variant supplies crop boxes: raw, post, post-GBI, or a validated combination?
- Is test-set adaptation permitted for the intended research or MOTChallenge submission, and how will adapted results be labeled for fair comparison?
- Which validation results are diagnostic, and what separate split or identity protocol is needed for unbiased performance evaluation?
- What crop-quality, visibility, size, and occlusion filters are required?
- Which robust track representation and outlier score should be used?
- How should temporal and spatial feasibility affect cross-track similarity?
- What ground-truth matching and synthetic-corruption protocols will validate split, merge, duplicate, and outlier decisions?
- What thresholds create automatic exclusions versus manual-review candidates?
- Must every correction receive human approval, or only uncertain and high-risk cases?
- What review interface and correction-file format should be used?
- How will corrected pseudo-identities be converted into ReID training and evaluation splits without leakage?

## Non-goals for the current discussion phase

- Do not implement crop extraction, embedding inference, clustering, or a review interface yet.
- Do not modify or overwrite the supplied tracks, images, detections, or embeddings at any later phase; derived artifacts must be written separately.
- Do not select thresholds or claim expected accuracy before validation evidence exists.
- Do not start ReID fine-tuning until the cleaning and review policy is approved.
