# RF-DETR 2XL MOT20 Detector Training Plan

## Goal

Fine-tune RF-DETR-2XL for one-class pedestrian detection using the ByteTrack-style
MOT20 plus CrowdHuman data mixture and the curated Byte65 supplemental images.
Produce detector outputs that can be passed to ByteTrack without coordinate or
confidence ambiguity.

The preferred model is RF-DETR-2XL. RF-DETR-Large is the fallback only if the
selected real-data configuration cannot satisfy the predeclared 2XL feasibility
gate. This plan does not authorize data conversion or a full training run until
every blocking gate below has recorded passing evidence.

## Approved Decisions

- Use `RFDETR2XLarge` via `rfdetr_plus`; PML 1.0-derived weights are
  research-only and must not enter a deployment or commercial path without a
  separate license review.
- Train exactly one class: `pedestrian`. The model uses `num_classes: 1` and
  all source conversion filters annotations to that class before training.
- Match ByteTrack's detector-data strategy: combine the leakage-safe MOT20
  training partition and CrowdHuman training data in one COCO-format dataset.
- Include the 21 exhaustively human-audited images and post-modification labels
  in `datasets/zip-files/byte65-modified-images-yolo.zip` in a separately named
  `local_test_adapted` baseline. The clean MOT20/CrowdHuman baseline remains
  immutable for held-out validation and comparison. Because the images are from
  MOT20 test sequences, the Byte65-including baseline is local research and is
  never a held-out MOTChallenge result.
- Preserve raw datasets, weights, checkpoints, generated COCO JSON, logs, and
  predictions as ignored local artifacts in unique, never-overwritten run-ID
  directories.

## ByteTrack Reference and Deviation Review

The pinned ByteTrack checkout at `repos/ByteTrack` commit
`d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` is the default reference for this
work. Follow its MOT20 workflow without a separate review when the behavior is
directly applicable or is a common implementation practice also used there:

- `exps/example/mot/yolox_x_mix_mot20_ch.py` for the one-class detector,
  MOT20/CrowdHuman mixture, temporal halves, evaluation cadence, and detector
  confidence/NMS operating-point reference;
- `tools/convert_mot20_to_coco.py`, `tools/convert_crowdhuman_to_coco.py`, and
  `tools/mix_data_test_mot20.py` for source field mapping, category treatment,
  image identity, and mixing behavior; and
- `yolox/tracker/byte_tracker.py` for tracker input, score bands, and
  association behavior.

Before implementation, pause for user review of any material departure from
that reference: source inclusion or label treatment, train/validation split,
augmentation or sampling policy, image geometry, training schedule or batch
semantics, evaluation limits/metrics, score/NMS policy, or tracker handoff.
Record the reference behavior, reason, expected impact, and validation for each
approved departure. The following are already identified as RF-DETR-specific
departures and therefore require that review before their implementation:

- query-capacity selection (`num_queries`, `num_select`, and `group_detr`)
  instead of YOLOX `max_labels` limits;
- RF-DETR square-resolution preprocessing and original-pixel inverse mapping
  instead of YOLOX's 896x1600 resize pipeline;
- RF-DETR-native optimization, augmentation, AMP, batch, and schedule settings
  instead of the YOLOX 80-epoch, eight-GPU recipe; and
- RF-DETR raw top-$Q$ output plus a separately calibrated ByteTrack threshold
  policy instead of the YOLOX detector's `test_conf=0.001` and `nmsthre=0.7`.

### Geometry Investigation

RF-DETR-2XL's installed configuration has `patch_size: 20` and
`num_windows: 2`, requiring each explicit inference dimension to be divisible
by 40. Its shape validator rejects ByteTrack's exact 896x1600 input because
896 is not divisible by 40. The approved first tracer-run geometry is RF-DETR's
native `resolution: 880` path: aspect-preserving resize to an 880px short side,
a 1333px long-side cap, and collator padding to the required block size. It does
not distort MOT20 frames to 880x880 and does not introduce a custom rectangular
training path. Any later fixed rectangular or model-level geometry change remains
a material deviation requiring review before implementation.

### Approved RF-DETR Departures

On 2026-09-03, the following material departures were reviewed and approved:

- implement true loss-ignore support for ignored MOT20 and CrowdHuman people;
- use the approved native RF-DETR 880px aspect-preserving input pipeline after
  finding ByteTrack's 896x1600 shape incompatible with RF-DETR-2XL;
- retain RF-DETR top-$Q$ outputs for detector evaluation and tune the tracker
  score/NMS policy only on MOT20 `val_half`; and
- use RF-DETR-native optimization, augmentation, batch, and schedule settings
  after the final-capacity probe rather than translating YOLOX's 80-epoch
  recipe.

## Data Contract

The canonical training representation is COCO JSON, consistent with the
ByteTrack conversion and mixing scripts:

- One category: `{"id": 1, "name": "pedestrian"}`.
- Image records preserve native width, height, source path, source dataset,
  sequence, frame number, and split role where applicable. MOT20 uses the
  first and second temporal halves emitted by the pinned ByteTrack converter
  as `train_half` and `val_half`; no image belongs to both. The mixed training
  manifest contains `train_half` plus both ByteTrack CrowdHuman `train` and
  `val` sources. `val_half` is the immutable detector selection split.
- Annotation boxes use absolute `x, y, width, height` pixels. Conversion
  preserves the source box as provenance, clips the training box to native
  bounds before serialization, rejects only non-finite or non-positive-area
  post-clip boxes, and records every clip/rejection reason by source. No rows
  may be silently dropped.
- MOT20 conversion reproduces the valid-pedestrian filtering in
  `repos/ByteTrack/tools/convert_mot20_to_coco.py` at pinned commit
  `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`: confidence field `7 == 1`,
  category field `8 == 1`, and no visibility threshold. Non-person and ignored
  person rows are logged separately from retained positives.
- CrowdHuman conversion uses full-body `fbox` as `bbox`, preserves `vbox` as
  provenance metadata where supported, and records the ByteTrack converter's
  `extra.ignore == 1` mapping. RF-DETR 1.9.4 currently filters `iscrowd` rows
  before constructing loss targets, rather than using them as loss-ignore
  regions. Conversion is blocked until a fixture verifies the installed loss
  behavior and an explicit source-specific treatment prevents ignored people
  from becoming unmeasured false-negative supervision; the options are a
  supported loss-ignore implementation, excluding affected images, or an
  approved replacement policy with measured impact.
- Byte65 uses only `post_modification_annotations/`. Its normalized YOLO rows
  are converted to the same absolute COCO box convention. Baseline and delta
  label files remain audit-only evidence. On 2026-09-04, the user confirmed an
  exhaustive human audit for every selected image; the dedicated Byte65 manifest
  records that user-confirmed audit and is eligible only for the explicitly
  named `local_test_adapted` baseline. Do not use unsupported `iscrowd` regions
  to stand in for missing labels.
- `byte65nms_68seq` is the separately materialized local source dataset from
  that post-modification archive. Its 2026-09-04 structural audit verified
  archive/CVAT label parity and in-bounds output geometry; the user separately
  confirmed the exhaustive human semantic audit. It contains MOT20 test
  sequences `MOT20-06` and `MOT20-08`, so it may only enter the named local
  test-adapted mix. See
  `finetuning/docs/experiments/2026-09-04-byte65nms-68seq-audit.md`.
- Every split and mixed manifest is content-hashed and records source revision,
  conversion revision, image-byte duplicate checks, source counts, label counts,
  density statistics, and every Byte65 policy field from `docs/MOTPolicy.md`.

## Model and Capacity Contract

RF-DETR is query-based, so it has no literal uncapped detection mode. Its
capacity must be selected from observed annotation density:

$$
Q > \max_{i \in \text{labeled training and validation images}} \left|\text{loss-participating pedestrians}(i)\right|
$$

where $Q$ is `num_queries`; set `num_select = Q` for high-density detector
evaluation and `eval_max_dets >= Q`. In installed RF-DETR 1.9.4,
`num_queries` is the total evaluation query count and must be divisible by
`group_detr`; each training group receives `Q / group_detr` queries. Retain
`group_detr: 13` unless a recorded source/API validation supports an
alternative. Training produces `Q * group_detr` predictions and evaluation
produces `Q`.

Before training, audit loss-participating and ignored labels separately for
MOT20 and CrowdHuman and labels for Byte65. The preflight must fail if any
labeled training or validation image has at least $Q$ loss-participating labels.
It must report percentiles, maxima, and source image IDs. Test images have no
official ground truth: record output saturation at `num_select = Q`, never a
ground-truth capacity claim. No confidence threshold or NMS may remove
detections before detector AP evaluation.

## ByteTrack Integration Contract

The RF-DETR adapter must provide the pinned ByteTrack implementation at
`repos/ByteTrack` commit `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` one
per-frame array of shape `N x 5`: `x1, y1, x2, y2, score`.

- Coordinates are `float32`, absolute original-image pixels, clamped to valid
  image bounds, and satisfy `x2 > x1`, `y2 > y1`.
- `score` is the RF-DETR single-class sigmoid confidence in `[0, 1]`.
- Every retained output has class `0` (`pedestrian`); class IDs are validated
  before the class-less ByteTrack array is created.
- Original frame width and height travel with the detections. RF-DETR output is
  already in original-image pixels; the ByteTrack call must use
  `img_info=(height, width)` and `img_size=(height, width)` so its internal
  `scale` is exactly `1`, or use an equivalently tested adapter that does not
  rescale. A non-square, resize-and-padding inverse fixture must prove this.
- Preserve MOT20 sequence names and 1-based frame IDs. The versioned raw export
  records sequence, frame, native dimensions, float32 boxes/scores, checkpoint
  hash, config hash, and explicit empty frames. It remains separate from the
  tracker-filtered output and provenance record.
- Detector evaluation exports the top $Q$ outputs with no score floor or NMS.
  Tracking begins from that raw export, applies a validation-selected score
  floor that retains ByteTrack's low-score association range, and uses no NMS
  unless a held-out duplicate-rate diagnostic exceeds its predeclared limit.
  Any class-agnostic NMS fallback must pin its IoU value, run only after score
  selection, and be selected on `val_half`, never MOT20 test data. ByteTrack
  high/low/new-track thresholds and duplicate policy are independently tuned,
  recorded, then frozen before test-adapted inference.

## Execution Phases

### 1. Accepted baseline and API-contract preflight

1. Accept the completed evidence in
  `finetuning/docs/experiments/2026-09-03-rfdetr-2xl-preflight.md`: RF-DETR 1.9.4,
  RF-DETR+ 1.0.2, the checkpoint MD5, Python/PyTorch/CUDA versions, one-GPU
  forward, and one-image smoke train.
2. Put the mandatory `LD_LIBRARY_PATH` export and the current single-GPU
  `CUDA_VISIBLE_DEVICES=0` plus `device="cuda"` workaround in the versioned
  run launcher. Revalidate this contract before dependency changes; do not
  claim multi-GPU support until a separate distributed launch, checkpoint
  resume, finite-loss, and per-rank-batch validation passes.
3. Verify from the installed source and a construction/forward/backward/eval
  fixture the selected `num_queries`, `num_select`, `group_detr`,
  `eval_max_dets`, one-class category-ID-to-index mapping, and postprocessor
  behavior. The fixture must construct COCO category `1` as class index `0`.
4. Load the selected initialization checkpoint and record its SHA-256 in
  addition to the verified publisher MD5. Assert that missing, unexpected, or
  shape-mismatched keys are restricted to the intentional one-class head and
  capacity-dependent query parameters.
5. This phase proves API and baseline loadability only. It cannot select 2XL
  capacity, batch size, or a multi-GPU strategy.

### 1.1 Ignored-annotation support

The project-owned implementation is complete for box-only RF-DETR 1.9.4
detection. `finetuning/src/mot20/detection/ignore_dataset.py` preserves COCO `iscrowd`
boxes through the existing RF-DETR transform pipeline as normalized
`ignored_boxes`; `ignore_criterion.py` leaves them out of Hungarian matching and
box regression while excluding unmatched query/classification-loss terms that
overlap an ignored box at the configured IoU threshold. The scoped context in
`rfdetr_integration.py` temporarily substitutes these components for a complete
`RFDETR.train()` call and always restores the installed factories afterward.

The implementation supports RF-DETR 1.9.4's default IA-BCE classification loss
and its ordinary sigmoid focal-loss mode. It rejects segmentation, keypoint,
varifocal, and position-supervised configurations while ignored boxes are
present, rather than silently providing incomplete masking. Focused synthetic
tests cover transformed-box preservation, focal and IA-BCE masking, factory
construction, and factory restoration. A real tracer run remains required to
verify the full model/data/trainer path before full training.

### 2. CrowdHuman acquisition and verification

1. Download the pinned revision of `sshao0516/CrowdHuman` from Hugging Face.
2. Retain the original archive names and ODGT annotations expected by
   ByteTrack: three training image archives, validation image archive,
   `annotation_train.odgt`, and `annotation_val.odgt`.
3. Record source repository revision, each archive SHA-256, extraction path,
   and file count. Retain the pinned revision already verified by preflight.
4. Verify the original expected splits: 15,000 training images and 4,370
   validation images; validate every ODGT image ID resolves to an image.

### 3. Deterministic dataset construction

1. Add reusable conversion code under `finetuning/src/mot20/detection/` and thin entry
  points under `finetuning/scripts/`; pin the ByteTrack commit and conversion reference
  paths in code and manifests.
2. Build separate COCO manifests for MOT20 `train_half` and `val_half`,
  CrowdHuman train and validation, Byte65, a clean mixed training manifest,
  and a separate Byte65-overlay mixed manifest. The clean mix follows
  ByteTrack by including both CrowdHuman splits; only the clean mix is eligible
  for held-out validation claims.
3. Give each source globally unique image and annotation IDs without renaming
  original sequence/frame identity. Check image-byte duplicates across every
  train/evaluation boundary and block manifest approval on an unresolved one.
4. Emit immutable source and split manifests with counts, category counts,
  density percentiles/maxima and image IDs, source checksums, source sampling
  weights, conversion revision, clipping/rejection counts, and policy fields.
5. Characterization tests cover empty images, exact ByteTrack MOT20 filters,
  ignored MOT20 rows, non-person rows, ignored CrowdHuman `fbox` rows, COCO
  category `1` to class `0`, YOLO normalization, clipping/invalid boxes,
  duplicate IDs/content, sequence/frame preservation, and temporal split
  separation. Add a trainer-level ignored-annotation fixture; block this phase
  if its verified RF-DETR loss behavior lacks an approved false-negative policy.
6. Preserve the user-confirmed Byte65 exhaustive-label audit in the dedicated
  test-adapted manifest. Its manifest and every downstream run report must contain test sequences,
  source detection/track variant, manual-review effects, label-production and
  cleaning procedure, thresholds/reviewer decisions, adaptation iterations,
  artifact paths, and metric classification.
7. Select a $Q$ satisfying the capacity contract, then run a real dense-batch
  2XL probe using final resolution, augmentations, AMP dtype, selected $Q$,
  physical batch, accumulation, backward pass, and optimizer state. Record
  VRAM headroom, images per second, finite loss, and projected epoch duration.

### 4. RF-DETR configuration and training

1. Create a versioned 2XL detection config with `num_classes: 1`, named class
  `pedestrian`, the Phase 3 capacity value, the approved clean/overlay manifest
  ID, and a distinct ignored output directory. Before a full run, review and
  freeze resolution, AMP dtype, physical/global batch, accumulation,
  optimizer/layer-wise learning rates, source sampling/repeat factors,
  steps-per-epoch, LR scaling rule, multi-scale behavior, augmentation, EMA,
  epochs, validation cadence, seed/worker seeding, determinism flags,
  checkpoint cadence, selection metric, tie-breaker, and stop rule.
2. Start from RF-DETR-native settings rather than copying YOLOX values. The
  ByteTrack recipe is a data-strategy comparison only, not an RF-DETR
  hyperparameter prescription.
3. Run a small tracer-bullet training slice and validate data loading,
  checkpoint restore, metric emission, detector output contract, finite loss,
  VRAM headroom, and projected runtime. Full training is blocked unless its
  predeclared feasibility threshold passes; otherwise record the failure and
  review the 2XL-to-Large fallback.

### 5. Detector evaluation and ByteTrack handoff

1. Evaluate the selected checkpoint on immutable MOT20 `val_half`. Record the
  exact evaluator `maxDets`, area ranges, category mapping, and metric names;
  label results above standard COCO limits as custom high-density metrics and
  report standard COCO metrics separately. Record saturation at $Q$.
2. Export raw per-frame detections and structurally validate the adapter
  against native image dimensions, zero and $Q$ detections, invalid classes,
  explicit empty frames, and a multi-frame sequence with no double scaling.
3. Tune detector score-floor, duplicate diagnostic/NMS policy, and ByteTrack
  high/low/new-track/match/buffer parameters only on `val_half`; retain score
  histograms and freeze the selected tracker configuration before inference on
  test-derived material.
4. Feed validated detections into ByteTrack and preserve a separate tracker
  configuration/provenance record. Label Byte65-overlay runs `local`,
  `test-adapted`, and `not leaderboard-comparable`, with every reporting field
  required by `docs/MOTPolicy.md`.

## Risks and Gates

| Risk | Gate |
| --- | --- |
| RF-DETR-2XL checkpoint cannot be retrieved or loaded | Stop and review 2XL evidence; choose Large only after review. |
| 2XL cannot fit the 24 GB GPUs at viable resolution/batch | Probe final real-data capacity after density selection; require finite loss, predeclared VRAM headroom, and projected runtime before full training. Review Large only after a recorded failure. |
| More pedestrians than query capacity | Raise $Q$ to a divisible data-backed value and rerun the final probe; never silently truncate. Test images use output-saturation evidence, not unavailable ground truth. |
| Third-party CrowdHuman mirror differs from ByteTrack's source layout | Verify ODGT plus image IDs, split counts, archive hashes, and converter behavior before mixing. |
| Ignored annotations create false-negative supervision | Prove installed loss semantics with a fixture and stop until a source-specific approved handling policy exists. |
| Test-derived Byte65 samples contaminate a benchmark claim | Maintain a clean default manifest, an opt-in overlay, exhaustive-label audit, and policy-complete test-adapted reports. |
| Coordinate transform mismatch at handoff | Use ByteTrack identity scale or a tested adapter; unit-test non-square resize/padding inversion and a multi-frame path. |
| Runtime or distributed assumptions drift | Use the pinned launcher for the single-GPU baseline. Treat multi-GPU as unavailable until its distinct validation passes. |
| 2XL license conflicts with a deployment path | Keep all PML 1.0-derived weights research-only pending license review. |

## Verification Evidence

- Baseline: preflight report, package/checkpoint URL or revision, MD5 and
  SHA-256, successful import/model load/synthetic forward, and the pinned
  launcher contract.
- Data: archive SHA-256 values, expected CrowdHuman split counts, ODGT image-ID
  resolution, ByteTrack commit and converter paths, COCO schema checks, split
  hashes, duplicate checks, annotation treatment, and source/density manifests.
- Capacity: observed label distributions and image IDs, chosen $Q$, proof of
  RF-DETR schema semantics, labeled-split capacity confirmation, test-output
  saturation, evaluator limits, and final real-batch probe evidence.
- Training: immutable config and manifest hashes, code/dependency revision,
  seed/worker/determinism settings, device/precision, command, checkpoint-key
  audit, checkpoint paths, and actual metrics.
- Handoff: versioned raw schema, geometry/empty-frame/class fixtures, frozen
  tracker thresholds and duplicate policy, and a ByteTrack run with identity
  scaling for original geometry.

## Next Milestone

Retrieval, environment baseline, and the bounded synthetic smoke run are
completed and recorded in `finetuning/docs/experiments/2026-09-03-rfdetr-2xl-preflight.md`.
Before dataset conversion, complete the Phase 1 source/API fixtures and the
Phase 3 annotation, split, ignore-semantics, and Byte65 audit gates. The 2XL
feasibility decision follows the selected-capacity real-batch probe, not the
synthetic smoke run.