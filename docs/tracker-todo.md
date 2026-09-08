# Tracker Workstream TODO

Living checklist for swapping this repository's RF-DETR detector into
BoostTrack++ and evaluating the result. `bd` is not currently on `PATH`, so this
file is the durable task record, following the precedent of
`finetuning/docs/status/2026-09-04-finetuning-blockers-todo.md`.

Authoritative context, contracts, and results index:
`docs/tracker-experiments.md`. Work that follows from the Phase 8b root-cause
analysis is tracked in `docs/tracker-improvements.md`. Update all three together.

Status legend: `[ ]` open, `[x]` done, `[~]` in progress, `[!]` blocked.

## Plan at a Glance

| Phase | Goal | Status |
| --- | --- | --- |
| 0 | External assets acquired and verified; contracts and docs written | Done |
| 1 | Tracking environment separate from the training `.venv` | Done |
| 2 | Layered artifact store under `artifacts/tracking/` with manifests | Done |
| 3 | Correct cache keying so variant swaps cannot silently reuse stale data | Done |
| 4 | Confirm RF-DETR inference geometry by reproducing $mAP_{50:95}=0.6202$ | Done (0.6209) |
| 5 | Export L1 detections at a low threshold, with structural validation | Done |
| 6 | Precompute L2 embeddings per detection variant and ReID variant | Done |
| 7 | Standalone runner, scale fix, non-strict OSNet load, embedding routing | Done |
| 8 | Stage 1 `val_half` runs against the `yoloxx20` baseline, plus TrackEval | Done |
| 8b | Root-cause the HOTA deficit despite better detection | Done |
| 9 | Stage 2 MOT20 `test` tracks, labelled `local_test_adapted` | Blocked on a decision |

Phases 3 and 4 are gates rather than steps: the grid is not trustworthy until
cache keying is correct, and no bulk detection export should run until the
geometry check passes.

## Phase 0 — External assets and documentation

- [x] Map BoostTrack detection, ReID, run, and output contracts from source.
- [x] Acquire `osnet_ain_ms_d_c_wtsonly.pth` (8,957,365 bytes, sha256
      `482f0944…d3c8440a`) from GitHub raw.
- [x] Verify it loads into `osnet_ain_x1_0`: 512-d output, finite, no
      unexpected keys, only `classifier.*` missing.
- [x] Acquire `mot20_sbs_S50.pth` (315,810,563 bytes, sha256
      `d3a39a1a…84180656`) from the BoT-SORT authors' published link.
- [x] Verify it is a MOT20-trained FastReID SBS ResNeSt50: `heads.weight`
      `(414, 2048)`, GeM pooling, BNNeck, deep stem, Split-Attention blocks.
- [x] Write `docs/tracker-experiments.md` and register it in the docs index.
- [x] Write this TODO and register it in the docs index.

## Phase 1 — Tracking environment

- [x] Create a tracking virtual environment separate from the detector training
      `.venv`, so the environment that produced the detector is untouched.
- [x] Install `torchreid`, `lap`, `cython_bbox`, and FastReID's dependencies.
- [x] Instantiate `osnet_ain_x1_0` through the installed `torchreid` package and
      confirm it matches the standalone-module verification already performed.
- [x] Instantiate `mot20_sbs_S50.pth` through `fast_reid.build_model` and record
      the embedding dimension. This promotes it from structurally verified to
      verified working.
- [x] Confirm vendored TrackEval runs end to end on any existing tracker output.

## Phase 2 — Artifact store

Layered artifact contract is specified in `docs/tracker-experiments.md`. Root is
`artifacts/tracking/`; embeddings are precomputed for every exported detection
so they are independent of tracker thresholds.

- [x] Create the `artifacts/tracking/` level layout and the `manifest.json`
      schema recording upstream variant IDs and checksums at each level.
- [x] Define and freeze the variant slug vocabulary for detector, ReID, and
      tracker configuration.
- [x] Register `yoloxx20` as a detector variant pointing at the existing
      `datasets/val_half/<seq>/det_yoloxx20/` files, so the published detector
      is a cell in the same grid rather than a special case.

## Phase 3 — Cache-keying correctness (precondition for any grid work)

- [x] Resolve the embedding cache hazard. Upstream's
      `./cache/embeddings/{video}_embedding.pkl` is keyed by sequence alone, so a
      ReID swap over unchanged detections silently returns the previous model's
      vectors. Rather than rekeying that cache, it is bypassed entirely: vectors
      are served from the L2 store, which is keyed by detection variant and ReID
      variant, and the extractor clears the in-memory cache and never writes it.
- [x] Detections are read per variant from the L1 store, so the upstream
      detector cache is likewise unused.
- [x] Add a regression check that a deliberate variant change invalidates the
      cache rather than silently reusing it.
- [x] Confirm the ECC transform cache stays keyed by sequence only. It is
      image-derived and correctly shared across every permutation.

## Phase 4 — Detector inference geometry

- [x] Determine whether `RFDETR.predict` reproduces the training-time
      aspect-preserving 1120px geometry, or defaults to a square resize.
- [x] Verification gate: run the epoch-5 checkpoint over `val_half` and compute
      $mAP_{50:95}$. Reproducing **0.6202** proves the inference geometry matches
      training. Do not dump detections for 4,463 frames until this passes.
- [x] NMS decision. Exported without NMS, then swept as derived variants. Greedy
      suppression at IoU 0.7 is best and matches the `nmsthre` used to produce
      the published YOLOX detections. `num_select = 390` saturation was observed
      but is not the limiting factor.

## Phase 5 — Detection export (L1)

- [x] Export `val_half` detections at a low threshold (proposed 0.05). A low
      threshold is required: BoostTrack's `dlo`/`duo` confidence boosting runs
      before the `det_thresh` cut, so exporting only above-threshold boxes
      silently degrades BoostTrack++ toward BoostTrack.
- [x] Write per-variant manifests with detector checkpoint sha256, threshold,
      geometry, and frame completeness.
- [x] Validate structurally: every frame present including empty ones, boxes
      within image bounds, scores in range, 1-based frame IDs.

## Phase 6 — Embedding export (L2)

- [x] Precompute embeddings for all exported detections for each ReID variant,
      keyed by detection variant × ReID variant.
- [x] Record embedding dimension, crop size, and whether grid splitting was
      enabled, since these change the vectors.
- [x] Validate row alignment against the corresponding detection file.

## Phase 7 — Tracker integration

- [x] Drive `BoostTrack` from a standalone runner instead of adding an adaptor
      inside upstream `main.py`. Detections come from the L1 store, so the
      YOLOX-coupled detector, dataloader, and letterbox path are not used at all
      and upstream `main.py`, `dataset.py`, and `detector.py` are untouched.
- [x] Ensure the letterbox rescale in `BoostTrack.update` resolves to a factor
      of 1.0 for original-pixel detections, or every box is silently
      mis-scaled.
- [x] Load OSNet non-strictly, asserting the missing key set is exactly
      `{classifier.weight, classifier.bias}`, rather than fabricating zero
      tensors into the checkpoint.
- [x] Route the tracker to read precomputed embeddings instead of computing
      them inline.
- [x] Keep tracker-side edits minimal and recorded; `repos/` is git-ignored, so
      every change must be documented in `docs/tracker-experiments.md`.

## Phase 8 — Stage 1 runs and evaluation (`val_half`, held out)

- [x] Run BoostTrack++ with the RF-DETR detection variant.
- [x] Run the identical tracker configuration with the `yoloxx20` detection
      variant as a controlled baseline; only the detector differs.
- [x] Evaluate both with vendored TrackEval and record HOTA, MOTA, IDF1, IDSW.
- [x] Write the Stage 1 receipt and update `docs/tracker-experiments.md` with
      the metrics table, artifact paths, and exact commands.

## Phase 8b — Root cause of the HOTA deficit

Findings and evidence tables are in `docs/tracker-experiments.md`; this is the
task record only.

- [x] Confirm the baseline is measured here rather than quoted from upstream.
- [x] Rule out the export-floor confound with a matched-floor derived variant.
- [x] Rule out `det_thresh` mistuning with a 0.4/0.5/0.6 sweep.
- [x] Instrument every tracker stage per frame for all 4,463 frames, with hooks
      that record and delegate so behaviour is unchanged.
- [x] Compare the detector's training labels against MOT20 `gt.txt` directly.
      They are the same boxes: mean IoU 0.9995, 100% mutual coverage. This rules
      out a label-convention explanation for the mAP/HOTA disagreement.
- [x] Measure matched-IoU distributions and per-edge residuals for both
      detectors, before and after the tracker. `LocA` 85.13 against 88.18 is the
      dominant cause; the error is symmetric jitter, not a correctable offset.
- [x] Sweep NMS IoU. Best is 0.7, worth +0.63 HOTA and −24% IDSW. `LocA` does
      not move across the sweep, so post-processing cannot close the gap.
- [x] Probe whether higher inference resolution recovers localization. It does
      not; the model only regresses boxes well at its trained geometry.
- [x] Score every detection variant under one pycocotools protocol, so the
      baseline's mAP is directly comparable to the detector's own 0.6209. The
      baseline wins: $mAP_{50:95}$ 0.6759 against 0.6369, and $mAP_{small}$
      0.3074 against 0.1651. RF-DETR leads only at IoU 0.5. The premise that the
      new detector is more accurate does not survive an equal-terms comparison.

## Phase 9 — Stage 2 (MOT20 `test`, gated on Stage 1)

- [!] Blocked on a decision, not on missing work. The best RF-DETR configuration
      reaches HOTA 68.81 against the baseline's 70.21, and the residual gap is a
      detector box-regression deficit that needs a retrain at higher effective
      resolution rather than any tracker-side change. Whether to spend that
      training run before producing test tracks is a scope call.
- [ ] Fix `main.py`'s discarded `args.result_folder.replace("-val", "-test")`
      before any test run, or test tracks overwrite validation results.
- [ ] Produce tracks for MOT20-04, -06, -07, -08. No local ground truth exists,
      so this stage produces tracks, not scores.
- [ ] Label all outputs `local_test_adapted` per `docs/MOTPolicy.md`. The
      detector's training mix contains 21 Byte65 MOT20-test images and
      `mot20_sbs_S50.pth` is MOT20-trained, so results are not held out and not
      leaderboard-comparable.

## Unattended overnight run, 2026-09-07

A detached supervisor, `finetuning/scripts/overnight_arm_supervisor.sh`, is
running inside the `nvpt-dm` container with PPID 1, so it survives the VS Code
session that launched it. It waits for arm B, then runs arm D.

Read `finetuning/artifacts/overnight-status.md` first: it records each stage with
timestamps, and `finetuning/artifacts/overnight-supervisor.log` has the detail.

- [x] Arm B launched (30 epochs, `rfdetr-2xl-i5-armb-2026-09-07-r1`).
- [x] Arm B finished 21:54, 34,541s wall, 30/30 epochs. Peak 0.6168 at epoch 1.
- [x] Arm D started automatically at 22:00, micro-batch 4, 14 GB of 24 GB per
      card, no OOM, and the console confirms `long-side cap in force: 1600px`.
- [ ] **Arm D will not be finished in the early morning.** Halving the
      micro-batch doubles the micro-batches per epoch, and each costs 1.44× the
      pixels, so 30 epochs projects to finish late morning at the earliest.
      This does not matter for the result: arms A, B, and C peak at epochs 5, 1,
      and 9 and decline monotonically thereafter, so arm D's peak should be
      captured within the first few evals, around 01:00–02:00. Read
      `metrics.csv` for the peak rather than waiting for the run to end.
- [ ] Arm D starts automatically when B exits. `max_size = 1600` on the arm-A
      mix, run dir `rfdetr-2xl-i5-armd-2026-09-07-r1`. If micro-batch 8 hits CUDA
      OOM within the first hour, the supervisor retries once at micro-batch 4 with
      `grad_accum_steps = 2` — effective batch stays 64, so the arm remains
      comparable to arm A — into a separate run dir `-r2`. The failed attempt is
      preserved, never overwritten.
- [ ] No GPU work can be interleaved: arms B and D each occupy all 8 GPUs, and
      arm D is not expected to finish before roughly 08:00. Detector evaluation
      for arms B and D therefore has to happen after the user returns.

### Runbook: evaluating a finished arm

Not scripted on purpose — each step writes a separate inspectable artifact, and a
chained script has never been run end to end. Run from the repo root, in the
container. Substitute the arm's run dir, its peak `val/mAP_50_95` from
`docs/results-reference.md`, and a variant slug such as `rfdetr2xl-armb-e1`.

1. `export_detections.py` under `.venv` at `--threshold 0.05`, passing
   `--expected-map <peak>`. The reported mAP is a checksum: a mismatch means the
   export did not reproduce training geometry and no detections are written.
2. `derive_filtered_variant.py` under `.venv-tracking` with `--min-score 0.10
   --nms-iou 0.70`. Do not export pre-filtered: BoostTrack boosts low-confidence
   detections that match existing tracks *before* applying its own cut.
3. `export_embeddings.py`, then `run_boosttrack.py --tracker btpp-default`, then
   `evaluate_tracks.py`, all under `.venv-tracking`.
4. `build_results_reference.py` to refresh `docs/results-reference.md`.

Arm D must be exported with its own config so `model.max_size = 1600` is applied;
see the note below.

### Fixed while setting this up

`export_detections.py` read `model.resolution` but ignored `model.max_size`, and
hardcoded `max_size=1333` into the geometry it recorded. Exporting arm D with it
would have run the 1600px-trained checkpoint at 1333px and written provenance
saying 1333 as though that were intended. The long-side cap now lives in
`mot20.detection.rfdetr_training.apply_long_side_cap`, both the trainer and the
exporter call it, and the exporter records the cap actually in force. Covered by
`LongSideCapTest` in `finetuning/tests/detection/test_rfdetr_training.py`.

## Open questions

- Whether the Stage 2 run should use `mot20_sbs_S50.pth` or reuse the generic
  OSNet model. The former is BoostTrack++ as published; the latter keeps the
  detector comparison controlled across both stages.
- Whether tracker hyperparameters should be swept, and if so which, once the
  detector swap is measured.
- Whether to retrain the detector at a higher effective resolution. The 1333px
  long-side cap, not the 1120px target, is what binds at 1920×1080, and it is
  the measured cause of the localization deficit. This is the only change
  identified that can close the remaining HOTA gap, and it is a full training
  run, so it needs an explicit decision.
