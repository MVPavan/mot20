# Tracker Experiments

This is the index for BoostTrack++ tracking experiments that use this
repository's own RF-DETR detector in place of the published YOLOX detector.
Detector fine-tuning experiments are indexed separately in
`finetuning/experiments.md`.

**Scope of this document.** It records the integration target, the acquired
external assets, the verified contracts the swap must satisfy, and the runbook
for evaluating a finished detector arm. It is authoritative for weight
checksums, the detection-handoff contract, and artifact/variant naming.

It is **not** where results live. Tracking runs have been executed — the
association sweep, arms A and D, and the `yoloxx20` baseline all have TrackEval
numbers. Every measured figure is in
[`docs/results-reference.md`](results-reference.md) (generated, authoritative for
values), interpreted in [`docs/experiment-report.md`](experiment-report.md).
Task status is in Beads.

## Integration Target

| Item | Value |
| --- | --- |
| Tracker | BoostTrack++ (Stanojevic and Todorovic), `repos/BoostTrack` at upstream commit `fb5bfc3` |
| Published detector | YOLOX-X ByteTrack checkpoints, selected in `default_settings.get_detector_path_and_im_size` |
| Replacement detector | RF-DETR 2XL, one pedestrian class, epoch 5 of the fifty-epoch Byte65 run |
| Detector checkpoint | `finetuning/artifacts/rfdetr-2xl-byte65-test-adapted-ddp-batch8-lr5e5-aspect-full-50e-2026-09-04-r1/checkpoint_best_total.pth` |
| Detector selection evidence | `best_total_source = regular`, `global_step = 2238`; peak $mAP_{50:95}=0.6202$ on MOT20 `val_half` |
| Detector receipt | `finetuning/docs/experiments/2026-09-04-rfdetr-2xl-byte65-aspect-full-50e.md` |
| ReID | Unchanged from BoostTrack++; see the external asset table below |

`repos/` is git-ignored, so neither the tracker checkout nor its weights are
under version control in this repository. The checksums recorded below are the
only durable identification of those files.

## External Assets

Both files were acquired on 2026-09-05 and placed in
`repos/BoostTrack/external/weights/`, which the upstream checkout does not
provide.

| File | Bytes | sha256 | Source | Verification |
| --- | ---: | --- | --- | --- |
| `osnet_ain_ms_d_c_wtsonly.pth` | 8,957,365 | `482f09440f28ac92b7aa215a56323430f03aaf2657273269af02a670d3c8440a` | `levan92/deep_sort_realtime` GitHub raw | Loaded into `osnet_ain_x1_0`; 512-d output, finite, zero unexpected keys |
| `mot20_sbs_S50.pth` | 315,810,563 | `d3a39a1ab54a63ac6724f2864a36485b0b5762fefced98941fca2a0784180656` | BoT-SORT authors' published MOT20-SBS-S50 link | ResNeSt50 deep stem and Split-Attention blocks, SBS GeM/BNNeck head, `heads.weight` is `(414, 2048)`, epoch 8 |

Notes on these assets:

- `osnet_ain_ms_d_c` is torchreid OSNet-AIN x1.0, domain-generalized over
  MSMT17, DukeMTMC, and CUHK03. BoostTrack uses it for validation splits
  because the MOT-specific FastReID models were trained on the evaluated val
  half (`tracker/embedding.py`, `_get_general_model`). Using it keeps the ReID
  side of a `val_half` measurement held out, matching the detector.
- The acquired OSNet file is the weights-only variant. Its inner `state_dict`
  is `module.`-prefixed exactly as BoostTrack expects, but the two
  `classifier.*` entries are absent. OSNet returns features before the
  classifier in eval mode, so the classifier is unused; a strict
  `load_state_dict` will nonetheless reject the file. The intended fix is a
  non-strict load asserting the missing set is exactly
  `{classifier.weight, classifier.bias}`, rather than fabricating zero tensors
  into the checkpoint.
- `mot20_sbs_S50.pth` is required only for a `--test_dataset` run. Its 414
  identity classes are the MOT20 ReID identity count and are the primary
  evidence that the file is MOT20-trained rather than the MOT17 sibling.
- A Hugging Face file of the same name exists at 336,553,343 bytes. It is not
  byte-identical to the authors' 315,810,563-byte file and was not used.
- Structural verification is what is recorded above, performed before the
  tracking environment existed. Both checkpoints have since been instantiated
  through their real frameworks (`fast_reid.build_model`, torchreid) under
  `.venv-tracking` by `tracking/scripts/export_embeddings.py`, which is what
  produced the L2 embeddings in `artifacts/tracking/embeddings/`.

## Verified Integration Contracts

These were read from the tracker source and must hold for any result from this
swap to be meaningful.

| Contract | Detail |
| --- | --- |
| Detection array | `Nx5` float `[x1, y1, x2, y2, score]`; xyxy, no class column; `update()` must be called every frame, with `np.empty((0, 5))` when empty |
| Coordinate space | `BoostTrack.update` divides boxes by the YOLOX letterbox scale factor before use. Supplying original-pixel boxes requires that factor to be 1.0, or every box is silently mis-scaled |
| Confidence ordering | `dlo_confidence_boost` and `duo_confidence_boost` run on the raw detection set *before* the `det_thresh` cut (0.4 for MOT20). Exporting only above-threshold detections removes the input those mechanisms operate on and silently degrades BoostTrack++ toward BoostTrack |
| Export threshold | Detections must therefore be exported at a low threshold; the published YOLOX path supplies everything above 0.1 |
| Detector coupling | Detector output is boxes and scores only. ReID crops and camera-motion compensation are computed from the raw image and are independent of the detector |
| No file ingestion | The upstream tracker has no path for externally supplied detections. `--public` is declared in `args.py` and never read. A file-backed adaptor satisfying `forward(batch, tag)` is required |
| Result format | `{frame},{id},{x1},{y1},{w},{h},{conf},-1,-1,-1`, tlwh in original pixels, written per sequence under `results/trackers/` |

Three upstream defects to account for:

- `main.py` discards the return value of `args.result_folder.replace("-val", "-test")`,
  so `--test_dataset` writes test results into the `MOT20-val` directory and can
  overwrite validation results. This must be fixed before any test-split run.
- `external/adaptors/fastreid_adaptor.py` hardcodes the MOT17 SBS config even
  when loading MOT20 weights. The two configs differ only in dataset names and
  output directory, so this is harmless for inference.
- **fp16 silently corrupts FastReID embeddings on this environment.** Measured
  and fixed; see the I3 section of `docs/tracker-improvements.md` for the
  numbers. This one is dangerous because it does not raise.

### Local changes to `repos/BoostTrack`

`repos/` is git-ignored, so every edit is recorded here. All are annotated
`mot20 local change` in the source.

| File | Change | Why |
| --- | --- | --- |
| `tracker/embedding.py` | Load the OSNet checkpoint non-strictly, asserting the missing key set is exactly `{classifier.weight, classifier.bias}` | The obtainable checkpoint is the weights-only variant with the classifier head stripped. OSNet returns features before the classifier in eval mode, so the head is unused; a genuinely wrong checkpoint still fails loudly |
| `tracker/embedding.py` | Removed `model.half()` from the FastReID branch of `initialize_model` | fp16 returns embeddings uncorrelated with fp32 on this torch build |
| `external/adaptors/fastreid_adaptor.py` | `forward` follows the module's own dtype instead of forcing `batch.half()` | Same defect; keeps the adaptor correct whichever dtype the caller selects |

Upstream `main.py`, `dataset.py`, and `detector.py` are untouched: the standalone
runner drives `BoostTrack` directly, so the YOLOX-coupled detector, dataloader,
and letterbox path are never used.

## Artifact Levels and Variant Naming

Every intermediate is a saved, addressable artifact so that detector, ReID, and
tracker choices can be combined without recomputation and without ambiguity
about which combination produced a result. Root is `artifacts/tracking/`, which
is git-ignored; the manifests and this document are the durable record.

| Level | Artifact | Key | Reused by |
| --- | --- | --- | --- |
| L0 | ECC camera-motion transforms | sequence | every permutation; image-derived only |
| L1 | Detections | detector variant, split | all ReID and tracker choices |
| L2 | ReID embeddings | detection variant, ReID variant, split | all tracker choices |
| L3 | Tracks | detection, ReID, tracker config, split | — |
| L4 | Metrics | same triple as L3 | — |

```
artifacts/tracking/
  detections/<det>/<split>/<sequence>/det.txt        + manifest.json
  embeddings/<det>__<reid>/<split>/<sequence>.pkl    + manifest.json
  tracks/<det>__<reid>__<tracker>/<split>/<sequence>.txt
  metrics/<det>__<reid>__<tracker>/<split>/
```

Each `manifest.json` records the upstream variant identifiers and checksums, so
any cell of the grid is reproducible from its path plus its manifests.

L2 embeddings are precomputed for **every exported detection**, not only those
surviving a tracker's `det_thresh`. This makes them independent of tracker
configuration at the cost of additional inference, which is the property that
makes the grid composable.

The published YOLOX detections already present under
`datasets/val_half/<sequence>/det_yoloxx20/` are registered as an ordinary
detector variant rather than a special case, so the published detector is one
cell of the same grid.

### Cache-keying hazard

Upstream keys its embedding cache by sequence name alone
(`./cache/embeddings/{video}_embedding.pkl`, loaded via `tag.split(":")[0]`).
Its only guard compares the cached embedding count against the detection count.
That guard cannot detect a changed ReID model over unchanged detections: the
counts match and stale vectors from the previous model are returned silently.
Correct keying by detection variant and ReID variant is a precondition for any
combination study, not an optimization. The ECC cache is exempt because its
contents depend only on the images.

## Planned Experiments

### Stage 1: `val_half`, held out

| Item | Value |
| --- | --- |
| Status | Completed 2026-09-05 |
| Split | MOT20-01, -02, -03, -05 second halves. BoostTrack's vendored `results/gt/MOT20-val/` is byte-equivalent in span to `datasets/val_half/` (MOT20-01 is `seqLength=214`, frames renumbered from 1) |
| Classification | Held out for both detector and ReID **on the RF-DETR side**. The detector trained on `train_half`; the ReID model is the generic OSNet, not the val-trained SBS model. The `yoloxx20` baseline is **not** held out: `bytetrack_x_mot20.tar` trained on the full MOT20 train set and has seen these frames, as `datasets/README.md` records. MOT17 is out of scope, so no clean external baseline is available and the caveat is permanent. See `docs/experiment-report.md` §0 |
| Variant | BoostTrack++ with ReID, upstream default settings |
| Evaluation | Vendored TrackEval, `run_mot_challenge.py --BENCHMARK MOT20 --SPLIT_TO_EVAL val` |
| Baseline | The same runner over the existing `datasets/val_half/<seq>/det_yoloxx20/det_yoloxx20.txt` detections, holding tracker and ReID fixed and varying only the detector |

#### Detector geometry gate

Before exporting detections, the epoch-5 checkpoint was re-evaluated through
RF-DETR's own `evaluate()` path. It returned $mAP_{50:95}=0.6209$ against the
0.6202 recorded during training, and `val/loss` 6.0158 against 6.0166. The
export then captured detections from that same verified loop, so the reported
mAP is a checksum on the export rather than a separate claim. Receipt:
`artifacts/tracking/geometry-gate-rfdetr2xl-e5.json`.

This gate was worth running: `RFDETR.predict()` resizes to a square, whereas
validation resizes the short side toward 1120 with an aspect-preserving 1333px
cap (1333x750 for MOT20's 1920x1080). Exporting through `predict()` would have
silently used the wrong geometry.

#### Results

All rows share ReID `osnet-ain-msdc` and BoostTrack++ default settings; only the
detector and, where noted, the post-boost `det_thresh` differ. Evaluated with
the vendored TrackEval against `results/gt/MOT20-val`.

| Detector | `det_thresh` | HOTA | MOTA | IDF1 | AssA | DetA | LocA | IDSW | IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yoloxx20` (baseline) | 0.4 | **70.21** | 85.03 | 82.54 | **67.13** | **73.50** | **88.18** | **1013** | 1762 |
| `rfdetr2xl-e5-t005` | 0.4 | 68.08 | 87.01 | 81.81 | 64.53 | 72.00 | 85.13 | 2113 | 2662 |
| `rfdetr2xl-e5-t005` | 0.5 | 68.35 | 86.46 | 82.79 | 65.31 | 71.68 | 85.32 | 1615 | 2208 |
| `rfdetr2xl-e5-t005` | 0.6 | 68.02 | 85.18 | 83.12 | 65.64 | 70.62 | 85.46 | 1323 | 1970 |
| `rfdetr2xl-e5-t010` | 0.4 | 68.18 | **87.01** | 81.87 | 64.64 | 72.07 | 85.14 | 2065 | 2624 |
| `rfdetr2xl-e5-t010-nms070` | 0.4 | 68.81 | 86.88 | 83.26 | 65.97 | 71.93 | 85.19 | 1571 | 2309 |
| `rfdetr2xl-e5-t010-nms0600` | 0.4 | 68.51 | 85.75 | **83.39** | 66.27 | 70.96 | 85.21 | 1427 | 2162 |
| `rfdetr2xl-e5-t010-nms0500` | 0.4 | 66.08 | 82.65 | 81.50 | 63.88 | 68.48 | 85.27 | 1297 | 2037 |

Ground truth has 1,418 identities and 615,137 boxes.

`-t010` is `-t005` filtered to score $\ge 0.1$, matching the floor the published
YOLOX detections were written at. `-nmsNNN` additionally applies greedy NMS at
that IoU. Both are derived from the exported rows by selection, so no inference
or ReID forward pass is repeated and the embedding rows stay aligned; see
`tracking/scripts/derive_filtered_variant.py`.

#### Interpretation

**The RF-DETR detector finds more people and draws looser boxes.** It raises
MOTA by 1.98 points at the stock threshold while HOTA falls 2.13. MOTA is scored
at a single IoU threshold and rewards recall; HOTA averages over localization
thresholds from 0.05 to 0.95. A detector that wins on recall and loses on box
tightness moves the two metrics in opposite directions, which is exactly what
happened. Measured under one protocol the same split shows in detection space:
$mAP_{50}$ 0.9585 against 0.9000, $mAP_{50:95}$ 0.6369 against 0.6759.

The baseline number is measured here, not quoted: `det_yoloxx20.txt` was run
through the same runner, ReID weights, tracker configuration, and evaluator as
every RF-DETR row. Only the detection file differs.

#### Root-cause analysis

##### The detector is not more accurate overall

Scored by `tracking/scripts/analyze_detections.py`, which runs every variant
through one pycocotools protocol against the same `valid` annotations, with
MOT20's ignore regions carried as `iscrowd`:

| | `yoloxx20` | `-t005` | `-t010` | `-t010-nms070` |
| --- | ---: | ---: | ---: | ---: |
| $mAP_{50}$ | 0.9000 | **0.9585** | **0.9585** | 0.9510 |
| $mAP_{75}$ | **0.8278** | 0.7595 | 0.7549 | 0.7483 |
| $mAP_{50:95}$ | **0.6759** | 0.6369 | 0.6337 | 0.6297 |
| $AR_{50:95}$ | **0.7104** | 0.6957 | 0.6883 | 0.6811 |
| $mAP_{small}$ | **0.3074** | 0.1651 | 0.1628 | 0.1626 |
| $mAP_{medium}$ | **0.6536** | 0.5983 | 0.5948 | 0.5895 |
| $mAP_{large}$ | **0.7216** | 0.7058 | 0.7043 | 0.6995 |
| Boxes per frame, mean | 130.0 | 283.5 | 212.8 | 194.0 |
| Boxes per frame, max | 213 | 386 | 355 | 331 |
| Duplicate pairs per frame at IoU $\ge$ 0.75 | 0 | 33.2 | 13.8 | 0 |

**The baseline detector has the higher $mAP_{50:95}$, 0.6759 against 0.6369.**
RF-DETR leads only at IoU 0.5, by 5.9 points, and trails at every stricter
threshold and in every size band. The detector's reported 0.6202 was never a
claim of superiority over this baseline; it had simply never been compared
against it on equal terms until now. The tracking result is therefore not a
paradox at all, and the earlier framing of "better detection, worse tracking"
was wrong: HOTA is measuring the same thing COCO AP measures once the loose-IoU
advantage is set aside.

The $mAP_{small}$ column shows the largest ratio, 0.1651 against 0.3074, but it
is not what drives the overall gap: only 1.2% of ground-truth instances fall in
COCO's small band. The distribution is 1.2% small, 61.5% medium, 37.3% large, so
the loss is carried by the medium band, 0.5948 against 0.6536. The small-object
collapse is corroborating evidence for the resolution finding below rather than
the mechanism itself.

##### Why the published COCO advantage does not transfer

RF-DETR's reported COCO margin over YOLOX was established under conditions that
none of this workstream reproduces, so it should not have been expected to carry
over:

| Condition | Published COCO setting | Here |
| --- | --- | --- |
| Baseline | Generic 80-class YOLOX-X trained on COCO at 640×640 | YOLOX-X trained by the ByteTrack authors on MOT17, MOT20, and CrowdHuman, single class, at 1600×896 |
| Instances per image | ~7 | **137.8** mean, 220 max |
| Input scale | Square 312–768 on ~640×480 images, at or above native | 1333×750 on 1920×1080, a 0.694 downscale |
| Training maturity | Long, fully scheduled runs | Best checkpoint is epoch 5 of 50, then decline |

The density row is the most consequential, because it disables the mechanism the
architecture wins by. Set prediction's advantage is duplicate-free output from
one-to-one Hungarian matching, removing the need for NMS. This export produced
33.2 duplicate pairs per frame at IoU $\ge$ 0.75, in every one of the 4,463
frames, with a `cardinality_error` of 7.84; and reintroducing classical NMS
improved tracking. One-to-one matching did not converge at this density, so the
architecture's cost was paid without collecting its benefit.

Note also that NMS slightly *lowers* AP (0.6369 → 0.6297) while *raising* HOTA
(68.18 → 68.81). COCO AP tolerates extra low-ranked duplicates that a tracker
must resolve into identities. Optimising the detector for AP alone would not
have found this setting.

##### Localization

Localization is the dominant cause and it is measurable before the tracker runs.
Raw detections at `det_thresh = 0.4`, Hungarian-matched to `gt.txt` at IoU 0.5
by `tracking/scripts/analyze_localization.py`:

| | `rfdetr2xl-e5-t005` | `yoloxx20` |
| --- | ---: | ---: |
| Boxes above 0.4 | 638,477 | 553,800 |
| Recall at IoU 0.5 | **0.9200** | 0.8876 |
| Precision at IoU 0.5 | 0.8864 | **0.9859** |
| Mean matched IoU | 0.8334 | **0.8713** |
| Matches with IoU $\ge$ 0.75 | 0.852 | **0.942** |
| Matches with IoU $\ge$ 0.90 | 0.214 | **0.393** |

RF-DETR wins recall by 3.2 points and retains half as many matches at IoU 0.90.
TrackEval reports the same fact as `LocA`, 85.13 against 88.18, and that deficit
is what depresses both HOTA factors.

The error is symmetric jitter, not a correctable offset. Per-edge residuals
normalised by ground-truth box size have means of −0.005, −0.005, +0.009, +0.001
for $x_1, y_1, x_2, y_2$, and median size ratios of 1.009 wide and 1.003 tall.
The absolute residuals are 27–46% larger than the baseline's on every edge.

##### Hypotheses tested

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| Export floor confound, 0.05 against the baseline's 0.1 | Ruled out | Matched-floor `-t010`: HOTA 68.18 against 68.08 |
| `det_thresh` mistuned for a DETR score distribution | Ruled out | 0.4/0.5/0.6 sweep peaks at 68.35, still 1.9 short |
| Training labels differ from MOT20 `gt.txt` | Ruled out | 615,137 boxes both sides, 100% mutual coverage, mean IoU 0.9995, zero edge bias |
| Systematic box-convention offset in model output | Ruled out | Residual means $\approx 0$; error is symmetric |
| A post-detection stage is broken or YOLOX-specific | Ruled out | Localization is unchanged across the tracker: 0.8334 → 0.8343 and 0.8713 → 0.8726 |
| Duplicate boxes from NMS-free set prediction | **Confirmed, partial** | NMS at 0.7 gives +0.63 HOTA and −24% IDSW |
| Detector box regression is genuinely looser | **Confirmed, dominant** | `LocA` 85.13 against 88.18 |
| Recoverable by raising inference resolution | Ruled out | $mAP_{75}$ falls 0.740 → 0.594 at cap 1600 and 0.689 at 1920 |

The label comparison is decisive and worth stating plainly: the COCO annotations
the detector was trained and scored against are the same boxes as the `gt.txt`
that TrackEval scores against. Good mAP and poor HOTA are not an artifact of two
disagreeing label sets.

NMS is a real but bounded fix. Sweeping the suppression IoU moves HOTA 68.18 →
68.81 → 68.51 → 66.08 at none/0.7/0.6/0.5; 0.7 matches the `nmsthre` the
published YOLOX detections were produced with, and below 0.6 suppression starts
deleting genuinely overlapping people in MOT20 crowds. Across that entire sweep
`LocA` moves only from 85.14 to 85.27, so no post-processing reaches the
remaining deficit.

##### Effective inference resolution

The validation transform is `RandomResize([1120], max_size=1333)`. At MOT20's
1920×1080 the 1333 cap binds rather than the 1120 target, so the model sees
1333×750, a scale of 0.694. The baseline YOLOX-X runs at `test_size = (896,
1600)`, a scale of 0.830, and therefore sees every pedestrian about 20% larger.
The measured edge-residual ratio is 1.27–1.46×, so resolution accounts for much
of the gap, with DETR's L1 and GIoU box regression against YOLOX's IoU loss a
plausible remainder.

This is not recoverable at inference time. Raising the cap leaves recall flat
(0.9134 → 0.9138 → 0.9167) while box regression degrades sharply and
non-monotonically, which is the signature of train/test geometry mismatch in the
windowed attention and positional embeddings. Closing it requires retraining at
a higher effective resolution.

##### Stage instrumentation

`tracking/scripts/diagnose_stages.py` records per-frame counts at each pipeline
stage for all 4,463 frames, hooking `dlo_confidence_boost`,
`duo_confidence_boost`, and `associate` so every hook records and delegates.
Per-frame means, RF-DETR against baseline: raw 283.5/130.0, promoted by boosting
5.13/1.58, entering association 148.2/125.7, live tracks 199.6/152.3, new tracks
0.864/0.469, unmatched tracks 52.3/27.1, output boxes 140.5/120.6. Against
615,137 ground-truth boxes the tracker emits 627,099 with RF-DETR and 538,112
with the baseline, so the extra output is close to correct in volume.

##### Artifacts

| Artifact | Contents |
| --- | --- |
| `artifacts/tracking/localization-analysis-val_half.json` | Matched IoU distributions and per-edge residuals, per sequence and combined, for detections and tracks |
| `artifacts/tracking/annotation-agreement-val_half.json` | COCO training labels against `gt.txt` |
| `artifacts/tracking/diagnosis/stages-*.{csv,json}` | Per-frame stage counts and per-sequence aggregates |
| `artifacts/tracking/geometry-probe-rfdetr2xl-e5-max{1600,1920}.json` | Resolution probes |
| `artifacts/tracking/detection-analysis-val_half.json` | Per-variant COCO scores, boxes per frame, duplicate rates |

### Stage 2: MOT20 `test`

| Item | Value |
| --- | --- |
| Status | Planned; not run, and gated on Stage 1 |
| Split | MOT20-04, -06, -07, -08 |
| Classification | `local_test_adapted`. The detector's training mix contains 21 manually audited Byte65 MOT20-test images, and `mot20_sbs_S50.pth` is MOT20-trained. Results are local deployment-development evidence and must not be presented as held-out or leaderboard-comparable, per `docs/MOTPolicy.md` |
| Metrics | None available locally; MOT20 test ground truth is not in this repository. This stage produces tracks, not scores |

## Runbook: evaluating a finished detector arm

Moved here from `docs/tracker-todo.md` when task tracking migrated to Beads.
Task *status* lives in Beads; this is a procedure, so it stays in the docs.

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

Each arm must be exported with **its own config**, so that `model.max_size` is
applied. `apply_long_side_cap` is shared by the trainer and the exporter for
this reason: a checkpoint trained at a raised cap and exported at the library
default would run at geometry the model never saw, and would record provenance
claiming the default as though it were intended. Covered by `LongSideCapTest` in
`finetuning/tests/detection/test_rfdetr_training.py`.

## Integration Questions, and How They Resolved

These were the open questions when the swap was designed. All four are settled;
they are kept because each resolution is a contract the pipeline still depends
on. Remaining work is tracked in Beads, not here.

- **Tracking environment.** There is no conda on this machine, so the upstream
  `boost-track-env.yml` could not be built as written. Resolved with a separate
  `.venv-tracking` virtual environment, leaving the detector training
  environment untouched. Both ReID checkpoints have since been instantiated
  through their real frameworks by `export_embeddings.py`, so the structural
  verification recorded above is no longer the only evidence.
- **RF-DETR inference geometry.** Training used aspect-preserving 1120px resize
  with padding while `RFDETR.predict` defaults to a square resize, which would
  have silently degraded a full detection dump. Resolved by sharing
  `apply_long_side_cap` between trainer and exporter and gating every export
  with `verify_detector_geometry.py`; the geometry probes are in
  `artifacts/tracking/geometry-*.json`.
- **NMS on RF-DETR set predictions.** Greedy suppression at IoU 0.7, matching
  the `nmsthre` the published YOLOX detections were produced with, is the best
  setting measured and is worth +0.63 HOTA. `num_select = 390` saturation was
  observed (MOT20-05 reached 386 in one frame before filtering, 331 after) but
  is not the limiting factor; localization is.
- **Detection export format and location.** Resolved as
  `<sequence>/det_<name>/det_<name>.txt`, following the convention already used
  by `datasets/val_half/` and `datasets/MOT20_TEST_DET/`.
