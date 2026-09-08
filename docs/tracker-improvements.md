# Detector and Tracker Improvement Plan

Progress tracker for closing the gap between this repository's fine-tuned
RF-DETR detector and the published ByteTrack YOLOX-X detector inside
BoostTrack++.

**Task status lives in Beads, not here.** Run `bd ready`, `bd list
--status=in_progress`, or `bd list --status=open`; this file keeps the analysis
and the measured evidence behind each item. `docs/tracker-todo.md` was deleted
when tracking moved to Beads.

Contracts and artifact paths live in `docs/tracker-experiments.md`; every metric
is generated into `docs/results-reference.md`; the cross-detector comparison and
its caveats are in `docs/experiment-report.md`.

## Where we are

**The `yoloxx20` baseline is not held out on `val_half`.** `bytetrack_x_mot20.tar`
trained on the full MOT20 train set, of which `val_half` is the second half, so
every gap in this file is measured against an advantaged opponent and is a lower
bound on RF-DETR. `datasets/README.md` already recorded this and the tracker docs
failed to carry it forward. Upstream `default_settings.py` substitutes the MOT17
model for MOT20 validation for exactly this reason, but **MOT17 is out of scope
(decided 2026-09-08)**, so no clean external baseline is available and the caveat
is permanent. Full evidence in `docs/experiment-report.md` §0.

### Current best, after Tier 0

Each detector with its own best ReID and association settings, on `val_half`
with the vendored TrackEval. Absolute values are inflated and **not held out**,
because `fastreid-sbs-s50-mot20` was trained on MOT20 and has seen these
identities; the delta stays meaningful because both sides use it.

| | `yoloxx20` | RF-DETR | Delta |
| --- | ---: | ---: | ---: |
| HOTA | **70.734** | 69.718 | **−1.016** |
| Detector variant | `yoloxx20` | `rfdetr2xl-e5-t010-nms070` | |
| ReID | `fastreid-sbs-s50-mot20` | `fastreid-sbs-s50-mot20` | |
| Association | `lambda_iou=0.35` | `iou_threshold=0.25, lambda_iou=0.35, lambda_shape=0.35` | |

Tier 0 moved RF-DETR from 68.08 to 69.72, but the baseline moved with it, so the
gap only closed from 2.13 to 1.02. Everything remaining is detector box
regression.

### Starting point, for reference

Measured on `val_half`, ReID `osnet-ain-msdc`, BoostTrack++ defaults, vendored
TrackEval. Both detectors run through the same runner and evaluator.

| | Baseline `yoloxx20` | Best RF-DETR `-t010-nms070` | Delta |
| --- | ---: | ---: | ---: |
| HOTA | **70.21** | 68.81 | −1.40 |
| DetA | **73.50** | 71.93 | −1.57 |
| AssA | **67.13** | 65.97 | −1.16 |
| LocA | **88.18** | 85.19 | −2.99 |
| MOTA | 85.03 | **86.88** | +1.85 |
| IDF1 | 82.54 | **83.26** | +0.72 |
| IDSW | **1013** | 1571 | +558 |
| $mAP_{50}$ | 0.9000 | **0.9510** | +0.051 |
| $mAP_{50:95}$ | **0.6759** | 0.6297 | −0.046 |

The deficit is detector box regression, not tracker integration. Everything
reachable from the tracker side has been measured; see Phase 8b.

## Evaluation integrity — read before starting I4

**I4 removes the only held-out evaluation this repository has.** Once `val_half`
is in the training mix, the table at the top of this file cannot be re-measured,
and no other Tier 1 change has a local yardstick.

**Decided 2026-09-06 (user):** maintain two dataset builds and two checkpoints,
which is what ByteTrack itself does — half-split ablations, full-train final
model. The competition build trains for an epoch count chosen on the ablation
build and takes the **final** checkpoint; its `valid` split is a formality for
the training loop and must not drive checkpoint selection. Given the epoch-5
peak already observed, the epoch count is a real experimental output of the
ablation build, not a default.

| Build | Training data | Purpose | Evaluation |
| --- | --- | --- | --- |
| Ablation | `mot20_train_half` + CrowdHuman + Byte65 | Compare I5, I6, I7 against each other and against the baseline | `val_half`, held out, valid |
| Competition | `mot20_train_half` + `mot20_val_half` + CrowdHuman + Byte65 | MOT20 `test` tracks only | None locally; `local_test_adapted` |

Do not evaluate the competition build on `val_half` and report the number
without this caveat attached. Both builds already carry `classification =
local_test_adapted` because the training mix contains 21 human-audited Byte65
MOT20-test images and is therefore not leaderboard-comparable regardless.

## Tier 0 — no retraining

### I1 — NMS at IoU 0.7 as the export default

- Sweep suppression IoU as derived variants: none / 0.7 / 0.6 / 0.5 gives
  HOTA 68.18 / 68.81 / 68.51 / 66.08. 0.7 matches the `nmsthre` the
  published YOLOX detections were produced with.
- Confirm the mechanism: RF-DETR emitted 33.2 duplicate pairs per frame at
  IoU $\ge$ 0.75 in every one of 4,463 frames; after NMS at 0.7, zero.

### I2 — Association hyperparameter sweep

Before this, only `det_thresh` had been swept. The rest were left at values
upstream tuned for a YOLOX score distribution and box-tightness profile, which
RF-DETR matches on neither count.

- Swept 19 points across two rounds, every one recorded in the L3/L4 store
  under its own tracker slug, with the human-readable overrides preserved in
  the L3 manifest.

Best per detector, both on `fastreid-sbs-s50-mot20`:

| Detector | Best setting | HOTA | Δ against its own defaults |
| --- | --- | ---: | ---: |
| `yoloxx20` | `lambda_iou=0.35` | **70.734** | +0.187 |
| `rfdetr2xl-e5-t010-nms070` | `iou_threshold=0.25, lambda_iou=0.35, lambda_shape=0.35` | **69.718** | +0.209 |

**The gap is 1.016, essentially unchanged from the untuned 1.04.** This is the
predicted outcome: association tuning cannot reach a localization deficit, and
`LocA` stays at 85.2 against 88.2 across every point swept.

##### The control matters

`lambda_iou=0.35` improves the baseline by +0.187 and RF-DETR by +0.186. It is
therefore **not** compensation for RF-DETR's looser boxes, despite being an
attractive story; it is simply a better MOT20 setting than upstream's 0.5 for
both detectors. Only the three-knob combination is detector-specific: it is
RF-DETR's best at +0.209 while costing the baseline −0.130. Any tuning claim
here must be checked against the other detector before it is believed.

##### Dead knobs found

- `dlo_boost_coef` has **no effect under BoostTrack++ defaults**. Overrides at
  0.3 and 0.7 produced results byte-identical to the default across all twelve
  metrics. The override applied correctly, the L3 manifest records
  `dlo_boost_coef_effective`, but `boost_track.py:337` only reads it inside
  `if not use_soft_boost and not use_varying_th:`, and BoostTrack++ sets both
  `use_sb` and `use_vt` to `True`. Under BoostTrack++ the boosting behaviour is
  governed by literals in the method body — `alpha = 0.65`,
  `threshold_s = 0.95`, `threshold_e = 0.8`, `n_steps = 20` — which are not
  exposed as settings at all.
- `GeneralSettings.values["max_age"]` is likewise never read: `boost_track.py:133`
  calls the static `GeneralSettings.max_age(video_name)` helper, which derives
  the value from frame rate and returns 50 for MOT20's 25 fps sequences, not the
  dictionary's 30. `run_boosttrack.py` rewires the helper so `--set max_age=...`
  actually takes effect.
- Raising `min_hits` was the intuitive fix for RF-DETR's 1.84× track-creation
  rate and it is a **dead end**: it does suppress spurious tracks, IDSW 1670 →
  1345 and IDs 2545 → 2315 at `min_hits=5`, but the DetA cost exceeds the AssA
  gain and HOTA falls 0.337.

### I3 — FastReID `mot20_sbs_S50.pth`

Acquired and verified in Phase 0, promoted to verified-working in Phase 1, and
never used in an experiment. This is the ReID model BoostTrack++ publishes with.

- Export L2 embeddings for the best RF-DETR variant and for `yoloxx20`, so
  the detector comparison stays controlled.
- Track and evaluate both.
- Label the result. `mot20_sbs_S50.pth` is MOT20-trained, so on `val_half`
  it has already seen these identities. Absolute numbers are not held out;
  the detector-to-detector delta remains meaningful because both sides use
  it.

Result. FastReID helps both detectors and helps RF-DETR roughly twice as much,
narrowing the gap from 1.40 to 1.04 HOTA:

| | RF-DETR `-t010-nms070` | | `yoloxx20` | |
| --- | ---: | ---: | ---: | ---: |
| ReID | `osnet-ain-msdc` | `fastreid-sbs-s50-mot20` | `osnet-ain-msdc` | `fastreid-sbs-s50-mot20` |
| HOTA | 68.809 | **69.509** | 70.208 | **70.547** |
| DetA | 71.928 | 71.970 | 73.496 | 73.610 |
| AssA | 65.965 | **67.269** | 67.133 | **67.677** |
| AssPr | 79.746 | **81.780** | 83.337 | **84.359** |
| LocA | 85.190 | 85.207 | 88.177 | 88.210 |
| IDF1 | 83.255 | **84.156** | 82.536 | **82.877** |
| IDSW | 1571 | 1670 | 1013 | 989 |

The asymmetry is consistent with the root cause. RF-DETR's IoU cue is weaker,
LocA 85.2 against 88.2, so it leans harder on appearance and gains more when
appearance improves. `LocA` is unchanged, as it must be: ReID cannot alter box
tightness.

#### Upstream defect found: FastReID in fp16 returns uncorrelated embeddings

`EmbeddingComputer.initialize_model` called `model.half()`, and
`FastReID.forward` unconditionally cast its input to half. On this environment's
torch build, SBS ResNeSt50 in fp16 returns **finite** embeddings of plausible
magnitude that are uncorrelated with the fp32 result, plus NaN on some crops:

| Sample | fp32 norm | fp16 finite | fp16 norm | cosine against fp32 |
| --- | ---: | ---: | ---: | ---: |
| MOT20-01 f1, 52 crops | 41.44 | 52/52 | 50.26 | **−0.0533** |
| MOT20-02 f50, 53 crops | 39.37 | 53/53 | 50.10 | **−0.0428** |
| MOT20-05 f200, 64 crops | 38.83 | 64/64 | 49.60 | **−0.0216** |

Most frames produce no NaN, so this fails silently: the tracker would associate
identities on noise and still emit plausible output. It surfaced only because
MOT20-01 overflowed to NaN and `export_embeddings.py` refuses non-finite values.
Both casts are now removed and annotated `mot20 local change`; see the Verified
Integration Contracts section. fp32 costs throughput only.

## Tier 1 — dataset and training

### I4 — Add MOT20 `val_half` to training (ByteTrack parity)

ByteTrack trains its YOLOX-X on the full MOT20 train set plus CrowdHuman. This
repository trains on `train_half` only, because `val_half` is held out. Adding
it back is ByteTrack parity and nearly doubles MOT20's share of the mix.

| | Current | With `val_half` |
| --- | ---: | ---: |
| CrowdHuman images | 19,370 (81.2%) | 19,370 (68.4%) |
| MOT20 images | 4,489 (18.8%) | 8,952 (31.6%) |
| Total | 23,859 | 28,322 |

Note this also partially addresses I5, since it lifts MOT20 from 18.8% to 31.6%
of training images. The two should not be run as one experiment without an
ablation, or neither effect can be attributed.

### I5 — Rebalance the CrowdHuman/MOT20 mix

The measured skew, and the leading hypothesis for why the best checkpoint is
epoch 5 of 50 and the remaining 45 epochs made the model worse.

| | CrowdHuman | MOT20 |
| --- | ---: | ---: |
| Training images | 19,370 (81%) | 4,489 (19%) |
| Person instances | 439,046 | 522,065 |
| Persons per image, mean | 22.7 | 116.3 |
| Median long side | 1024 | 1654 |
| Resize scale applied | 1.302 (upscaled) | 0.806 (downscaled) |

Two regime mismatches follow. The scale regimes differ by 1.6×, so box
regression is learned predominantly at one effective object scale and applied at
another. The density regimes differ by 5×, and Hungarian matching operates per
image, so 81% of optimisation steps specialise queries for a sparsity MOT20 does
not have. The 33.2 overlapping pairs per frame are the visible consequence:
one-to-one matching is not producing duplicate-free output. Note this paragraph
states the sparsity hypothesis, and it remains **untested** — density and
training-mix sparsity are confounded in every arm measured so far, and arm C's
detections would separate them. See `docs/experiment-report.md` §2.3 and Beads
`mot-0p4`.

Selected as the first Tier 1 experiment. Three arms, single-variable, all on the
**ablation** build so `val_half` stays a valid yardstick, and all from the same
`weights/rfdetr/rf-detr-xxlarge.pth` base with the current resolution and
schedule so only the mix changes:

| Arm | Training mix | MOT20 share | Peak mAP@50:95 | Peak epoch | Final | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A | Current: 81% CrowdHuman / 19% MOT20 | 18.8% | **0.6202** | 5 of 50 | 0.5844 | Done — HOTA 68.81 |
| B | MOT20 oversampled 4× to ~50% of images | 48.0% | 0.6168 | 1 of 30 | 0.5696 | Done 2026-09-07 |
| C | MOT20-only fine-tune, no CrowdHuman | 100% | 0.6139 | 9 of 100 | 0.5579 | Done 2026-09-07 |

**Verdict: the mix hypothesis is falsified.** Sweeping the MOT20 share across
essentially its whole range, 18.8% → 48.0% → 100%, moves peak mAP@50:95 by
0.0063, about 1% relative, and moves it the *wrong* way: the unmodified arm A is
the best of the three. Every arm also reproduces the same shape — an early peak
followed by monotone decline — so removing CrowdHuman entirely does not fix it.
Whatever causes the post-peak decline is not the training mix.

Two consequences. The 33.2 duplicate pairs per frame cannot be blamed on
CrowdHuman's sparsity, since arm C never saw CrowdHuman. And the reasoning above
about scale and density regimes, while measured correctly, does not predict the
outcome; it is retained here as a recorded hypothesis that testing rejected.

This raises the prior on I6 (resolution), which is the one cause with direct
measured support: localization, not recall, is what separates this detector from
the baseline, and the cap binds before the resolution target.

- Oversampling mechanism for arm B: manifest duplication, declared via
  `intentional_oversampling` metadata so the audit accepts the duplicate
  `file_name`s instead of rejecting them as an accident.
- Peak epochs recorded above. Note these are absurdly early — arm B peaks at
  epoch 1 of 30 — so a fixed-epoch competition build should use single-digit
  epochs, not the 30–50 originally budgeted.
- Carrying an arm through to HOTA is deferred: no arm beat arm A on mAP, so
  there is no candidate that would plausibly beat its HOTA 68.81.

### I6 — Raise the training long-side cap

`RandomResize([1120], max_size=1333)` means the 1333 cap binds at 1920×1080, not
the 1120 target: the model sees 1333×750, a scale of 0.694. The baseline
YOLOX-X runs at `test_size = (896, 1600)`, a scale of 0.830, and therefore sees
every pedestrian about 20% larger. Measured edge residuals are 1.27–1.46× the
baseline's.

- `max_size = 1600` puts MOT20 at 0.833, matching the baseline exactly, for
  about 1.44× the pixels. Run as arm D, `rfdetr-2xl-i5-armd-2026-09-07-r1`,
  on the arm-A mix so resolution is the only variable against arm A.
  Micro-batch 4 with `grad_accum_steps = 2` keeps the effective batch at 64:
  arm B measured 21 GB of 24 GB at the 1333 cap, so 1.44× the tokens does
  not fit at micro-batch 8. Measured 14 GB per card at micro-batch 4.
  Completed 2026-09-08 05:39, 27,795s, 30 epochs, peak 0.6282 at epoch 9.

**Result: I6 works, but not by the mechanism it was proposed for.** Carried
through the full chain to HOTA, held-out OSNet ReID, `btpp-default`:

| | YOLOX baseline | RF-DETR arm A | RF-DETR arm D |
| --- | ---: | ---: | ---: |
| HOTA | 70.208 | 68.809 | **70.777** |
| DetA | 73.496 | 71.928 | 72.979 |
| AssA | 67.133 | 65.965 | **68.775** |
| LocA | 88.177 | 85.190 | 85.321 |
| IDF1 | 82.536 | 83.255 | **85.637** |
| IDSW | 1,013 | 1,571 | 1,262 |
| Frag | 5,243 | 4,289 | **3,704** |
| CLR_FN | 84,122 | 59,544 | **51,462** |

This is the first configuration to beat the baseline: HOTA +0.569 over YOLOX and
+1.968 over arm A.

The mechanism, however, contradicts the stated rationale. **LocA moved 85.190 →
85.321, by 0.131**, and is still 2.856 below the baseline. Raising the resolution
did not fix box regression in the way this section predicted, even though
mAP@75 on the exported checkpoint did improve, 0.7400 → 0.7523. That mAP gain
translated into almost no LocA gain.

The HOTA gain is instead almost entirely association: AssA +2.810 over arm A,
which now exceeds the baseline's AssA by 1.642. The route is recall, not
localization — DetRe 76.952 → 78.256 and 8,082 fewer false negatives give the
tracker more continuous evidence per identity, so identities survive: IDSW
1,571 → 1,262, Frag 4,289 → 3,704, IDF1 83.255 → 85.637.

So the localization deficit is real, unfixed, and still the largest single gap
to the baseline. It is simply no longer the binding constraint on HOTA. Precision
remains the other weakness: DetPr 82.300 versus 87.688, and 21,242 false
positives versus 6,947.

- Localization is now the clearest remaining target, but resolution is not
  the lever. Reconsider I8 (CrowdHuman box convention) and the regression
  loss weighting before spending another run on pixels.
- Do not attempt this at inference only. Probes at caps 1600 and 1920 leave
  recall flat, 0.9134 → 0.9138 → 0.9167, while $mAP_{75}$ falls 0.7400 →
  0.5943 → 0.6890. The model only regresses boxes well at its trained
  geometry.

### I7 — Training schedule and EMA

- `lr_scheduler = "step"` with `lr_drop = 40` never fired usefully, because
  the model peaked at epoch 5. Reconsider the schedule length and shape once
  the mix is fixed; a shorter cosine run is the obvious candidate.
- `use_ema = true`, yet the selected checkpoint came from the regular
  branch, not the EMA branch. Determine whether EMA was actually updating.

### I8 — CrowdHuman box convention

Phase 8b proved the **validation** labels are byte-identical to MOT20 `gt.txt`,
mean IoU 0.9995. It did not check the **training** labels, and CrowdHuman is 81%
of them. CrowdHuman full-body boxes are amodal annotator estimates of occluded
extent, produced by a different process than MOT20's.

## Decisions taken

| Date | Decision | Rationale |
| --- | --- | --- |
| 2026-09-06 | Two builds: ablation keeps `val_half` held out, competition folds it into training and takes a fixed-epoch final checkpoint | Preserves a valid yardstick for I5/I6/I7 while giving ByteTrack parity for the test submission |
| 2026-09-06 | Run I5 before I6, single-variable, on the ablation build | I5 is roughly 5× cheaper per epoch and has the cleaner hypothesis; running it first establishes the recipe before spending on resolution |

## Open decisions

- Whether to spend GPU time beyond I5 at all, given the detector is currently
  behind the baseline it is meant to replace. Producing MOT20 `test` tracks with
  the current detector is possible today and is a legitimate, if weaker, outcome.
- Which I5 arm to prefer if both improve: 50/50 oversampling keeps CrowdHuman's
  variety, MOT20-only removes the regime mismatch entirely.
