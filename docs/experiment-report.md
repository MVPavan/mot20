# Experiment Report — RF-DETR 2XL vs ByteTrack YOLOX-X on MOT20 `val_half`

Narrative summary of every detector experiment run in this repository, with the
detection matrix and the TrackEval tracking matrix side by side against the
published ByteTrack YOLOX-X baseline, and the conclusions that follow.

**Authority.** This file is *interpretation*. Every figure it quotes is
reproduced from [`docs/results-reference.md`](results-reference.md), which is
generated from stored artifacts by
[`tracking/scripts/build_results_reference.py`](../tracking/scripts/build_results_reference.py)
and is the authoritative source for values. Where the two ever disagree, the
generated file is right and this one is stale. Task status lives in
[`docs/tracker-improvements.md`](tracker-improvements.md) and
[`docs/tracker-todo.md`](tracker-todo.md); contracts and artifact naming live in
[`docs/tracker-experiments.md`](tracker-experiments.md).

## Read this first

Three caveats bound everything below.

1. **Nothing here is leaderboard-comparable.** All work is classified
   `local_test_adapted` per [`docs/MOTPolicy.md`](MOTPolicy.md): the detector's
   training mix contains 21 human-audited Byte65 MOT20-*test* images. The split
   used throughout is MOT20 `val_half` — the second half of MOT20-01/02/03/05,
   4,463 frames, 615,137 pedestrian boxes, 1,418 identities.
2. **The YOLOX baseline is measured here, not quoted.** Every `yoloxx20` row was
   produced by this repository's own runner and vendored TrackEval, not copied
   from the BoostTrack++ repo or paper. Only the detection file differs between
   a baseline row and an RF-DETR row, so the deltas isolate the detector.
3. **The `yoloxx20` baseline is *not* held out — it trained on these frames.**
   See §0. This is the single largest caveat in the document.
4. **The headline comparison uses held-out ReID.** `osnet-ain-msdc` has never
   seen MOT20. Rows using `fastreid-sbs-s50-mot20` are inflated on both sides,
   because that checkpoint is MOT20-trained and has seen these identities.

## 0. The baseline is contaminated on this split

`val_half` is the **second half of the MOT20 *train* sequences**. The published
ByteTrack MOT20 detector was trained on the *full* MOT20 train set, so it has
seen every frame it is evaluated on here. The RF-DETR arms have not.

| | Trained on | Sees `val_half` during training? |
| --- | --- | --- |
| RF-DETR arms A–D | `mot20_train_half` + CrowdHuman + 21 Byte65 | **No** — held out |
| `yoloxx20` (`bytetrack_x_mot20.tar`) | **full MOT20 train** + CrowdHuman | **Yes** |
| `yoloxx17` (`bytetrack_x_mot17.pth.tar`) | MOT17 + CrowdHuman + CityPersons + ETHZ | No — never saw MOT20 at all. **Excluded by decision, 2026-09-08** |

### Provenance chain, verified

1. **`val_half` is literally the second half of MOT20 train.**
   `datasets/README.md` documents the ranges (MOT20-01 216–429, -02 1393–2782,
   -03 1204–2405, -05 1659–3315, rebased to frame 1). Spot-checked by md5: the
   first and last image of each `val_half` sequence are byte-identical to the
   corresponding original train frames. The four sequences are MOT20-01/02/03/05,
   which are the *train* sequences; test is 04/06/07/08.
2. **The `yoloxx20` variant reads those files and nothing else.** Its manifest is
   `origin: registered-external`, `source_pattern:
   datasets/val_half/{sequence}/det_yoloxx20/det_yoloxx20.txt`, totalling 580,369
   detections over 4,463 frames — exactly the row count of those four files.
3. **Those files come from the MOT20-trained detector.** `datasets/README.md`
   labels them *"YOLOX-X detections from the MOT20-trained detector"* and already
   warns: *"Because the `x20` detector is described as trained on the complete
   MOT20 training set, its results on this temporal split should not be treated
   as clean detector holdout measurements."*

Corroborating, from outside the repo:

- **BoostTrack's own code refuses to use it for validation.** In
  `repos/BoostTrack/default_settings.py`, `--dataset mot20` selects
  `bytetrack_x_mot20.tar` at (896, 1600) *only* when `args.test_dataset` is set.
  Otherwise it falls back with the comment *"Just use the mot17 test model as the
  ablation model for 20"*.
- **ByteTrack publishes no MOT20 half-split ablation model.** `yolox_x_mix_mot20_ch.py`
  is the *test* model, trained on `mix_mot20_ch` = MOT20 train + CrowdHuman.
  The only half-split ablation config, `yolox_x_ablation.py`, is MOT17-based.
- **The measurements have the shape of a leak.** `yoloxx20` reaches precision
  0.9859 at IoU 0.5 with only 6,947 false positives across 615,137 ground-truth
  boxes, and mAP@75 of 0.8278 — on frames it trained on.

**Limit of this verification.** No YOLOX weights are on disk, so the detector
identity rests on the dataset README's own labelling and BoostTrack's
configuration mapping, not on a checksum of the checkpoint. The frame overlap,
by contrast, is verified byte-for-byte.

**Consequence for every comparison below.** The RF-DETR-vs-YOLOX deltas are
measured against an advantaged opponent, so they are *lower bounds*. Arm D's
HOTA +0.569 is a win over a detector that had the answers; the held-out margin
should be larger, by an amount nothing here measures.

The RF-DETR-vs-RF-DETR comparisons are unaffected — arms A–D share byte-identical
train and valid manifests, so §1.1, §2.2 and every arm-to-arm delta are clean.

**No clean external baseline is available, by decision.** `det_yoloxx17`
detections sit unregistered at `datasets/val_half/<seq>/det_yoloxx17/` and are
the comparison BoostTrack itself would make, but MOT17 is out of scope for this
project (decided 2026-09-08). The consequence is accepted deliberately:
`yoloxx20` stays the reference, its leakage is permanent and unquantified, and
**every cross-detector delta in this document carries the §0 caveat for good**.

Building a clean baseline would mean training YOLOX-X on `mot20_train_half`
ourselves. That is the only way to remove the caveat, and it has not been costed.
Until then the trustworthy comparisons are RF-DETR arm-to-arm, which share
byte-identical manifests.

## 1. What was run

### 1.1 Detector training arms

Four arms, all from the same base checkpoint `rf-detr-xxlarge.pth`
(sha256 `bf418652…c5d553ae`), `num_queries = num_select = eval_max_dets = 390`,
`group_detr = 13`, BF16, lr 5e-5, effective batch 64 on 8× RTX 3090, and a
byte-identical held-out `valid` split. Arms A–C change only the training mix;
arm D changes only the input geometry.

| Arm | What varies | MOT20 share | Train images | Peak mAP@50:95 | at epoch | Final |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | baseline mix, long-side cap 1333 | 18.8% | 23,859 | 0.6202 | 5 of 50 | 0.5844 |
| B | MOT20 oversampled 4× | 48.0% | 37,263 | 0.6168 | 1 of 30 | 0.5696 |
| C | MOT20 only, no CrowdHuman | 100% | 4,489 | 0.6139 | 9 of 100 | 0.5579 |
| D | arm-A mix, **long-side cap 1600** | 18.8% | 23,859 | **0.6282** | 9 of 30 | 0.6140 |

Two facts about this table matter more than the peaks themselves.

**Every arm peaks absurdly early and then declines monotonically.** Arm B peaks
at epoch 1 of 30. Whatever causes the decline survives removing CrowdHuman
entirely (arm C), so it is not the training mix. A fixed-epoch competition build
should budget single-digit epochs, not the 30–50 originally planned.

**Arm D is the only arm that beat arm A.** Sweeping the MOT20 share across
essentially its whole range moves peak mAP by 0.0063 — about 1% relative — and
moves it the *wrong* way. Changing the input geometry moved it +0.0080 the right
way, on the same data.

### 1.2 Why the geometry change was the one that worked

RF-DETR's training transform is `RandomResize([1120], max_size=1333)`. At MOT20's
1920×1080 the **1333 cap binds, not the 1120 target**: the model was seeing
1333×750, a scale of 0.694. The `resolution = 1120` setting was inert. The
baseline YOLOX-X runs `test_size = (896, 1600)` → 1593×896, a scale of 0.830, and
therefore sees every pedestrian about 20% larger.

Arm D sets `max_size = 1600`, which puts MOT20 at 0.833 — matching the baseline
exactly — for about 1.44× the pixels, at micro-batch 4 × `grad_accum_steps` 2 so
the effective batch stays 64. Measured 14 GB per card.

This is a **training-time** fix. Raising the cap at inference only makes things
worse: recall stays flat (0.9134 → 0.9138 → 0.9167 at caps 1333/1600/1920) while
mAP@75 collapses (0.7400 → 0.5943 → 0.6890). The model regresses boxes well only
at the geometry it was trained at.

### 1.3 Detection variants carried to tracking

| Slug | Meaning |
| --- | --- |
| `yoloxx20` | ByteTrack's published YOLOX-X MOT20 detector, `test_size=(896,1600)`, `nmsthre=0.7`, `test_conf=0.001`. **Trained on full MOT20 train — see §0** |
| `rfdetr2xl-e5-t005` | arm A, epoch-5 checkpoint, export score threshold 0.05, no NMS |
| `rfdetr2xl-e5-t010-nms070` | arm A, threshold 0.10 then greedy NMS at IoU 0.70 |
| `rfdetr2xl-armd-e9-t005` | arm D, epoch-9 checkpoint, threshold 0.05, no NMS |
| `rfdetr2xl-armd-e9-t010-nms070` | arm D, threshold 0.10 then greedy NMS at IoU 0.70 |

RF-DETR is a set-prediction model and exports without NMS by design. That turned
out to be wrong for this data: at MOT20 density one-to-one matching does not
converge, and the raw export carries **33.2 duplicate pairs per frame at
IoU ≥ 0.75, in every one of the 4,463 frames**. Greedy NMS at IoU 0.70 removes
all of them and is the export default.

## 2. Detection matrix

### 2.1 Under one COCO protocol

[`tracking/scripts/analyze_detections.py`](../tracking/scripts/analyze_detections.py):
pycocotools against the same `valid` annotations for both detectors, MOT20 ignore
regions carried as `iscrowd`, `maxDets = 390` for every variant. Neither
detector's own evaluation code is involved.

| Metric | arm A `-t005` | arm A `-t010-nms070` | arm D `-t005` | arm D `-t010-nms070` | `yoloxx20` |
| --- | ---: | ---: | ---: | ---: | ---: |
| mAP@50 | 0.9585 | 0.9510 | **0.9663** | 0.9528 | 0.9000 |
| mAP@75 | 0.7595 | 0.7483 | 0.7716 | 0.7603 | **0.8278** |
| mAP@50:95 | 0.6369 | 0.6297 | 0.6446 | 0.6351 | **0.6759** |
| AR@50:95 | 0.6957 | 0.6811 | 0.7018 | 0.6892 | **0.7104** |
| mAP small | 0.1651 | 0.1626 | 0.1726 | 0.1698 | **0.3074** |
| mAP medium | 0.5983 | 0.5895 | 0.6030 | 0.5963 | **0.6536** |
| mAP large | 0.7058 | 0.6995 | 0.7170 | 0.7110 | **0.7216** |
| boxes total | 1,265,422 | 865,842 | 1,261,854 | 873,131 | 580,369 |
| boxes/frame mean | 283.54 | 194.00 | 282.74 | 195.64 | 130.04 |
| dup pairs/frame @IoU≥0.75 | 33.219 | 0.000 | 31.400 | 0.000 | 0.000 |
| frames with duplicates | 4,463 | 0 | 4,463 | 0 | 0 |
| box height median (px) | 127.3 | 132.7 | 128.0 | 132.7 | 135.5 |

**The shape of the detection result is a crossover at IoU threshold.** RF-DETR
wins mAP@50 by up to +0.066; YOLOX wins mAP@75 by +0.068 and mAP@50:95 by
+0.041 even against arm D. RF-DETR finds more of the people; YOLOX puts tighter
boxes on the ones it finds. The gap widens as the IoU requirement rises, which is
the signature of a localization deficit, not a recall deficit.

**Arm D improves every column but does not change the verdict.** Against arm A at
the same export setting: mAP@50 +0.0018, mAP@75 **+0.0120**, mAP@50:95 +0.0054,
AR +0.0081. The largest gain is again at the strict IoU threshold, exactly where
a resolution fix should show. It still trails YOLOX by 0.0408 on mAP@50:95 and
0.0675 on mAP@75. **Detection says YOLOX wins, for both arms.** §4 shows tracking
says the opposite for arm D — see §5.

**Resolution did not fix the duplicates either.** Arm D's raw export still carries
31.4 duplicate pairs per frame in all 4,463 frames, against arm A's 33.2. The
one-to-one matching failure at MOT20 density is not a resolution artifact.

The small-object column is the worst single number: 0.1698 against 0.3074. It
carries little weight here — only 1.2% of `val_half` boxes are COCO-small — but
arm D moved it +0.0072, consistent with the geometry finding, since a 0.694
downscale hurts the smallest boxes most.

### 2.2 Under RF-DETR's own validation loop, arm A vs arm D

The library's own ignore-aware validation, on the exported checkpoints. This is
the like-for-like measurement of what the geometry change bought.

| Metric | arm A, epoch 5 (cap 1333) | arm D, epoch 9 (cap 1600) | Δ |
| --- | ---: | ---: | ---: |
| `val/mAP_50` | 0.9345 | 0.9389 | +0.0044 |
| `val/mAP_75` | 0.7400 | 0.7523 | **+0.0123** |
| `val/mAP_50_95` | 0.6209 | 0.6280 | +0.0071 |
| `val/mAR` | 0.6987 | 0.7046 | +0.0059 |
| `val/precision` | 0.8922 | 0.8977 | +0.0055 |
| `val/recall` | 0.9134 | 0.9197 | +0.0063 |
| `val/F1` | 0.9027 | 0.9086 | +0.0059 |
| `val/cardinality_error` | 7.8443 | 7.3798 | −0.4645 |
| `val/loss_bbox` | 0.0171 | 0.0165 | −0.0006 |
| `val/loss_giou` | 0.1949 | 0.1889 | −0.0060 |

mAP@75 improved most, which is exactly what a resolution fix should do. Hold on
to that: §4 shows it did **not** translate into tracking-time localization.

### 2.3 Set prediction does not converge at MOT20 density

This is the finding with the widest implications, so it is stated once here in
full rather than left scattered across the tables above.

**The claim being tested.** DETR-family detectors replace NMS with one-to-one
Hungarian matching during training: each ground-truth object is assigned exactly
one query, so at convergence the model emits one box per object and duplicate
suppression is unnecessary. That is the architecture's selling point, and RF-DETR
exports without NMS because of it.

**What was measured.** On `val_half`, raw exports before any suppression:

| | arm A `-t005` | arm D `-t005` | `yoloxx20` |
| --- | ---: | ---: | ---: |
| Duplicate pairs per frame at IoU ≥ 0.75 | 33.219 | 31.400 | 0.000 |
| Frames containing at least one duplicate | **4,463 of 4,463** | **4,463 of 4,463** | 0 |
| Boxes per frame, mean | 283.54 | 282.74 | 130.04 |
| Boxes per frame, max (cap is `num_select = 390`) | 386 | 383 | 213 |
| `val/cardinality_error` | 7.8443 | 7.3798 | — |

Ground truth averages 137.8 instances per image and peaks at 220. The detector is
emitting roughly twice that, in every frame, and in the densest frames it comes
within 4 boxes of its own 390-query ceiling.

**Why: density.** COCO val2017 averages about 7 instances per image. MOT20 is
about 19× denser. One-to-one matching is a per-image assignment problem, and its
difficulty scales with the number of objects competing for queries. At 137.8
mean instances the assignment does not converge to a clean one-to-one solution,
so the model hedges by placing several queries on the same person. The
`cardinality_error` of 7.84 is the model's own count of how far off it is.

**What has been ruled out.** Resolution. Arm D runs at 1.44× the pixels with the
effective scale matched to the baseline's 0.830, and its duplicate rate is 31.4
against arm A's 33.2 — still in 100% of frames. Whatever is failing here is not
an input-resolution artifact.

**What has *not* been ruled out: CrowdHuman's sparsity.** The obvious hypothesis
is that 81% of training images averaging 22.7 persons teaches the matching to
expect sparsity. That hypothesis is **untested**. Arm C trained on MOT20 only
with no CrowdHuman at all and would settle it, but arm C's detections were never
exported, so its duplicate rate has never been measured. Arm C tells us only
that removing CrowdHuman does not change the *mAP curve shape*; it says nothing
about duplicates. Both arms measured above are CrowdHuman-heavy.

**What works instead.** Reintroducing classical greedy NMS at IoU 0.70 — the same
`nmsthre` the published YOLOX detections were produced with — removes every
duplicate and is worth **+0.63 HOTA**. The suppression IoU matters: none / 0.70 /
0.60 / 0.50 gives HOTA 68.18 / 68.81 / 68.51 / 66.08.

**The cost.** RF-DETR paid set prediction's price — a per-image assignment that
scales badly with density, and a hard `num_select` ceiling that the data nearly
saturates — without collecting its benefit, which is duplicate-free output. On
this data the architecture's distinguishing mechanism is not merely inactive; it
is producing input that is *harder* for the tracker than a conventional detector
would, since each duplicate pair is another chance for association to bind an
identity to the wrong box.

**And mAP would never have found this.** NMS *lowers* mAP@50:95 (0.6369 → 0.6297
for arm A) while *raising* HOTA. COCO AP tolerates extra low-ranked duplicates
that a tracker must resolve into identities. Optimising the detector on AP alone
would have kept the worse setting.

## 3. Localization matrix

[`tracking/scripts/analyze_localization.py`](../tracking/scripts/analyze_localization.py):
Hungarian match against `gt.txt` at IoU 0.5, detections filtered at score ≥ 0.40
(the tracker's `det_thresh`). Residuals are per-edge and normalised by
ground-truth box size, which separates a correctable convention offset from
uncorrectable regression jitter.

| Metric | arm A `-t010-nms070` | arm D `-t010-nms070` | `yoloxx20` |
| --- | ---: | ---: | ---: |
| boxes | 634,487 | 639,978 | 553,800 |
| matched | 564,045 | 569,751 | 546,012 |
| recall @IoU 0.5 | 0.9169 | **0.9262** | 0.8876 |
| precision @IoU 0.5 | 0.8890 | 0.8903 | **0.9859** |
| mean matched IoU | 0.8336 | 0.8356 | **0.8713** |
| median matched IoU | 0.8475 | 0.8492 | **0.8845** |
| fraction IoU ≥ 0.70 | 0.9290 | 0.9327 | **0.9740** |
| fraction IoU ≥ 0.75 | 0.8524 | 0.8593 | **0.9421** |
| fraction IoU ≥ 0.80 | 0.7111 | 0.7204 | **0.8659** |
| fraction IoU ≥ 0.90 | 0.2143 | 0.2213 | **0.3926** |
| edge residual mean x1 | −0.0046 | −0.0064 | −0.0006 |
| edge residual mean y2 | +0.0012 | +0.0030 | +0.0010 |
| edge residual \|mean\| x1 | 0.0602 | 0.0603 | **0.0468** |
| edge residual \|mean\| x2 | 0.0629 | 0.0611 | **0.0473** |
| edge residual \|mean\| y1 | 0.0234 | 0.0225 | **0.0184** |
| edge residual \|mean\| y2 | 0.0449 | 0.0445 | **0.0308** |

Three things this establishes.

**The error is jitter, not an offset.** Signed residual means are near zero on
every edge for every variant, while the absolute means are 1.22–1.44× the
baseline's. There is no systematic shift to subtract out, so no annotation
convention fix is available; this is regression noise.

**Raising resolution barely touched it.** Arm D's absolute residuals are
0.0603 / 0.0611 / 0.0225 / 0.0445 against arm A's 0.0602 / 0.0629 / 0.0234 /
0.0449. Mean matched IoU moved 0.8336 → 0.8356, by 0.0020. The mAP@75 gain in
§2.2 did not survive contact with the tracker's operating point.

**The tracker is not the culprit.** Measuring the same statistic before and after
BoostTrack++ leaves it unchanged (arm D: 0.8356 detections → 0.8362 tracks;
YOLOX: 0.8713 → 0.8726). Kalman smoothing is not degrading box quality. Whatever
localization is lost, is lost in the detector.

**Label conventions are identical.** Measured directly against each other, with
no model involved, the COCO training annotations and `gt.txt` match on 615,137 of
615,137 boxes at mean IoU 0.9995. A label mismatch cannot explain any of this.

## 4. Tracking matrix (TrackEval)

Vendored TrackEval, BoostTrack++ at its published MOT20 settings with
`det_thresh = 0.40`, held-out `osnet-ain-msdc` ReID, `val_half`.

### 4.1 Headline

| Metric | `yoloxx20` | RF-DETR arm A | RF-DETR arm D | D − YOLOX |
| --- | ---: | ---: | ---: | ---: |
| **HOTA** | 70.208 | 68.809 | **70.777** | **+0.569** |
| DetA | **73.496** | 71.928 | 72.979 | −0.517 |
| AssA | 67.133 | 65.965 | **68.775** | **+1.642** |
| LocA | **88.177** | 85.190 | 85.321 | −2.856 |
| MOTA | 85.031 | 86.883 | **87.976** | **+2.945** |
| MOTP | **87.084** | 83.357 | 83.545 | −3.539 |
| IDF1 | 82.536 | 83.255 | **85.637** | **+3.101** |

Arm D is the first configuration in this project to beat the baseline on HOTA:
**+0.569 over YOLOX and +1.968 over arm A**, from a single change to the training
long-side cap.

The +1.968 over arm A is a clean, controlled delta — identical data, identical
tracker, one variable. The +0.569 over YOLOX is a **lower bound**: per §0, that
baseline trained on these frames. Its LocA 88.177 and DetPr 87.688 are the two
columns most likely inflated by having seen the boxes.

### 4.2 The components, decomposed

HOTA is the geometric mean of DetA and AssA integrated over localization
thresholds α ∈ 0.05…0.95, and LocA is the mean IoU over matched pairs. Splitting
each into its precision and recall halves shows exactly where each detector wins.

| Metric | `yoloxx20` | arm A | arm D | D − YOLOX |
| --- | ---: | ---: | ---: | ---: |
| DetPr | **87.688** | 82.300 | 82.300 | −5.388 |
| DetRe | 76.687 | 76.952 | **78.256** | **+1.569** |
| AssPr | **83.337** | 79.746 | 81.295 | −2.042 |
| AssRe | 71.174 | 70.981 | **73.428** | **+2.254** |

Every RF-DETR win is a recall win; every loss is a precision or localization
loss. That is one consistent trade, visible in all four halves.

### 4.3 CLEAR and identity counts

| Metric | `yoloxx20` | arm A | arm D | D − YOLOX |
| --- | ---: | ---: | ---: | ---: |
| CLR_TP | 531,015 | 555,593 | **563,675** | +32,660 |
| CLR_FN | 84,122 | 59,544 | **51,462** | **−32,660** |
| CLR_FP | **6,947** | 19,570 | 21,242 | +14,295 |
| IDSW | **1,013** | 1,571 | 1,262 | +249 |
| Frag | 5,243 | 4,289 | **3,704** | **−1,539** |
| MT (of 1,418) | 1,021 | 1,139 | **1,171** | +150 |
| ML | **47** | 57 | 51 | +4 |
| IDP | **88.456** | 86.148 | 87.849 | −0.607 |
| IDR | 77.359 | 80.550 | **83.534** | **+6.175** |
| Output boxes | 537,962 | 575,163 | 584,917 | +46,955 |

This is the clearest single view of the trade. Arm D misses **32,660 fewer
people** than YOLOX and produces **14,295 more false positives** — roughly 2.3
recovered detections per extra false alarm. It follows 150 more identities mostly
tracked and fragments 1,539 fewer times, at the cost of 249 more identity
switches.

### 4.4 Every measured run, held-out OSNet ReID

| Run | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arm D `-t010-nms070`, defaults | **70.777** | 72.979 | 68.775 | 85.321 | 87.976 | 85.637 | 1,262 |
| `yoloxx20`, defaults | 70.208 | 73.496 | 67.133 | 88.177 | 85.031 | 82.536 | 1,013 |
| arm A `-t010-nms070`, defaults | 68.809 | 71.928 | 65.965 | 85.190 | 86.883 | 83.255 | 1,571 |
| arm A `-t010-nms0600`, defaults | 68.510 | 70.964 | 66.272 | 85.212 | 85.749 | 83.388 | 1,427 |
| arm A `-t005`, `det_thresh=0.50` | 68.353 | 71.683 | 65.311 | 85.321 | 86.457 | 82.793 | 1,615 |
| arm A `-t010`, defaults | 68.175 | 72.068 | 64.640 | 85.137 | 87.012 | 81.874 | 2,065 |
| arm A `-t005`, defaults | 68.080 | 71.996 | 64.526 | 85.128 | 87.007 | 81.805 | 2,113 |
| arm A `-t005`, `det_thresh=0.60` | 68.018 | 70.619 | 65.635 | 85.462 | 85.176 | 83.117 | 1,323 |
| arm A `-t010-nms0500`, defaults | 66.076 | 68.483 | 63.883 | 85.271 | 82.654 | 81.495 | 1,297 |

NMS at IoU 0.70 is the sweet spot: 0.60 costs 0.30 HOTA and 0.50 costs 2.73.
Raising `det_thresh` above 0.40 never helped.

### 4.5 Association sweep, FastReID (NOT held out)

Twenty-one association configurations on arm A. `fastreid-sbs-s50-mot20` has seen
these identities, so absolute values are inflated on both sides; the delta
remains meaningful because both detectors use it.

| Run | HOTA | DetA | AssA | LocA | IDF1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `yoloxx20`, `lambda_iou=0.35` | **70.734** | 73.587 | 68.059 | 88.210 | 83.165 |
| `yoloxx20`, defaults | 70.547 | 73.610 | 67.677 | 88.210 | 82.877 |
| arm A, `iou_threshold=0.25, lambda_iou=0.35, lambda_shape=0.35` | 69.718 | 71.999 | 67.642 | 85.206 | 84.510 |
| arm A, `lambda_iou=0.35` | 69.695 | 72.030 | 67.570 | 85.220 | 84.409 |
| arm A, defaults | 69.509 | 71.970 | 67.269 | 85.207 | 84.156 |
| arm A, `min_hits=5` | 69.172 | 71.386 | 67.157 | 85.302 | 84.079 |

**The whole sweep spans 0.55 HOTA and does not close the gap.** Best RF-DETR
association tuning gains +0.21 over defaults, but the baseline gains with it, so
the gap only moved from 2.13 to 1.02. `dlo_boost_coef` at 0.3 and 0.7 produced
byte-identical results — that knob does nothing on this data.

Switching ReID from OSNet to FastReID was worth +0.70 HOTA. **Arm D has not yet
been run with FastReID**, so the FastReID column has no arm D entry; that is the
one missing cell in this report.

## 5. Reading the two matrices together

The detection matrix and the tracking matrix disagree, and the disagreement is
the most useful result in this project.

| | Detection says | Tracking says |
| --- | --- | --- |
| Winner | `yoloxx20`, mAP@50:95 0.6759 vs arm D's 0.6351 | arm D, HOTA 70.777 vs 70.208 |
| Margin | −0.041 against RF-DETR | +0.569 for RF-DETR |

Both matrices are now measured for the same checkpoint under the same protocols,
and they point in opposite directions. This is not arm A being weak: arm D is
the better detector on every mAP column *and* still loses the detection matrix
while winning the tracking matrix.

Three reasons they diverge.

**mAP integrates over a recall axis the tracker never uses.** COCO AP averages
precision over all recall levels including the low-confidence tail. BoostTrack++
thresholds at `det_thresh = 0.40` and throws that tail away. The 1,265,422 raw
RF-DETR boxes become 638,477 at the tracker's threshold — half the detector's
mAP-relevant output is never seen by the tracker.

**mAP has no notion of temporal continuity.** A recovered detection is worth one
true positive to COCO. To HOTA it can be worth an entire identity: it is the
frame that keeps a track alive across an occlusion. That asymmetry is why arm D's
1,539 fewer fragmentations buy more HOTA than its 14,295 extra false positives
cost.

**NMS is the sharpest case of the two disagreeing.** Greedy NMS at IoU 0.70
*lowers* mAP@50:95 (0.6337 → 0.6297) while *raising* HOTA (68.175 → 68.809). A
set-prediction detector exported without NMS scores well and hands the tracker a
strictly harder input: 33.2 duplicate pairs per frame is 33.2 chances per frame
for the association step to bind an identity to the wrong box.

**Practical consequence: mAP is not a sufficient selection signal for this
project.** Any candidate detector or export setting has to be carried through the
full chain — export, NMS, embeddings, tracking, TrackEval — before it is
accepted. The three-arm mix ablation is the cautionary case in the other
direction: it was stopped at mAP because no arm beat arm A, and that was probably
the right call, but it is an assumption that was never tested.

## 6. Conclusions

**0. Every RF-DETR-vs-YOLOX number here understates RF-DETR.** The baseline
trained on the evaluation frames (§0); the RF-DETR arms did not. Read the
cross-detector deltas as lower bounds and the arm-to-arm deltas as exact. MOT17
is out of scope, so no clean external baseline is available and this caveat is
permanent rather than pending.

**1. The best configuration is arm D, and it beats the baseline even so.** HOTA
70.777 against 70.208, with held-out ReID and identical tracker settings on both
sides. Also ahead on MOTA (+2.945), IDF1 (+3.101), fragmentation (−1,539) and
mostly tracked (+150).

**2. The win came from geometry, not from data.** Three arms sweeping the MOT20
share from 18.8% to 100% moved peak mAP by 1% relative, the wrong way. One arm
changing the effective input scale from 0.694 to 0.833 moved it +0.008 the right
way and +1.968 HOTA. The mix hypothesis is falsified and recorded as such.

**3. The win did not arrive by its predicted mechanism.** I6 was proposed to fix
box regression. **LocA moved 85.190 → 85.321, by 0.131**, and remains 2.856 below
the baseline. The gain is association: AssA +2.810 over arm A, which now exceeds
the baseline's. The route is recall — DetRe 76.952 → 78.256, 8,082 fewer misses —
giving the tracker more continuous evidence per identity, so identities survive.

**4. RF-DETR trades precision for recall, consistently, at every level.** It wins
DetRe, AssRe, IDR, MT, CLR_FN and mAP@50; it loses DetPr, AssPr, IDP, CLR_FP,
LocA and mAP@75. This is one property of the detector showing up in twelve
metrics, not twelve findings.

**5. Localization is the largest remaining gap, and resolution is not the lever.**
2.856 LocA and 1.22–1.44× the baseline's edge jitter. The error is symmetric, so
there is no convention offset to correct, and it is not introduced by the
tracker. Arm D shows the obvious lever has been pulled and yielded 0.131.

**6. Set prediction without NMS is wrong for MOT20 density.** 33.2 duplicate
pairs per frame, in 4,463 of 4,463 frames, and arm D's 1.44× resolution barely
moved it (31.4, still every frame). One-to-one Hungarian matching does not
converge at 137.8 instances per image, about 19× COCO's density. Whether
CrowdHuman's sparsity is the cause remains **untested** — that needs arm C's
detections exported, which has never been done. See §2.3.

**7. The training schedule is broken and remains unexplained.** Every arm peaks
in single-digit epochs and declines monotonically for the rest of the run. Arm A
lost 0.037 mAP over its last 45 epochs (0.6202 at epoch 5 → 0.5832 at epoch 49). This is compute being spent to make the
model worse, on all four arms, and no hypothesis has been tested yet.

## 7. Open items

- **A clean baseline, if one is ever wanted.** Not `yoloxx17` — MOT17 is out of
  scope. The only in-scope option is training YOLOX-X on `mot20_train_half`
  ourselves. Not costed, not scheduled. Absent it, §0 stands permanently.
- **Arm D with FastReID.** The one missing cell. Needed for a like-for-like
  comparison against the published-ReID column, remembering that checkpoint is
  not held out.
- **Localization, by some lever other than resolution.** Reconsider the
  regression loss weighting. `max_size = 1920` gives native scale at ~2.1× the
  pixels but arm D's evidence says to expect little.
- **I7: why every arm peaks early and decays.** Currently the largest unexplained
  effect in the project, and the cheapest to exploit if understood.
- **I4: MOT20 `val_half` folded into training** for ByteTrack parity. The build
  exists (28,322 images, deliberately empty `valid` split). Once run, `val_half`
  stops being a valid yardstick, so all comparative work should finish first.
- **Phase 9: MOT20 `test`.** Not started.
