# Consolidated Results Reference

Generated 2026-09-08 by `tracking/scripts/build_results_reference.py`.
**Do not edit by hand** — re-run the generator after any new experiment.

Every *measured result* below — detection accuracy, localization, tracking,
training curves, geometry probes, dataset audits — is read at generation time
from a stored manifest, metrics CSV, or analysis JSON, so those cannot drift
from the artifacts. A small number of descriptive constants are still literals
in the generator rather than artifact reads: the frame and identity counts in
1.1, the size/density block in 1.3, the source-domain comparison in 2.1, and
the effective-geometry line in 3. Treat those four as transcribed, not derived.

Interpretation and conclusions live in `docs/experiment-report.md`;
detector-gap root-cause analysis in `docs/tracker-improvements.md`;
integration contracts and artifact naming in `docs/tracker-experiments.md`.
Task status lives in Beads (`bd ready`, `bd list --status=open`).
This file is numbers only.

## Scope and caveats

- Split is MOT20 `val_half`: the second half of MOT20-01/02/03/05, 4,463 frames.
- Every tracking row was produced by this repository's own runner
  (`tracking/scripts/run_boosttrack.py`) and the vendored TrackEval. The
  `yoloxx20` baseline is **measured here**, not quoted from the BoostTrack++ repo
  or paper. Only the detection file differs between a baseline row and an RF-DETR row.
- **The `yoloxx20` baseline is not held out on this split.** Its detections come
  from `bytetrack_x_mot20.tar`, trained on the full MOT20 train set, of which
  `val_half` is the second half. Every RF-DETR-vs-baseline delta here is measured
  against a detector that saw the evaluation frames in training; RF-DETR-vs-RF-DETR
  deltas are unaffected. No YOLOX weights are on disk, so the detector identity
  rests on `datasets/README.md` and BoostTrack's config mapping, not a checksum.
  See `docs/experiment-report.md` section 0.
- All work is classified `local_test_adapted` per `docs/MOTPolicy.md`: the detector's
  training mix contains 21 human-audited Byte65 MOT20-test images, so no result here
  is leaderboard-comparable.
- Rows using `fastreid-sbs-s50-mot20` are **not held out**: that checkpoint is
  MOT20-trained and has seen `val_half` identities. Absolute values are inflated;
  detector-to-detector deltas remain meaningful because both sides use it.

## 1. Ground truth and dataset statistics

### 1.1 `val_half` ground truth

| Quantity | Value |
| --- | ---: |
| Frames | 4,463 |
| Pedestrian boxes (`gt.txt`, conf=1 cls=1) | 615,137 |
| COCO `valid` boxes (`iscrowd=0`) | 615,137 |
| COCO ignore regions (`iscrowd=1`) | 61,382 |
| Ground-truth identities | 1,418 |

### 1.2 Do the training labels match `gt.txt`?

The detector is scored against COCO annotations; HOTA is scored against `gt.txt`.
Measured directly against each other, no model involved:

| Quantity | Value |
| --- | ---: |
| Matched pairs | 615,137 |
| GT coverage | 1.0000 |
| COCO coverage | 1.0000 |
| Mean matched IoU | **0.9995** |
| Median matched IoU | 1.0000 |
| Fraction with IoU ≥ 0.70 | 1.0000 |
| Fraction with IoU ≥ 0.75 | 1.0000 |
| Fraction with IoU ≥ 0.80 | 1.0000 |
| Fraction with IoU ≥ 0.90 | 1.0000 |
| Fraction with IoU ≥ 0.95 | 0.9998 |
| Mean edge residual x1 | 0.0000 |
| Mean edge residual x2 | -0.0000 |
| Mean edge residual y1 | 0.0000 |
| Mean edge residual y2 | -0.0005 |

**The two label sets are the same boxes.** A label-convention mismatch cannot
explain any disagreement between mAP and HOTA.

### 1.2b Overlap inside the ground truth itself

The control for the duplicate-pair figures in section 4. `analyze_detections.py`
counts every *pair of predictions* over an IoU threshold without matching them
to ground truth, so genuinely overlapping distinct people inflate it. Applying
the same rule to `gt.txt`, where every box is a distinct annotated person,
bounds how much of that figure is legitimate crowding.

| IoU threshold | GT pairs per frame | Frames containing one |
| ---: | ---: | ---: |
| 0.50 | 10.3872 | 4,422 (99.08%) |
| 0.60 | 3.9045 | 4,136 (92.67%) |
| 0.75 | 0.5848 | 1,877 (42.06%) |
| 0.90 | 0.0475 | 212 (4.75%) |

Highest GT-to-GT IoU anywhere in the split: 0.9830.

### 1.3 Object size and density, `val_half`

| Quantity | Value |
| --- | ---: |
| Instances per image, mean | 137.8 |
| Instances per image, max | 220 |
| COCO size band: small (<32²) | 7,435 (1.2%) |
| COCO size band: medium | 378,023 (61.5%) |
| COCO size band: large (≥96²) | 229,679 (37.3%) |
| Box height percentiles (px) | p5 63, p25 105, median 137, p75 161 |

For scale, COCO val2017 averages roughly 7 instances per image. MOT20 is about
19× denser, which is the regime DETR-style one-to-one matching struggles in.

## 2. Training dataset builds

| Build | Train images | Train annotations | CrowdHuman | MOT20 | Byte65 | MOT20 share | Valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 23,859 | 1,154,031 | 19,370 | 4,468 | 21 | 18.8% | 4,463 |
| B | 37,263 | 2,908,881 | 19,370 | 17,872 | 21 | 48.0% | 4,463 |
| C | 4,489 | 587,538 | 0 | 4,468 | 21 | 100.0% | 4,463 |
| I4 | 28,322 | 1,830,550 | 19,370 | 8,931 | 21 | 31.6% | 0 |

Arm B repeats the same 4,468 MOT20 images four times; it adds no new imagery.
The I4 build folds `val_half` into training for ByteTrack parity and carries a
deliberately empty `valid` split so validation cannot drive checkpoint selection.

### 2.1 Source-domain mismatch inside the mix

| Quantity | CrowdHuman | MOT20 |
| --- | ---: | ---: |
| Persons per image, mean | 22.7 | 116.3 |
| Median long side (px) | 1,024 | 1,654 |
| Resize scale applied by the training transform | 1.302 (upscaled) | 0.806 (downscaled) |

## 3. Detector training runs

All arms share the same base checkpoint `rf-detr-xxlarge.pth`
(sha256 `bf418652…c5d553ae`), resolution 1120, batch 8 × 8 GPUs, lr 5e-5,
`num_queries = num_select = eval_max_dets = 390`, `group_detr = 13`, BF16, and a
byte-identical held-out `valid` split. Epoch counts are chosen to match
optimisation steps, not epochs, because images per epoch differ up to 8×.

| Arm | Mix | Peak mAP@50:95 | at epoch | at step | Final | Evals | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | 81% CrowdHuman (as trained) | **0.6202** | 5 | 2,237 | 0.5844 | 50 | complete |
| B | 48% MOT20, 4x oversampled | **0.6168** | 1 | 1,165 | 0.5696 | 15 | complete |
| C | 100% MOT20, no CrowdHuman | **0.6139** | 9 | 709 | 0.5579 | 20 | complete |
| D | 81% CrowdHuman, max_size 1600, micro-batch 4 x grad_accum 2 | **0.6282** | 9 | 3,729 | 0.6140 | 15 | complete |

### 3.1 Arm A — 81% CrowdHuman (as trained)

| Epoch | Step | regular mAP@50:95 | EMA mAP@50:95 |
| ---: | ---: | ---: | ---: |
| 0 | 372 | 0.5738 | 0.5759 |
| 1 | 745 | 0.5981 | 0.5952 |
| 2 | 1,118 | 0.6082 | 0.6091 |
| 3 | 1,491 | 0.6049 | 0.6146 |
| 4 | 1,864 | 0.6127 | 0.6177 |
| 5 | 2,237 | 0.6202 | 0.6192 |
| 6 | 2,610 | 0.6141 | 0.6197 |
| 7 | 2,983 | 0.6161 | 0.6185 |
| 8 | 3,356 | 0.6149 | 0.6198 |
| 9 | 3,729 | 0.6147 | 0.6194 |
| 10 | 4,102 | 0.6144 | 0.6174 |
| 11 | 4,475 | 0.6127 | 0.6159 |
| 12 | 4,848 | 0.6086 | 0.6150 |
| 13 | 5,221 | 0.6089 | 0.6122 |
| 14 | 5,594 | 0.6112 | 0.6121 |
| 15 | 5,967 | 0.6063 | 0.6106 |
| 16 | 6,340 | 0.6008 | 0.6101 |
| 17 | 6,713 | 0.6055 | 0.6093 |
| 18 | 7,086 | 0.6039 | 0.6073 |
| 19 | 7,459 | 0.6049 | 0.6074 |
| 20 | 7,832 | 0.6008 | 0.6057 |
| 21 | 8,205 | 0.5979 | 0.6041 |
| 22 | 8,578 | 0.6031 | 0.6040 |
| 23 | 8,951 | 0.6028 | 0.6036 |
| 24 | 9,324 | 0.6014 | 0.6020 |
| 25 | 9,697 | 0.5977 | 0.6019 |
| 26 | 10,070 | 0.5991 | 0.6002 |
| 27 | 10,443 | 0.5982 | 0.5999 |
| 28 | 10,816 | 0.5930 | 0.5988 |
| 29 | 11,189 | 0.5985 | 0.5979 |
| 30 | 11,562 | 0.6009 | 0.5979 |
| 31 | 11,935 | 0.5907 | 0.5973 |
| 32 | 12,308 | 0.5880 | 0.5958 |
| 33 | 12,681 | 0.5915 | 0.5959 |
| 34 | 13,054 | 0.5885 | 0.5944 |
| 35 | 13,427 | 0.5922 | 0.5938 |
| 36 | 13,800 | 0.5902 | 0.5944 |
| 37 | 14,173 | 0.5921 | 0.5932 |
| 38 | 14,546 | 0.5889 | 0.5931 |
| 39 | 14,919 | 0.5876 | 0.5921 |
| 40 | 15,292 | 0.5882 | 0.5888 |
| 41 | 15,665 | 0.5876 | 0.5874 |
| 42 | 16,038 | 0.5865 | 0.5863 |
| 43 | 16,411 | 0.5865 | 0.5863 |
| 44 | 16,784 | 0.5849 | 0.5854 |
| 45 | 17,157 | 0.5854 | 0.5852 |
| 46 | 17,530 | 0.5852 | 0.5855 |
| 47 | 17,903 | 0.5844 | 0.5847 |
| 48 | 18,276 | 0.5833 | 0.5843 |
| 49 | 18,649 | 0.5832 | 0.5844 |

### 3.2 Arm B — 48% MOT20, 4x oversampled

| Epoch | Step | regular mAP@50:95 | EMA mAP@50:95 |
| ---: | ---: | ---: | ---: |
| 1 | 1,165 | 0.6147 | 0.6168 |
| 3 | 2,331 | 0.6099 | 0.6143 |
| 5 | 3,497 | 0.6009 | 0.6065 |
| 7 | 4,663 | 0.5994 | 0.5995 |
| 9 | 5,829 | 0.5907 | 0.5937 |
| 11 | 6,995 | 0.5832 | 0.5885 |
| 13 | 8,161 | 0.5846 | 0.5860 |
| 15 | 9,327 | 0.5814 | 0.5844 |
| 17 | 10,493 | 0.5790 | 0.5810 |
| 19 | 11,659 | 0.5800 | 0.5795 |
| 21 | 12,825 | 0.5776 | 0.5782 |
| 23 | 13,991 | 0.5769 | 0.5785 |
| 25 | 15,157 | 0.5734 | 0.5731 |
| 27 | 16,323 | 0.5699 | 0.5703 |
| 29 | 17,489 | 0.5694 | 0.5696 |

### 3.3 Arm C — 100% MOT20, no CrowdHuman

| Epoch | Step | regular mAP@50:95 | EMA mAP@50:95 |
| ---: | ---: | ---: | ---: |
| 4 | 354 | 0.6074 | 0.6111 |
| 9 | 709 | 0.6118 | 0.6139 |
| 14 | 1,064 | 0.6052 | 0.6095 |
| 19 | 1,419 | 0.5993 | 0.6031 |
| 24 | 1,774 | 0.5949 | 0.5971 |
| 29 | 2,129 | 0.5878 | 0.5914 |
| 34 | 2,484 | 0.5854 | 0.5882 |
| 39 | 2,839 | 0.5812 | 0.5827 |
| 44 | 3,194 | 0.5782 | 0.5796 |
| 49 | 3,549 | 0.5741 | 0.5762 |
| 54 | 3,904 | 0.5727 | 0.5737 |
| 59 | 4,259 | 0.5687 | 0.5704 |
| 64 | 4,614 | 0.5672 | 0.5694 |
| 69 | 4,969 | 0.5674 | 0.5671 |
| 74 | 5,324 | 0.5665 | 0.5668 |
| 79 | 5,679 | 0.5650 | 0.5659 |
| 84 | 6,034 | 0.5615 | 0.5617 |
| 89 | 6,389 | 0.5591 | 0.5592 |
| 94 | 6,744 | 0.5588 | 0.5589 |
| 99 | 7,099 | 0.5558 | 0.5579 |

### 3.4 Arm D — 81% CrowdHuman, max_size 1600, micro-batch 4 x grad_accum 2

| Epoch | Step | regular mAP@50:95 | EMA mAP@50:95 |
| ---: | ---: | ---: | ---: |
| 1 | 745 | 0.5962 | 0.6025 |
| 3 | 1,491 | 0.6190 | 0.6234 |
| 5 | 2,237 | 0.6220 | 0.6246 |
| 7 | 2,983 | 0.6235 | 0.6266 |
| 9 | 3,729 | 0.6237 | 0.6282 |
| 11 | 4,475 | 0.6233 | 0.6273 |
| 13 | 5,221 | 0.6243 | 0.6263 |
| 15 | 5,967 | 0.6242 | 0.6258 |
| 17 | 6,713 | 0.6165 | 0.6235 |
| 19 | 7,459 | 0.6217 | 0.6234 |
| 21 | 8,205 | 0.6140 | 0.6193 |
| 23 | 8,951 | 0.6183 | 0.6182 |
| 25 | 9,697 | 0.6165 | 0.6164 |
| 27 | 10,443 | 0.6129 | 0.6142 |
| 29 | 11,189 | 0.6126 | 0.6140 |

## 4. Detection accuracy

### 4.1 One protocol, all variants

`tracking/scripts/analyze_detections.py`: pycocotools against the same `valid`
annotations, MOT20 ignore regions carried as `iscrowd`, `maxDets = 390` for every
variant. Values are read from the accumulator directly, because pycocotools'
`summarize()` computes the headline AP with a hardcoded `maxDets = 100` that is
absent from this list and silently yields −1.

| Metric | `rfdetr2xl-armd-e9-t005` | `rfdetr2xl-armd-e9-t010-nms070` | `rfdetr2xl-e5-t005` | `rfdetr2xl-e5-t010` | `rfdetr2xl-e5-t010-nms070` | `yoloxx20` | `rfdetr2xl-armc-e9-t005` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mAP@50 | 0.9663 | 0.9528 | 0.9585 | 0.9585 | 0.9510 | 0.9000 | 0.9572 |
| mAP@75 | 0.7716 | 0.7603 | 0.7595 | 0.7549 | 0.7483 | 0.8278 | 0.7423 |
| mAP@50:95 | 0.6446 | 0.6351 | 0.6369 | 0.6337 | 0.6297 | 0.6759 | 0.6258 |
| AR@50:95 | 0.7018 | 0.6892 | 0.6957 | 0.6883 | 0.6811 | 0.7104 | 0.6919 |
| mAP small | 0.1726 | 0.1698 | 0.1651 | 0.1628 | 0.1626 | 0.3074 | 0.1549 |
| mAP medium | 0.6030 | 0.5963 | 0.5983 | 0.5948 | 0.5895 | 0.6536 | 0.5901 |
| mAP large | 0.7170 | 0.7110 | 0.7058 | 0.7043 | 0.6995 | 0.7216 | 0.6943 |
| boxes total | 1,261,854 | 873,131 | 1,265,422 | 949,595 | 865,842 | 580,369 | 1,328,762 |
| boxes/frame mean | 282.74 | 195.64 | 283.54 | 212.77 | 194.00 | 130.04 | 297.73 |
| boxes/frame max | 383 | 326 | 386 | 355 | 331 | 213 | 390 |
| dup pairs/frame @IoU≥0.75 | 31.400 | 0.000 | 33.219 | 13.764 | 0.000 | 0.000 | 48.984 |
| frames with duplicates | 4,463 | 0 | 4,463 | 4,450 | 0 | 0 | 4,463 |
| box height median (px) | 128.0 | 132.7 | 127.3 | 132.3 | 132.7 | 135.5 | 128.7 |

Survivors at a score threshold:

| Threshold | `rfdetr2xl-armd-e9-t005` | `rfdetr2xl-armd-e9-t010-nms070` | `rfdetr2xl-e5-t005` | `rfdetr2xl-e5-t010` | `rfdetr2xl-e5-t010-nms070` | `yoloxx20` | `rfdetr2xl-armc-e9-t005` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ≥ 0.10 | 958,353 | 873,131 | 949,595 | 949,595 | 865,842 | 580,369 | 972,608 |
| ≥ 0.40 | 644,154 | 639,978 | 638,477 | 638,477 | 634,487 | 553,800 | 642,078 |
| ≥ 0.50 | 624,672 | 621,907 | 617,411 | 617,411 | 614,858 | 548,609 | 621,741 |
| ≥ 0.60 | 604,965 | 603,045 | 594,635 | 594,635 | 592,964 | 542,704 | 601,190 |

**The baseline detector has the higher mAP@50:95.** RF-DETR leads only at IoU 0.5.

### 4.2 RF-DETR's own validation metrics, epoch-5 checkpoint

Produced by the library's own ignore-aware validation loop. Reproducing the
recorded 0.6202 is the gate that proved the export used training geometry.

| Metric | Value |
| --- | ---: |
| `val/mAP_50` | 0.9345 |
| `val/mAP_75` | 0.7400 |
| `val/mAP_50_95` | 0.6209 |
| `val/mAR` | 0.6987 |
| `val/precision` | 0.8922 |
| `val/recall` | 0.9134 |
| `val/F1` | 0.9027 |
| `val/cardinality_error` | 7.8443 |
| `val/loss_bbox` | 0.0171 |
| `val/loss_giou` | 0.1949 |
| verdict against expected 0.6202 | **MATCH** |

### 4.3 Inference-time resolution probes

Raising RF-DETR's long-side cap at inference only. Recall is flat; box
regression degrades sharply. The model regresses boxes well only at its
trained geometry, so resolution is a training-time fix.

| Metric | cap 1333 (trained) | cap 1600 | cap 1920 |
| --- | ---: | ---: | ---: |
| `val/mAP_50` | 0.9345 | 0.9292 | 0.9305 |
| `val/mAP_75` | 0.7400 | 0.5943 | 0.6890 |
| `val/mAP_50_95` | 0.6209 | 0.5503 | 0.5874 |
| `val/mAR` | 0.6987 | 0.6371 | 0.6675 |
| `val/precision` | 0.8922 | 0.8908 | 0.8911 |
| `val/recall` | 0.9134 | 0.9138 | 0.9167 |
| `val/loss_bbox` | 0.0171 | 0.0185 | 0.0193 |
| `val/loss_giou` | 0.1949 | 0.2253 | 0.2170 |

Effective input geometry at 1920×1080: cap 1333 → 1333×750 (scale 0.694);
cap 1600 → 1600×900 (0.833); cap 1920 → 1920×1080 (1.000). The baseline
YOLOX-X runs at `test_size = (896, 1600)` → 1593×896, scale 0.830.

## 5. Localization quality

`tracking/scripts/analyze_localization.py`: Hungarian match against `gt.txt` at
IoU 0.5, detections filtered at score ≥ 0.4
(the tracker's `det_thresh`). `det:` rows are raw detections, `trk:` rows are
tracker output, so the pair isolates what the tracker does to box quality.

| Metric | `det:rfdetr2xl-armd-e9-t005` | `det:rfdetr2xl-armd-e9-t010-nms070` | `det:rfdetr2xl-e5-t005` | `det:rfdetr2xl-e5-t010-nms070` | `det:yoloxx20` | `trk:rfdetr2xl-armd-e9-t010-nms070__osnet-ain-msdc__btpp-default` | `trk:rfdetr2xl-e5-t005__osnet-ain-msdc__btpp-default` | `trk:rfdetr2xl-e5-t010-nms070__osnet-ain-msdc__btpp-default` | `trk:yoloxx20__osnet-ain-msdc__btpp-default` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| boxes | 644,154 | 639,978 | 638,477 | 634,487 | 553,800 | 632,525 | 627,099 | 623,257 | 538,112 |
| GT boxes | 615,137 | 615,137 | 615,137 | 615,137 | 615,137 | 615,137 | 615,137 | 615,137 | 615,137 |
| matched | 571,793 | 569,751 | 565,953 | 564,045 | 546,012 | 564,352 | 558,775 | 556,559 | 531,193 |
| recall @IoU 0.5 | 0.9295 | 0.9262 | 0.9200 | 0.9169 | 0.8876 | 0.9174 | 0.9084 | 0.9048 | 0.8635 |
| precision @IoU 0.5 | 0.8877 | 0.8903 | 0.8864 | 0.8890 | 0.9859 | 0.8922 | 0.8910 | 0.8930 | 0.9871 |
| mean matched IoU | 0.8354 | 0.8356 | 0.8334 | 0.8336 | 0.8713 | 0.8362 | 0.8343 | 0.8345 | 0.8726 |
| median matched IoU | 0.8490 | 0.8492 | 0.8473 | 0.8475 | 0.8845 | 0.8494 | 0.8480 | 0.8481 | 0.8855 |
| fraction IoU ≥ 0.50 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| fraction IoU ≥ 0.70 | 0.9323 | 0.9327 | 0.9284 | 0.9290 | 0.9740 | 0.9348 | 0.9315 | 0.9324 | 0.9756 |
| fraction IoU ≥ 0.75 | 0.8587 | 0.8593 | 0.8518 | 0.8524 | 0.9421 | 0.8623 | 0.8557 | 0.8567 | 0.9450 |
| fraction IoU ≥ 0.80 | 0.7197 | 0.7204 | 0.7105 | 0.7111 | 0.8659 | 0.7234 | 0.7143 | 0.7152 | 0.8709 |
| fraction IoU ≥ 0.90 | 0.2209 | 0.2213 | 0.2139 | 0.2143 | 0.3926 | 0.2220 | 0.2152 | 0.2158 | 0.3999 |
| edge residual mean x1 | -0.0064 | -0.0064 | -0.0047 | -0.0046 | -0.0006 | -0.0068 | -0.0051 | -0.0050 | -0.0010 |
| edge residual mean x2 | 0.0032 | 0.0031 | 0.0087 | 0.0087 | -0.0019 | 0.0032 | 0.0089 | 0.0088 | -0.0018 |
| edge residual mean y1 | -0.0055 | -0.0055 | -0.0052 | -0.0052 | -0.0005 | -0.0058 | -0.0055 | -0.0055 | -0.0008 |
| edge residual mean y2 | 0.0030 | 0.0030 | 0.0012 | 0.0012 | 0.0010 | 0.0030 | 0.0013 | 0.0013 | 0.0011 |
| edge residual |mean| x1 | 0.0604 | 0.0603 | 0.0604 | 0.0602 | 0.0468 | 0.0605 | 0.0605 | 0.0603 | 0.0467 |
| edge residual |mean| x2 | 0.0612 | 0.0611 | 0.0630 | 0.0629 | 0.0473 | 0.0610 | 0.0628 | 0.0627 | 0.0469 |
| edge residual |mean| y1 | 0.0225 | 0.0225 | 0.0234 | 0.0234 | 0.0184 | 0.0225 | 0.0233 | 0.0233 | 0.0183 |
| edge residual |mean| y2 | 0.0446 | 0.0445 | 0.0449 | 0.0449 | 0.0308 | 0.0436 | 0.0439 | 0.0438 | 0.0297 |
| size ratio median height | 1.0050 | 1.0050 | 1.0029 | 1.0029 | 0.9990 | 1.0056 | 1.0035 | 1.0035 | 1.0000 |
| size ratio median width | 1.0063 | 1.0062 | 1.0094 | 1.0093 | 0.9954 | 1.0068 | 1.0101 | 1.0100 | 0.9960 |

Localization is unchanged across the tracker, so no post-detection stage
degrades box quality. The error is symmetric jitter, not a correctable offset:
edge residual means are near zero while their absolute values are 22–46% larger
than the baseline's.

## 6. Per-stage tracker instrumentation

`tracking/scripts/diagnose_stages.py`, all 4,463 frames. Hooks record and delegate,
so tracker behaviour is unchanged. Values are per-frame means.

| Stage | rfdetr-dt040 | yoloxx20-dt040 |
| --- | ---: | ---: |
| raw | 283.536 | 130.040 |
| raw_above_thresh | 143.060 | 124.087 |
| promoted | 5.133 | 1.575 |
| entering | 148.193 | 125.662 |
| live_tracks | 199.635 | 152.312 |
| matched | 147.329 | 125.193 |
| new | 0.864 | 0.469 |
| unmatched_tracks | 52.306 | 27.119 |
| output | 140.511 | 120.572 |

Tracker output volume against ground truth, whole split:

| Sequence | GT boxes | RF-DETR output | YOLOX output |
| --- | ---: | ---: | ---: |
| MOT20-01 | 10,810 | 10,200 (0.944×) | 8,669 (0.802×) |
| MOT20-02 | 91,855 | 84,784 (0.923×) | 72,060 (0.784×) |
| MOT20-03 | 193,410 | 193,286 (0.999×) | 174,970 (0.905×) |
| MOT20-05 | 319,062 | 338,829 (1.062×) | 282,413 (0.885×) |
| **total** | **615,137** | **627,099 (1.019×)** | **538,112 (0.875×)** |

Single-frame example, MOT20-02 frame 100 (GT 62 pedestrians):

| Stage | RF-DETR | YOLOX |
| --- | ---: | ---: |
| raw detections | 149 | 56 |
| already above `det_thresh` | 61 | 53 |
| promoted by boosting | 3 | 0 |
| entering association | 64 | 53 |
| live tracks | 88 | 70 |
| matched | 64 | 53 |
| new tracks | 0 | 0 |
| unmatched tracks | 24 | 17 |
| **output boxes** | **61** | **52** |

## 7. Tracking results, MOT20 `val_half`

Vendored TrackEval against `repos/BoostTrack/results/gt/MOT20-val`.
Ground truth: 615,137 boxes, 1,418 identities.

Detector slug legend:

| Slug | Meaning |
| --- | --- |
| `yoloxx20` | ByteTrack's published YOLOX-X MOT20 detector, `test_size=(896,1600)`, `nmsthre=0.7`, `test_conf=0.001` |
| `rfdetr2xl-e5-t005` | RF-DETR 2XL arm-A epoch-5 checkpoint, export score threshold 0.05 |
| `rfdetr2xl-e5-t010` | same checkpoint, export score threshold 0.10 |
| `rfdetr2xl-e5-t010-nms070` | threshold 0.10 then greedy NMS at IoU 0.70 |
| `rfdetr2xl-e5-t010-nms0600` / `-nms0500` | the same at IoU 0.60 / 0.50 |

Unless a row says otherwise the tracker is BoostTrack++ at its published
MOT20 settings with `det_thresh = 0.40`.

### 7.1 OSNet ReID (`osnet-ain-msdc`), held out

Generic domain-generalized ReID. These rows are held out: the model has never seen MOT20.

| Run | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDSW | IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rfdetr2xl-armd-e9-t010-nms070` — `max_age=70` | 70.875 | 72.970 | 68.974 | 85.322 | 87.965 | 85.795 | 1,258 | 2,290 |
| `rfdetr2xl-armd-e9-t010-nms070` — `lambda_iou=0.65` | 70.814 | 72.990 | 68.836 | 85.321 | 87.976 | 85.665 | 1,254 | 2,313 |
| `rfdetr2xl-armd-e9-t010-nms070` — defaults | 70.777 | 72.979 | 68.775 | 85.321 | 87.976 | 85.637 | 1,262 | 2,312 |
| `rfdetr2xl-armd-e9-t010-nms070` — `lambda_shape=0.35` | 70.758 | 72.983 | 68.734 | 85.321 | 87.973 | 85.632 | 1,257 | 2,308 |
| `rfdetr2xl-armd-e9-t010-nms070` — `iou_threshold=0.35` | 70.717 | 72.985 | 68.654 | 85.325 | 87.973 | 85.499 | 1,273 | 2,367 |
| `rfdetr2xl-armd-e9-t010-nms070` — `lambda_iou=0.35`, `lambda_shape=0.35` | 70.715 | 73.000 | 68.636 | 85.323 | 87.974 | 85.505 | 1,262 | 2,306 |
| `rfdetr2xl-armd-e9-t010-nms070` — `iou_threshold=0.25` | 70.713 | 72.953 | 68.678 | 85.314 | 87.981 | 85.655 | 1,241 | 2,270 |
| `rfdetr2xl-armd-e9-t010-nms070` — `lambda_iou=0.35` | 70.695 | 72.991 | 68.606 | 85.325 | 87.960 | 85.517 | 1,284 | 2,312 |
| `rfdetr2xl-armd-e9-t010-nms070` — `lambda_iou=0.25` | 70.689 | 72.974 | 68.610 | 85.324 | 87.946 | 85.529 | 1,296 | 2,300 |
| `rfdetr2xl-armd-e9-t010-nms070` — `iou_threshold=0.25`, `lambda_iou=0.35` | 70.682 | 72.945 | 68.625 | 85.316 | 87.958 | 85.559 | 1,268 | 2,263 |
| `rfdetr2xl-armd-e9-t010-nms070` — `lambda_iou=0.2` | 70.663 | 72.988 | 68.547 | 85.329 | 87.941 | 85.425 | 1,305 | 2,303 |
| `rfdetr2xl-armd-e9-t010-nms070` — `min_hits=4` | 70.659 | 72.792 | 68.720 | 85.367 | 87.732 | 85.620 | 1,176 | 2,185 |
| `rfdetr2xl-armd-e9-t010-nms070` — `iou_threshold=0.25`, `lambda_iou=0.35`, `lambda_shape=0.35` | 70.652 | 72.941 | 68.570 | 85.312 | 87.972 | 85.474 | 1,253 | 2,266 |
| `rfdetr2xl-armd-e9-t010-nms070` — `max_age=30` | 70.627 | 72.968 | 68.497 | 85.324 | 87.952 | 85.263 | 1,295 | 2,374 |
| `rfdetr2xl-armd-e9-t010-nms070` — `lambda_mhd=0.35` | 70.592 | 72.931 | 68.463 | 85.317 | 87.946 | 85.412 | 1,292 | 2,313 |
| `rfdetr2xl-armd-e9-t010-nms070` — `min_hits=5` | 70.520 | 72.576 | 68.651 | 85.407 | 87.461 | 85.574 | 1,078 | 2,084 |
| `rfdetr2xl-armd-e9-t010-nms070` — `iou_threshold=0.25`, `lambda_iou=0.35`, `lambda_shape=0.35`, `max_age=30` | 70.500 | 72.905 | 68.310 | 85.310 | 87.967 | 85.291 | 1,279 | 2,323 |
| `yoloxx20` — defaults | 70.208 | 73.496 | 67.133 | 88.177 | 85.031 | 82.536 | 1,013 | 1,762 |
| `rfdetr2xl-e5-t010-nms070` — defaults | 68.809 | 71.928 | 65.965 | 85.190 | 86.883 | 83.255 | 1,571 | 2,309 |
| `rfdetr2xl-e5-t010-nms0600` — defaults | 68.510 | 70.964 | 66.272 | 85.212 | 85.749 | 83.388 | 1,427 | 2,162 |
| `rfdetr2xl-e5-t005` — `det_thresh=0.50` | 68.353 | 71.683 | 65.311 | 85.321 | 86.457 | 82.793 | 1,615 | 2,208 |
| `rfdetr2xl-e5-t010` — defaults | 68.175 | 72.068 | 64.640 | 85.137 | 87.012 | 81.874 | 2,065 | 2,624 |
| `rfdetr2xl-e5-t005` — defaults | 68.080 | 71.996 | 64.526 | 85.128 | 87.007 | 81.805 | 2,113 | 2,662 |
| `rfdetr2xl-e5-t005` — `det_thresh=0.60` | 68.018 | 70.619 | 65.635 | 85.462 | 85.176 | 83.117 | 1,323 | 1,970 |
| `rfdetr2xl-e5-t010-nms0500` — defaults | 66.076 | 68.483 | 63.883 | 85.271 | 82.654 | 81.495 | 1,297 | 2,037 |

<details><summary>Secondary metrics</summary>

| Run | DetPr | DetRe | AssPr | AssRe | MOTP | Frag | MT | ML | CLR_TP | CLR_FP | CLR_FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rfdetr2xl-armd-e9-t010-nms070` — `max_age=70` | 82.288 | 78.257 | 81.334 | 73.631 | 83.546 | 3,687 | 1,169 | 51 | 563,688 | 21,322 | 51,449 |
| `rfdetr2xl-armd-e9-t010-nms070` — `lambda_iou=0.65` | 82.302 | 78.266 | 81.430 | 73.449 | 83.545 | 3,686 | 1,171 | 51 | 563,697 | 21,270 | 51,440 |
| `rfdetr2xl-armd-e9-t010-nms070` — defaults | 82.300 | 78.256 | 81.295 | 73.428 | 83.545 | 3,704 | 1,171 | 51 | 563,675 | 21,242 | 51,462 |
| `rfdetr2xl-armd-e9-t010-nms070` — `lambda_shape=0.35` | 82.301 | 78.260 | 81.283 | 73.412 | 83.548 | 3,693 | 1,171 | 51 | 563,671 | 21,261 | 51,466 |
| `rfdetr2xl-armd-e9-t010-nms070` — `iou_threshold=0.35` | 82.313 | 78.252 | 81.388 | 73.227 | 83.547 | 3,714 | 1,172 | 51 | 563,606 | 21,178 | 51,531 |
| `rfdetr2xl-armd-e9-t010-nms070` — `lambda_iou=0.35`, `lambda_shape=0.35` | 82.315 | 78.267 | 81.181 | 73.357 | 83.547 | 3,700 | 1,172 | 51 | 563,654 | 21,231 | 51,483 |
| `rfdetr2xl-armd-e9-t010-nms070` — `iou_threshold=0.25` | 82.285 | 78.238 | 81.102 | 73.364 | 83.546 | 3,699 | 1,170 | 52 | 563,664 | 21,220 | 51,473 |
| `rfdetr2xl-armd-e9-t010-nms070` — `lambda_iou=0.35` | 82.316 | 78.257 | 81.192 | 73.297 | 83.548 | 3,731 | 1,172 | 51 | 563,580 | 21,220 | 51,557 |
| `rfdetr2xl-armd-e9-t010-nms070` — `lambda_iou=0.25` | 82.315 | 78.238 | 81.273 | 73.271 | 83.550 | 3,749 | 1,172 | 51 | 563,478 | 21,193 | 51,659 |
| `rfdetr2xl-armd-e9-t010-nms070` — `iou_threshold=0.25`, `lambda_iou=0.35` | 82.284 | 78.231 | 80.997 | 73.388 | 83.543 | 3,723 | 1,172 | 52 | 563,583 | 21,255 | 51,554 |
| `rfdetr2xl-armd-e9-t010-nms070` — `lambda_iou=0.2` | 82.327 | 78.245 | 81.302 | 73.192 | 83.551 | 3,764 | 1,169 | 51 | 563,447 | 21,186 | 51,690 |
| `rfdetr2xl-armd-e9-t010-nms070` — `min_hits=4` | 82.634 | 77.773 | 81.512 | 73.244 | 83.599 | 3,368 | 1,151 | 57 | 559,897 | 19,048 | 55,240 |
| `rfdetr2xl-armd-e9-t010-nms070` — `iou_threshold=0.25`, `lambda_iou=0.35`, `lambda_shape=0.35` | 82.277 | 78.232 | 80.946 | 73.375 | 83.543 | 3,706 | 1,171 | 52 | 563,648 | 21,245 | 51,489 |
| `rfdetr2xl-armd-e9-t010-nms070` — `max_age=30` | 82.307 | 78.239 | 81.421 | 73.086 | 83.548 | 3,711 | 1,172 | 51 | 563,530 | 21,207 | 51,607 |
| `rfdetr2xl-armd-e9-t010-nms070` — `lambda_mhd=0.35` | 82.284 | 78.215 | 81.064 | 73.201 | 83.549 | 3,737 | 1,174 | 51 | 563,498 | 21,219 | 51,639 |
| `rfdetr2xl-armd-e9-t010-nms070` — `min_hits=5` | 82.893 | 77.327 | 81.691 | 73.069 | 83.644 | 3,114 | 1,129 | 61 | 556,459 | 17,373 | 58,678 |
| `rfdetr2xl-armd-e9-t010-nms070` — `iou_threshold=0.25`, `lambda_iou=0.35`, `lambda_shape=0.35`, `max_age=30` | 82.275 | 78.194 | 81.103 | 73.018 | 83.548 | 3,701 | 1,173 | 52 | 563,512 | 21,117 | 51,625 |
| `yoloxx20` — defaults | 87.688 | 76.687 | 83.337 | 71.174 | 87.084 | 5,243 | 1,021 | 47 | 531,015 | 6,947 | 84,122 |
| `rfdetr2xl-e5-t010-nms070` — defaults | 82.300 | 76.952 | 79.746 | 70.981 | 83.357 | 4,289 | 1,139 | 57 | 555,593 | 19,570 | 59,544 |
| `rfdetr2xl-e5-t010-nms0600` — defaults | 82.484 | 75.728 | 80.125 | 71.121 | 83.427 | 4,727 | 1,096 | 58 | 546,826 | 17,928 | 68,311 |
| `rfdetr2xl-e5-t005` — `det_thresh=0.50` | 83.203 | 75.982 | 80.923 | 69.624 | 83.520 | 4,612 | 1,093 | 70 | 547,597 | 14,154 | 67,540 |
| `rfdetr2xl-e5-t010` — defaults | 82.105 | 77.256 | 80.130 | 69.321 | 83.321 | 4,482 | 1,149 | 56 | 558,058 | 20,750 | 57,079 |
| `rfdetr2xl-e5-t005` — defaults | 82.104 | 77.175 | 80.341 | 69.086 | 83.334 | 4,533 | 1,148 | 57 | 557,766 | 20,440 | 57,371 |
| `rfdetr2xl-e5-t005` — `det_thresh=0.60` | 83.851 | 74.370 | 81.317 | 69.655 | 83.705 | 4,742 | 1,017 | 84 | 535,430 | 10,156 | 79,707 |
| `rfdetr2xl-e5-t010-nms0500` — defaults | 82.732 | 72.795 | 80.161 | 68.473 | 83.528 | 5,934 | 1,004 | 64 | 525,492 | 15,759 | 89,645 |

</details>

### 7.2 FastReID (`fastreid-sbs-s50-mot20`), NOT held out

BoostTrack++'s published ReID. MOT20-trained, so it has seen `val_half` identities; absolute values are inflated. The association sweep points live here.

| Run | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDSW | IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yoloxx20` — `lambda_iou=0.35` | 70.734 | 73.587 | 68.059 | 88.210 | 84.963 | 83.165 | 970 | 1,800 |
| `yoloxx20` — defaults | 70.547 | 73.610 | 67.677 | 88.210 | 84.973 | 82.877 | 989 | 1,821 |
| `yoloxx20` — `iou_threshold=0.25`, `lambda_iou=0.35`, `lambda_shape=0.35` | 70.417 | 73.572 | 67.463 | 88.202 | 84.977 | 82.746 | 1,009 | 1,800 |
| `rfdetr2xl-e5-t010-nms070` — `iou_threshold=0.25`, `lambda_iou=0.35`, `lambda_shape=0.35` | 69.718 | 71.999 | 67.642 | 85.206 | 86.828 | 84.510 | 1,628 | 2,461 |
| `rfdetr2xl-e5-t010-nms070` — `iou_threshold=0.25`, `lambda_iou=0.35`, `lambda_shape=0.35`, `max_age=30` | 69.716 | 71.974 | 67.664 | 85.206 | 86.815 | 84.571 | 1,660 | 2,507 |
| `rfdetr2xl-e5-t010-nms070` — `lambda_iou=0.35` | 69.695 | 72.030 | 67.570 | 85.220 | 86.805 | 84.409 | 1,680 | 2,520 |
| `rfdetr2xl-e5-t010-nms070` — `iou_threshold=0.25`, `lambda_iou=0.35` | 69.673 | 71.993 | 67.561 | 85.208 | 86.837 | 84.484 | 1,642 | 2,456 |
| `rfdetr2xl-e5-t010-nms070` — `iou_threshold=0.25` | 69.658 | 71.987 | 67.539 | 85.205 | 86.825 | 84.509 | 1,626 | 2,458 |
| `rfdetr2xl-e5-t010-nms070` — `lambda_shape=0.35` | 69.622 | 72.029 | 67.432 | 85.211 | 86.816 | 84.270 | 1,676 | 2,556 |
| `rfdetr2xl-e5-t010-nms070` — `max_age=30` | 69.604 | 72.005 | 67.419 | 85.215 | 86.817 | 84.226 | 1,636 | 2,595 |
| `rfdetr2xl-e5-t010-nms070` — `lambda_mhd=0.35` | 69.575 | 72.024 | 67.345 | 85.220 | 86.774 | 84.241 | 1,691 | 2,551 |
| `rfdetr2xl-e5-t010-nms070` — `lambda_iou=0.65` | 69.571 | 72.002 | 67.358 | 85.207 | 86.829 | 84.225 | 1,663 | 2,558 |
| `rfdetr2xl-e5-t010-nms070` — `max_age=70` | 69.542 | 71.969 | 67.333 | 85.204 | 86.834 | 84.272 | 1,682 | 2,512 |
| `rfdetr2xl-e5-t010-nms070` — `lambda_iou=0.35`, `lambda_shape=0.35` | 69.531 | 72.005 | 67.278 | 85.214 | 86.807 | 84.112 | 1,671 | 2,540 |
| `rfdetr2xl-e5-t010-nms070` — `lambda_iou=0.25` | 69.518 | 72.000 | 67.255 | 85.220 | 86.771 | 84.181 | 1,711 | 2,521 |
| `rfdetr2xl-e5-t010-nms070` — `iou_threshold=0.35` | 69.513 | 71.993 | 67.256 | 85.218 | 86.793 | 84.152 | 1,700 | 2,613 |
| `rfdetr2xl-e5-t010-nms070` — defaults | 69.509 | 71.970 | 67.269 | 85.207 | 86.818 | 84.156 | 1,670 | 2,545 |
| `rfdetr2xl-e5-t010-nms070` — `dlo_boost_coef=0.7` | 69.509 | 71.970 | 67.269 | 85.207 | 86.818 | 84.156 | 1,670 | 2,545 |
| `rfdetr2xl-e5-t010-nms070` — `dlo_boost_coef=0.3` | 69.509 | 71.970 | 67.269 | 85.207 | 86.818 | 84.156 | 1,670 | 2,545 |
| `rfdetr2xl-e5-t010-nms070` — `lambda_iou=0.2` | 69.495 | 72.009 | 67.204 | 85.222 | 86.773 | 84.122 | 1,690 | 2,519 |
| `rfdetr2xl-e5-t010-nms070` — `min_hits=4` | 69.351 | 71.688 | 67.222 | 85.259 | 86.462 | 84.135 | 1,488 | 2,411 |
| `rfdetr2xl-e5-t010-nms070` — `min_hits=5` | 69.172 | 71.386 | 67.157 | 85.302 | 86.085 | 84.079 | 1,345 | 2,315 |

<details><summary>Secondary metrics</summary>

| Run | DetPr | DetRe | AssPr | AssRe | MOTP | Frag | MT | ML | CLR_TP | CLR_FP | CLR_FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yoloxx20` — `lambda_iou=0.35` | 87.797 | 76.709 | 84.609 | 71.825 | 87.106 | 5,335 | 1,014 | 47 | 530,530 | 6,924 | 84,607 |
| `yoloxx20` — defaults | 87.802 | 76.729 | 84.359 | 71.469 | 87.098 | 5,310 | 1,014 | 47 | 530,629 | 6,937 | 84,508 |
| `yoloxx20` — `iou_threshold=0.25`, `lambda_iou=0.35`, `lambda_shape=0.35` | 87.771 | 76.710 | 84.022 | 71.416 | 87.095 | 5,322 | 1,013 | 47 | 530,674 | 6,943 | 84,463 |
| `rfdetr2xl-e5-t010-nms070` — `iou_threshold=0.25`, `lambda_iou=0.35`, `lambda_shape=0.35` | 82.351 | 76.992 | 81.689 | 72.009 | 83.373 | 4,404 | 1,141 | 59 | 555,424 | 19,687 | 59,713 |
| `rfdetr2xl-e5-t010-nms070` — `iou_threshold=0.25`, `lambda_iou=0.35`, `lambda_shape=0.35`, `max_age=30` | 82.354 | 76.960 | 81.640 | 72.061 | 83.365 | 4,408 | 1,143 | 59 | 555,267 | 19,577 | 59,870 |
| `rfdetr2xl-e5-t010-nms070` — `lambda_iou=0.35` | 82.384 | 77.002 | 81.856 | 71.883 | 83.373 | 4,440 | 1,140 | 58 | 555,302 | 19,655 | 59,835 |
| `rfdetr2xl-e5-t010-nms070` — `iou_threshold=0.25`, `lambda_iou=0.35` | 82.356 | 76.982 | 81.459 | 72.008 | 83.377 | 4,439 | 1,141 | 58 | 555,404 | 19,595 | 59,733 |
| `rfdetr2xl-e5-t010-nms070` — `iou_threshold=0.25` | 82.339 | 76.989 | 81.522 | 71.944 | 83.372 | 4,392 | 1,142 | 60 | 555,445 | 19,724 | 59,692 |
| `rfdetr2xl-e5-t010-nms070` — `lambda_shape=0.35` | 82.367 | 77.011 | 81.918 | 71.711 | 83.361 | 4,405 | 1,141 | 59 | 555,427 | 19,712 | 59,710 |
| `rfdetr2xl-e5-t010-nms070` — `max_age=30` | 82.381 | 76.977 | 82.041 | 71.650 | 83.376 | 4,394 | 1,141 | 59 | 555,234 | 19,554 | 59,903 |
| `rfdetr2xl-e5-t010-nms070` — `lambda_mhd=0.35` | 82.393 | 76.987 | 81.793 | 71.654 | 83.369 | 4,472 | 1,144 | 58 | 555,125 | 19,655 | 60,012 |
| `rfdetr2xl-e5-t010-nms070` — `lambda_iou=0.65` | 82.344 | 77.001 | 82.028 | 71.574 | 83.365 | 4,383 | 1,140 | 58 | 555,503 | 19,722 | 59,634 |
| `rfdetr2xl-e5-t010-nms070` — `max_age=70` | 82.332 | 76.974 | 81.655 | 71.655 | 83.367 | 4,387 | 1,146 | 58 | 555,469 | 19,638 | 59,668 |
| `rfdetr2xl-e5-t010-nms070` — `lambda_iou=0.35`, `lambda_shape=0.35` | 82.364 | 76.990 | 81.775 | 71.602 | 83.365 | 4,426 | 1,143 | 58 | 555,327 | 19,676 | 59,810 |
| `rfdetr2xl-e5-t010-nms070` — `lambda_iou=0.25` | 82.379 | 76.973 | 81.566 | 71.658 | 83.373 | 4,466 | 1,142 | 58 | 555,121 | 19,649 | 60,016 |
| `rfdetr2xl-e5-t010-nms070` — `iou_threshold=0.35` | 82.361 | 76.982 | 81.977 | 71.436 | 83.373 | 4,432 | 1,143 | 57 | 555,280 | 19,682 | 59,857 |
| `rfdetr2xl-e5-t010-nms070` — defaults | 82.336 | 76.973 | 81.780 | 71.582 | 83.370 | 4,410 | 1,142 | 59 | 555,394 | 19,672 | 59,743 |
| `rfdetr2xl-e5-t010-nms070` — `dlo_boost_coef=0.7` | 82.336 | 76.973 | 81.780 | 71.582 | 83.370 | 4,410 | 1,142 | 59 | 555,394 | 19,672 | 59,743 |
| `rfdetr2xl-e5-t010-nms070` — `dlo_boost_coef=0.3` | 82.336 | 76.973 | 81.780 | 71.582 | 83.370 | 4,410 | 1,142 | 59 | 555,394 | 19,672 | 59,743 |
| `rfdetr2xl-e5-t010-nms070` — `lambda_iou=0.2` | 82.387 | 76.976 | 81.549 | 71.592 | 83.373 | 4,465 | 1,141 | 58 | 555,102 | 19,639 | 60,035 |
| `rfdetr2xl-e5-t010-nms070` — `min_hits=4` | 82.660 | 76.407 | 81.993 | 71.433 | 83.436 | 3,963 | 1,124 | 64 | 550,974 | 17,629 | 64,163 |
| `rfdetr2xl-e5-t010-nms070` — `min_hits=5` | 82.902 | 75.889 | 82.163 | 71.285 | 83.501 | 3,617 | 1,095 | 66 | 546,994 | 16,108 | 68,143 |

</details>

## 8. Provenance

| Artifact | Path |
| --- | --- |
| Detection scores | `artifacts/tracking/detection-analysis-val_half.json`, `-armc-`, `-i4-` |
| Localization | `artifacts/tracking/localization-analysis-val_half.json` |
| Label agreement | `artifacts/tracking/annotation-agreement-val_half.json` |
| Geometry gate and probes | `artifacts/tracking/geometry-gate-*.json`, `geometry-probe-*.json` |
| Stage instrumentation | `artifacts/tracking/diagnosis/stages-*.{csv,json}` |
| Detections (L1) | `artifacts/tracking/detections/<detector>/<split>/` |
| Embeddings (L2) | `artifacts/tracking/embeddings/<detector>__<reid>/<split>/` |
| Tracks (L3) | `artifacts/tracking/tracks/<combination>/<split>/` |
| Metrics (L4) | `artifacts/tracking/metrics/<combination>/<split>/` |
| Training runs | `finetuning/artifacts/<run>/metrics.csv`, `run-provenance.json` |

