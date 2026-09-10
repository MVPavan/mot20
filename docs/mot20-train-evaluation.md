# Full MOT20 train evaluation: I4, arm E, YOLOX-X

Produced 2026-09-09 on request. Tracked in `mot-n2n.3`.

Detection mAP and full TrackEval metrics for three detectors over the **entire
MOT20 train split** — MOT20-01/02/03/05 at full length, 8,931 frames — rather
than the `val_half` second half that `docs/results-reference.md` covers.

## 0. Read this before quoting anything here

**Every number on this page is training-set fit, not generalization.** All three
detectors were trained on 100% of MOT20 train:

| Detector | Overlap with this evaluation set |
| --- | --- |
| `rfdetr2xl-i4-e8` | 8,931 / 8,931 frames, verified in the build's COCO manifest (MOT20-01 429, -02 2,782, -03 2,405, -05 3,315) |
| `rfdetr2xl-arme-e19` | same build recipe as I4, same coverage, and it trained 20 epochs on it |
| `yoloxx20-official` | ByteTrack's YOLOX-X MOT20 release, trained on the complete MOT20 train set |

MOT20 test ships no public ground truth, so full train is the only fully
scorable split — but nothing here is held out and no row belongs in the same
table as a `val_half` figure. All work remains `local_test_adapted` per
`docs/MOTPolicy.md`.

Two things still make the comparison worth having. Full-length sequences run to
3,315 frames against `val_half`'s 1,657, so association is stressed roughly
twice as hard. And the three models are contaminated to a broadly comparable
degree, so relative ordering carries information even though absolute values are
inflated. The exception is arm E, which is *more* contaminated than the others by
construction — see section 3.

## 1. The YOLOX baseline had to be replaced, and it is stronger than the old one

The `yoloxx20` detections used throughout `docs/results-reference.md` were
supplied as a prebuilt bundle covering only `val_half` and the test sequences.
No YOLOX weights were on disk, so scoring that baseline on train required
re-running the detector.

ByteTrack's official release (`weights/bytetrack_x_mot20.tar`, sha256
`021d7bc4…de89de64`, 792,835,731 bytes, Google Drive id
`1HX2_JpMOjOIj1Z9rJjoet9XNy_cCAs5U` from `repos/ByteTrack/README.md`) **does not
reproduce the supplied bundle**. On `val_half/MOT20-01`:

| | detections | per frame | max score |
| --- | ---: | ---: | ---: |
| supplied `yoloxx20` | 9,826 | 45.916 | 0.9518 |
| official checkpoint | 13,004 | 60.766 | 0.9810 |

It is the same detector *family* — on frame 1 all 52 supplied boxes match one of
my 64 at IoU ≥ 0.5, 47 at IoU ≥ 0.9, and the top five are the same five people
within 1–2 px — but the weights differ. Ruled out as explanations: input geometry
(896×1600 → top 0.9683, 736×1920 → 0.9683, 800×1440 → 0.9698), precision (fp16
and fp32 agree to four decimals), layer fusion, and all three scoring conventions
(`obj*cls` 0.9683, obj only 0.9995, cls only 0.9697). The top score is stable at
~0.968 in every configuration and no post-processing can lower a model's top
score to the supplied 0.9355. This matches `datasets/README.md`, which sources
those files from a "delivered detector bundle" rather than the official release.

So `yoloxx20-official` is registered as its own variant. Scored on the same
`val_half` frames, it is **substantially stronger** than the bundle the repo has
been treating as the YOLOX baseline:

| Metric | supplied `yoloxx20` | `yoloxx20-official` |
| --- | ---: | ---: |
| mAP@50 | 0.9000 | **0.9683** |
| mAP@75 | 0.8278 | **0.8798** |
| mAP@50:95 | 0.6759 | **0.7135** |
| AR@50:95 | 0.7104 | **0.7553** |
| boxes/frame | 130.04 | 153.57 |

Every RF-DETR-versus-baseline margin in `docs/results-reference.md` was measured
against the weaker model. The two must never be pooled.

`mot-n2n.4` then carried the official checkpoint all the way through tracking on
`val_half`, under the identical BoostTrack++ protocol, and the headline claim
inverts:

| Run (val_half, btpp defaults) | HOTA | DetA | AssA | LocA | MOTA | IDF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yoloxx20-official` / osnet | **76.543** | 79.495 | 73.768 | 87.627 | 92.396 | 88.714 |
| `yoloxx20-official` / fastreid | 76.169 | 79.549 | 73.005 | 87.646 | 92.337 | 87.587 |
| `rfdetr2xl-i4-e8` / osnet | 75.865 | 78.806 | 73.110 | 87.480 | 91.986 | 88.637 |
| `rfdetr2xl-armd-e9-t015` / osnet | 70.883 | 72.976 | 68.983 | 85.321 | 87.979 | 85.838 |
| supplied `yoloxx20` / osnet | 70.208 | 73.496 | 67.133 | 88.177 | 85.031 | 82.536 |

Arm D's 70.883 does not beat the real ByteTrack baseline — it trails it by
**5.66 HOTA**. The contamination-matched I4 build trails it by 0.678. The gain
is in both components, DetA +6.00 and AssA +6.64 over the supplied bundle, so it
is not a post-processing artifact. Note the one metric that moves the other way:
LocA drops 88.177 → 87.627, so the official detector localizes marginally worse
while tracking far better.

## 2. Detection accuracy, full MOT20 train

`tracking/scripts/analyze_detections.py`, pycocotools against
`datasets/finetuning/rfdetr-mot20-fulltrain-eval-2026-09-09`, MOT20 ignore
regions carried as `iscrowd`, `maxDets = 390`. 1,134,614 positive ground-truth
boxes and 126,855 ignore regions. RF-DETR rows are raw `-t005` set predictions
with no NMS; the YOLOX row has already passed its own NMS at 0.7.

| Metric | `rfdetr2xl-i4-e8-t005` | `rfdetr2xl-arme-e19-t005` | `yoloxx20-official` |
| --- | ---: | ---: | ---: |
| mAP@50 | 0.9851 | **0.9895** | 0.9685 |
| mAP@75 | 0.8843 | **0.9292** | 0.8815 |
| **mAP@50:95** | 0.7184 | **0.7653** | 0.7156 |
| AR@50:95 | 0.7656 | **0.8066** | 0.7582 |
| mAP small | 0.3091 | **0.4447** | 0.3079 |
| mAP medium | 0.6908 | **0.7461** | 0.6867 |
| mAP large | 0.7614 | **0.7958** | 0.7630 |
| boxes total | 2,376,773 | 1,829,239 | 1,273,732 |
| boxes/frame mean | 266.13 | 204.82 | 142.62 |
| boxes/frame max | 387 | 362 | 261 |
| dup pairs/frame @IoU≥0.75 | 26.052 | 14.213 | 0.002 |
| frames with duplicates | 8,931 | 8,918 | 13 |
| box height median (px) | 126.5 | 131.8 | 136.5 |

GT-matched redundant predictions per frame (the GT-anchored metric from
`mot-n2n.1`, IoU ≥ 0.5 to ground truth):

| Score threshold | I4 | arm E | YOLOX-official |
| --- | ---: | ---: | ---: |
| ≥ 0.05 | 58.933 | 32.008 | 5.180 |
| ≥ 0.10 | 31.337 | 14.217 | 5.180 |
| ≥ 0.20 | 8.496 | 4.970 | 3.680 |
| ≥ 0.50 | 1.141 | 1.047 | **1.696** |

GT recall at ≥ 0.05: I4 0.9895, arm E 0.9939, YOLOX 0.9756.

Note the crossover at ≥ 0.50: once low-confidence set predictions are gone, the
NMS'd YOLOX output carries *more* redundant boxes per ground-truth person than
either RF-DETR. The duplicate problem is a low-threshold phenomenon, not an
intrinsic property of set prediction at the operating point.

## 3. Tracking, full MOT20 train

BoostTrack++ at published MOT20 settings (`det_thresh = 0.40`), detections at
score ≥ 0.10 then greedy NMS at IoU 0.70 for RF-DETR. Vendored TrackEval against
`repos/BoostTrack/results/gt/MOT20-train`, the full-length official ground truth:
1,134,614 boxes, 2,215 identities.

| Combination | HOTA | DetA | AssA | LocA | MOTA | IDF1 | MOTP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `arme-e19` / osnet | **80.588** | **83.207** | **78.094** | **89.060** | **94.939** | **91.629** | 88.104 |
| `arme-e19` / fastreid | 80.541 | 83.203 | 78.007 | 89.064 | 94.898 | 91.492 | 88.106 |
| `i4-e8` / fastreid | 76.353 | 79.001 | 73.858 | 87.499 | 92.034 | 89.042 | 86.257 |
| `yoloxx20-official` / osnet | 76.311 | 79.783 | 73.050 | 87.668 | 92.647 | 88.142 | 86.428 |
| `i4-e8` / osnet | 75.968 | 78.972 | 73.142 | 87.493 | 92.071 | 88.785 | 86.253 |
| `yoloxx20-official` / fastreid | 75.931 | 79.828 | 72.287 | 87.682 | 92.614 | 87.076 | 86.439 |

Secondary detection and association components:

| Combination | DetRe | DetPr | AssRe | AssPr | IDR | IDP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `arme-e19` / osnet | 86.627 | 89.091 | 82.337 | 86.743 | 90.363 | 92.932 |
| `arme-e19` / fastreid | 86.619 | 89.096 | 81.782 | 87.437 | 90.221 | 92.800 |
| `i4-e8` / fastreid | 83.493 | 86.029 | 78.314 | 84.965 | 87.729 | 90.394 |
| `yoloxx20-official` / osnet | 84.637 | 85.926 | 77.604 | 84.261 | 87.480 | 88.813 |
| `i4-e8` / osnet | 83.477 | 86.010 | 78.211 | 83.668 | 87.478 | 90.132 |
| `yoloxx20-official` / fastreid | 84.636 | 85.983 | 76.334 | 85.194 | 86.394 | 87.769 |

Identity and fragmentation counts:

| Combination | IDSW | MT | PT | ML | Frag | IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `arme-e19` / osnet | 1,500 | 2,034 | 136 | 45 | 4,103 | 2,942 |
| `arme-e19` / fastreid | 1,659 | 2,032 | 138 | 45 | 4,206 | 3,152 |
| `i4-e8` / fastreid | 1,954 | 1,964 | 192 | 59 | 5,057 | 3,640 |
| `yoloxx20-official` / osnet | 2,217 | 2,005 | 154 | 56 | 3,895 | 4,384 |
| `i4-e8` / osnet | 1,845 | 1,962 | 194 | 59 | 4,935 | 3,344 |
| `yoloxx20-official` / fastreid | 2,480 | 2,006 | 154 | 55 | 4,107 | 4,923 |

Raw counts (2,215 ground-truth identities, 1,134,614 ground-truth boxes):

| Combination | CLR_TP | CLR_FP | CLR_FN | Dets |
| --- | ---: | ---: | ---: | ---: |
| `arme-e19` / osnet | 1,090,967 | 12,275 | 43,647 | 1,103,242 |
| `arme-e19` / fastreid | 1,090,733 | 12,345 | 43,881 | 1,103,078 |
| `i4-e8` / fastreid | 1,073,677 | 27,493 | 60,937 | 1,101,170 |
| `yoloxx20-official` / osnet | 1,085,498 | 32,090 | 49,116 | 1,117,588 |
| `i4-e8` / osnet | 1,073,851 | 27,356 | 60,763 | 1,101,207 |
| `yoloxx20-official` / fastreid | 1,085,067 | 31,771 | 49,547 | 1,116,838 |

## 4. What the numbers say

**Arm E's lead is an artifact of the design, not evidence of a better model.**
It tops every metric by a wide margin — 80.588 HOTA against 76.353 and 76.311 —
but arm E was built as a contaminated-validation diagnostic (`mot-yk3.1`) and
trained for 20 epochs while selecting on this very data, against I4's 8. Its
held-out quality is unmeasured and unmeasurable here. Read the gap as how much
extra fitting 12 more epochs on the evaluation frames buys, roughly +4.2 HOTA,
not as a model ranking.

**I4 and the official YOLOX are level.** 76.353 versus 76.311 HOTA is a 0.042
gap; the per-epoch noise floor measured in `mot-6po` is 0.003–0.007 mAP, and the
`val_half` work already showed rank inversions on larger margins than this. Treat
them as tied. The composition differs though: YOLOX wins DetA (79.783 vs 78.972
with OSNet) and RF-DETR wins AssA (73.858 vs 73.050 at their respective bests),
which is the same detection-versus-association trade seen on `val_half`.

**RF-DETR is much more precise, YOLOX recalls more.** At comparable HOTA, I4
produces 27,356–27,493 false positives against YOLOX's 31,771–32,090, while
YOLOX finds ~11,600 more true positives. RF-DETR's advantage shows up as
identity stability: 1,845–1,954 ID switches against YOLOX's 2,217–2,480, and
3,344–3,640 tracks against 4,384–4,923 for the same 2,215 real identities.

**ReID choice barely matters, and it is not consistent.** FastReID helps I4
(+0.385 HOTA) and hurts YOLOX (−0.380) and arm E (−0.047). The spread is well
inside the range where `val_half` rank inversions were already demonstrated.
Given that `fastreid-sbs-s50-mot20` is MOT20-trained and adds a second
contamination axis, the held-out OSNet control is the more honest default.

**Localization remains RF-DETR's weak point, and arm E is the exception.** LocA:
arm E 89.06, YOLOX 87.67, I4 87.49. On `val_half` the gap ran the other way by
~2.9 points in YOLOX's favour. Arm E's 20 epochs of box-regression fitting on
these exact frames is precisely the localization overfitting identified in
`mot-ayx`, seen from the training-set side.

## 5. Provenance

| Artifact | Path |
| --- | --- |
| Evaluation dataset (COCO, full train) | `datasets/finetuning/rfdetr-mot20-fulltrain-eval-2026-09-09` |
| TrackEval ground truth | `repos/BoostTrack/results/gt/MOT20-train` |
| Detection scores | `artifacts/tracking/detection-analysis-train-rfdetr.json`, `-train-yolox.json` |
| val_half YOLOX anchor | `artifacts/tracking/detection-analysis-valhalf-yoloxofficial.json` |
| Detections (L1) | `artifacts/tracking/detections/<detector>/train/` |
| Embeddings (L2) | `artifacts/tracking/embeddings/<detector>__<reid>/train/` |
| Tracks (L3) | `artifacts/tracking/tracks/<combination>/train/` |
| Metrics (L4) | `artifacts/tracking/metrics/<combination>/train/` |
| YOLOX weights | `weights/bytetrack_x_mot20.tar` |
| YOLOX inference | `tracking/scripts/infer_yoloxx20.py` |
| Dataset builder | `finetuning/scripts/build_mot20_full_split_dataset.py` |

Three latent defects were found and fixed while producing this page; all are
recorded on `mot-n2n.3`. In summary: BoostTrack's ECC warp cache was keyed by
sequence name only and would have applied `val_half` warps to train frames and
then overwritten the cache; `video_name` doubles as the `max_age` lookup key and
silently falls back to 30 for an unregistered name; and `export_embeddings.py`
took the ReID model from `--test-dataset` while naming the output directory from
`--reid`, so the two could disagree without complaint. No previously published
number is affected by any of them — an audit of all 25 embedding manifests found
0 mislabelled.
