# BoostTrack++ Parameter Inventory

Complete inventory of every value that changes BoostTrack++ behaviour, including
the two postprocessing passes. Written to answer one question: **what can we
actually turn?**

**Scope.** This document is authoritative for *what exists, where it acts, and
whether it is reachable*. It is not where results live. Measured tracking numbers
are in [`docs/results-reference.md`](results-reference.md), interpreted in
[`docs/experiment-report.md`](experiment-report.md). Integration contracts and
artifact naming are in [`docs/tracker-experiments.md`](tracker-experiments.md).

**Source.** `repos/BoostTrack` at upstream commit `fb5bfc3`, plus the two local
ReID-loading changes recorded in `docs/tracker-experiments.md`, driven by
`tracking/scripts/run_boosttrack.py`. `repos/` is git-ignored, so line references
are to that checkout. Values are the effective MOT20 defaults after
`dataset_specific_settings` resolution.

**Review status.** Audited against source by `gpt-5.6-sol` on 2026-09-09; 33
corrections and additions applied, 2 audit findings rejected as wrong (recorded
at the end).

## Legend

| Mark | Meaning |
| --- | --- |
| **Set** | Reachable via `--set NAME=VALUE` on `tracking/scripts/run_boosttrack.py` |
| **Set†** | Accepted by `--set`, but only effective because our runner patches upstream |
| **Cond** | Accepted by `--set`, but dormant unless another setting is also changed |
| **Runner** | A settings-dict entry that `--set` explicitly refuses; reached through a dedicated runner flag instead |
| **Hard** | A literal or structural choice in the source; changing it requires editing `repos/BoostTrack` |
| **Inert** | Real upstream parameter with no effect in our integration |

## Summary

| Category | Count |
| --- | ---: |
| Settings-dictionary entries, total | 20 |
| — accepted by `--set` | 16 |
| — refused by `--set`, reached through a runner flag | 4 |
| Of the 16: requiring our runner patch (`max_age`) | 1 |
| Of the 16: dormant under BoostTrack++ defaults (`dlo_boost_coef`) | 1 |
| Hardcoded values and structural choices, tracker (Stages 0–9) | 66 |
| Hardcoded values and structural choices, postprocessing (Stages 10–11) | 15 |
| Inert in our integration | 4 |
| Total rows in this document | 108 |

Counts are derived from the tables below, not asserted independently.

`--set` accepts exactly the 20 dictionary names minus the 4 the runner refuses
(`dataset`, `test_dataset`, `use_embedding`, `use_ecc`;
`run_boosttrack.py:169-171`). Nothing in the dictionaries is globally dead:
`dlo_boost_coef` becomes live if `use_sb` and `use_vt` are both set to `False`.

## Cross-stage couplings

Read this before designing any sweep. Each item is a case where one value moves
more than one thing, which is what makes naive sweeps uninformative.

1. **The appearance weight is not independently settable.** It is derived as
   `1.5 * (1 + lambda_iou + lambda_shape + lambda_mhd)`
   (`tracker/assoc.py:205-207`), so every geometry-weight change moves it too.
   This is a confound in any sweep of those three lambdas and is the leading
   explanation for our flat association sweep.
2. **`iou_threshold` does four jobs** (Stage 7): the composite-cost shortcut
   test, the mask that zeroes the confidence and shape terms, the normal overlap
   requirement for a match to survive, and — halved — the appearance-rescue
   floor. It cannot isolate any one of them.
3. **DLO does not only rescue sub-threshold detections.** Its score updates use
   `np.maximum` over **every** detection with no sub-threshold mask
   (`tracker/boost_track.py:336-355`). Raised scores then flow into the
   association confidence product and into the appearance-update trust formula,
   so DLO changes Stage 7 costs and Stage 8 memory rates, not just Stage 6
   survival. The same is true of DUO's promoted scores.
4. **`s_sim_corr` acts in two stages.** `shape_similarity()` consults it globally
   (`tracker/assoc.py:9-14`) and is called from both DLO
   (`boost_track.py:329-332`) and final matching (`assoc.py:198-203`). Flipping
   it changes rescue decisions *and* match ordering.
5. **The Mahalanobis `limit` and softmax temperature are shared** between DLO and
   final matching, which call the same helper (`assoc.py:38-47`). The
   identically-valued DUO limit at `boost_track.py:288` is a separate literal and
   moves independently.
6. **Motion agreement is normalised across detections.** `MhDist_similarity`
   softmax-normalises down axis 0 (`assoc.py:45`), so a pair's motion score
   depends on how many other detections are in the frame. Our detector supplies
   2.2× as many boxes as YOLOX, so this term is systematically weaker for us,
   independent of `lambda_mhd`.
7. **A rejected match is not re-solved.** Pairs failing the Stage 7 validity test
   are moved to the unmatched lists and the assignment is not recomputed
   (`assoc.py:138-163`), so one invalid high-cost pair can block an otherwise
   valid alternative for that track.
8. **Camera compensation rewrites position but not velocity or covariance.**
   `camera_update()` overwrites `x[:4]` only (`boost_track.py:92-98`), leaving
   the velocity estimate and the uncertainty describing pre-shift coordinates.

## Stage 0 — Inputs we supply

| Parameter | Current | What it does | Status |
| --- | --- | --- | --- |
| detection export floor | 0.05 / 0.10 | Lowest detector score exported. Must stay low or Stages 4–5 have nothing to rescue | Ours |
| NMS overlap | 0.70 (best measured) | Duplicate-box suppression applied before the tracker sees detections | Ours |
| ReID model | `osnet-ain-msdc` | Which appearance model produced the stored vectors. Alternatives: `fastreid-sbs-s50-mot20`, or `none` | Ours |
| crop size | 128x256 | Box resize before the appearance model reads it (`embedding.py`, `_get_general_model`). **Applies at embedding-export time only** — we replace the embedder with a precomputed lookup, so editing this does not change a tracking run | Inert |
| `grid_off` | `True` | `True` = one vector per person; `False` = head/torso/legs thirds. Export-time only | Inert |
| `normalize` | `True` | ImageNet mean/std normalisation of the crop. Export-time only | Inert |
| `dataset` | `mot20` | Selects the dataset-specific `det_thresh` and `dlo_boost_coef` overrides (`default_settings.py:38-41,86-89`). Fixed by the runner | Runner |
| `test_dataset` | from `--split` | Upstream uses it to select ReID checkpoints (`embedding.py:165-175`); inert here because embeddings are precomputed | Runner / Inert |
| `dets is None` vs empty array | empty array | `None` returns immediately **without** incrementing the frame counter or predicting; `np.empty((0,5))` advances the tracker normally (`boost_track.py:167-172`). Our runner always passes an array | Hard |
| input coordinate rescale | 1.0 | Every incoming box is divided by `min(tensor_h/img_h, tensor_w/img_w)` (`boost_track.py:174-177`). We pass a shape-matched dummy tensor so this is exactly 1.0; a wrong tensor shape silently mis-scales every box | Hard |

## Stage 1 — Camera motion compensation (ECC)

Constructed at `tracker/boost_track.py:155`; implementation in `tracker/ecc.py`.

| Parameter | Current | What it does | Status |
| --- | --- | --- | --- |
| `use_ecc` | `True` | Turn camera-shake correction on/off. `--set` refuses it; use `--no-cmc` | Runner |
| `warp_mode` | `MOTION_EUCLIDEAN` | Motion model to fit: translation, Euclidean (shift+rotate), affine, or homography. **Homography would be applied incorrectly** — `camera_update()` discards the homogeneous third component instead of dividing by it (`boost_track.py:92-98`), so switching requires a downstream fix | **Hard** |
| `scale` | `350` | Image is resized to 350 px wide before estimating. Lower is faster and coarser | **Hard** |
| `eps` | `1e-4` | Convergence tolerance | **Hard** |
| `max_iter` | `100` | Maximum refinement iterations | **Hard** |
| `use_cache` | `True` | Reuse warps computed on a previous run of the same video | **Hard** |
| grayscale + `INTER_LINEAR` | fixed | Frames are converted to gray and resized linearly before registration (`ecc.py:52-56,62-76`) | **Hard** |
| `inputMask` | `None` | The whole resized frame participates in registration; no region is excluded (`ecc.py:94`) | **Hard** |
| `gaussFiltSize` | `1` | Final positional argument to `findTransformECC`, overriding OpenCV's usual smoothing (`ecc.py:94`) | **Hard** |
| cache file / key | `video_name` / `tag-frame` | The cache file is chosen by `video_name`, entries keyed by tag plus frame (`ecc.py:127-143`). Our runner qualifies `video_name` per split so `val_half` and `train` cannot poison each other (`run_boosttrack.py:214-226`) | **Hard** |
| cache-hit state handling | returns early | A cache hit returns **before** updating `prev_image` (`ecc.py:141-146`), so the next cache miss registers against a stale frame. Latent defect; harmless only when a sequence is fully cached or fully uncached | **Hard** |

## Stage 2 — Motion model (Kalman filter)

`tracker/kalmanfilter.py`. State is centre-x, centre-y, height, width/height
ratio, plus one velocity per component.

| Parameter | Current | What it does | Status |
| --- | --- | --- | --- |
| covariance policy | `ConstantNoise` | Which uncertainty model to use. `CovariancePolicy` is an abstract base class, so alternatives are a supported extension point | **Hard** |
| `ndim` / observation dims | `8` / `4` | Eight state components, four of them observed (`kalmanfilter.py:83-100`) | **Hard** |
| `dt` | `1` | Timestep linking each position to its velocity (`kalmanfilter.py:89-96`) | **Hard** |
| initial position covariance | `10` | Diagonal entry for x, y, h, ratio on a new track (`kalmanfilter.py:48-54`) | **Hard** |
| initial velocity covariance | `10,000` | `eye × 1000` for the velocity block, then the whole matrix `× 10` — i.e. 1000× the position entry. These are variances, not standard deviations | **Hard** |
| initial velocities | `0` | The state starts at zero and only the first four elements are set from the detection (`kalmanfilter.py:99-102`) | **Hard** |
| measurement noise `R` | `diag(1, 1, 10, 0.01)` | How much to trust the detector per component: x, y, height, ratio. Height is trusted 10× less than position; ratio is trusted heavily | **Hard** |
| process noise `Q` | `1.0` position, `0.01` velocity | Expected deviation from constant-velocity motion | **Hard** |
| confidence → noise | **disabled** | A detection score is passed into `update()` and discarded: `project()` calls `get_R(x, 0)` and `ConstantNoise.get_R` ignores its argument anyway. Open extension point | **Hard** |
| aspect epsilon | `1e-6` | Ratio is `w / (h + 1e-6)` (`boost_track.py:20-33`) | **Hard** |
| nonpositive-ratio clamp | width → `0` | If the predicted ratio goes `<= 0`, the reconstructed box has zero width (`boost_track.py:36-49`) | **Hard** |
| CMC state update | position only | Camera compensation rewrites `x[:4]`; velocities and covariance are left describing pre-shift coordinates (`boost_track.py:92-98`) | **Hard** |

## Stage 3 — Track freshness

`tracker/boost_track.py:76-81`. Weights tracks in matching and sizes the box
stretch in Stage 4.

| Parameter | Current | What it does | Status |
| --- | --- | --- | --- |
| `coef` | `0.9` | Freshness decays 10% per frame unseen, once the track is old enough | **Hard** |
| `n` | `7` | While `age < 7`, freshness is `0.9^(7-age)` regardless of whether the track was just seen — so it *rises* with age rather than decaying. `0.478` at creation, `0.531` at the first prediction (the first value ever used), reaching `1.0` at age 7 | **Hard** |

## Stage 4 — Rescue A: detections that look like known tracks (DLO)

`tracker/boost_track.py:320-357`, helpers in `tracker/assoc.py`. Runs **before**
the Stage 6 cutoff. See cross-stage coupling 3: it rewrites the scores of *all*
detections, not only sub-threshold ones.

| Parameter | Current | What it does | Status |
| --- | --- | --- | --- |
| `use_dlo_boost` | `True` | Turn this rescue on/off | **Set** |
| `use_rich_s` | `True` | `True` = similarity is the mean of motion agreement, shape match and stretched overlap. `False` = plain overlap only | **Set** |
| `use_sb` | `True` | Enable the soft-boost formula | **Set** |
| `use_vt` | `True` | Enable the varying-threshold promotion | **Set** |
| `dlo_boost_coef` | `0.5` | Scales the boost, but only inside the branch taken when `use_sb` and `use_vt` are **both** `False`. Dormant at defaults — overrides at 0.3 and 0.7 gave byte-identical output — but live if those two are also turned off | **Cond** |
| rich-similarity weights | `1/3` each | The three components are averaged with fixed equal weight (`boost_track.py:329-333`). No setting exposes the balance | **Hard** |
| soft-boost `alpha` | `0.65` | Blend: 65% original score, 35% similarity-derived score | **Hard** |
| soft-boost exponent | `1.5` | Applied to similarity before blending. Higher means only strong matches are boosted | **Hard** |
| `threshold_s` | `0.95` | Similarity needed to force-promote a detection matching a track seen last frame | **Hard** |
| `threshold_e` | `0.80` | Floor the threshold decays to for long-missing tracks | **Hard** |
| `n_steps` | `20` | Frames over which 0.95 decays to 0.80, i.e. 0.0075 per frame missing | **Hard** |
| promoted score | `det_thresh + 1e-5` | Promoted detections land just above the cutoff and no higher | **Hard** |
| `k1` | `0.25` | Detection-box stretch per side | **Hard** |
| `k2` | `0.50` | Track-box stretch per side. Both stretches are scaled by `1 - track_confidence` and use the **track's** confidence for both boxes (`assoc.py:82-100`), so young tracks are stretched as much as stale ones (Stage 3) | **Hard** |
| `s_sim_corr` | `False` | `False` selects `shape_similarity_v1`, which divides the height error by the maximum **width**. `True` selects the corrected `v2`. Also affects Stage 7 | **Set** |
| shape transfer function | `exp(-(dw_err + dh_err))` | Fixed exponential with unit scale; no temperature is exposed in either variant (`assoc.py:16-35`) | **Hard** |
| Mahalanobis `limit` | `13.2767` | Motion distances beyond this count as no agreement (99% chi-square, 4 dof). Shared with Stage 7 | **Hard** |
| softmax temperature | `1.0` | How sharply motion-agreement scores compete. Shared with Stage 7 | **Hard** |
| Mahalanobis `n_dims` | `4` | Compare on all four state components. Shared by DLO, DUO and Stage 7 | **Hard** |
| covariance approximation | diagonal | Cross-covariances are discarded; only reciprocals of the diagonal are used (`boost_track.py:269-284`) | **Hard** |

## Stage 5 — Rescue B: detections that match no track (DUO)

`tracker/boost_track.py:286-318`. Also runs before the cutoff.

| Parameter | Current | What it does | Status |
| --- | --- | --- | --- |
| `use_duo_boost` | `True` | Turn this rescue on/off | **Set** |
| Mahalanobis `limit` | `13.2767` | A detection must be **further** than this from every track to qualify. A separate literal from Stage 4's | **Hard** |
| `iou_limit` | `0.3` | Among qualifying boxes overlapping each other above this, only local score maxima survive. Tests each box against its direct neighbours, so ties and chained overlaps can retain several | **Hard** |
| promoted score | `det_thresh + 1e-4` | Where qualifying detections land | **Hard** |
| first-frame skip | `frame_count > 1` | Does not run on frame 1 | **Hard** |

## Stage 6 — The detection cutoff

| Parameter | Current | What it does | Status |
| --- | --- | --- | --- |
| `det_thresh` | `0.4` | Everything below this is discarded, **after** both rescues. Sweeping it 0.4 → 0.6 moved HOTA more than the entire association sweep | **Set** |

## Stage 7 — Matching

`tracker/assoc.py:166-208`.

| Parameter | Current | What it does | Status |
| --- | --- | --- | --- |
| `use_embedding` | `True` | Use appearance at all. `--set` refuses it; use `--reid none` | Runner |
| base overlap coefficient | `1.0` | The cost starts as raw, unweighted overlap before any lambda term is added (`assoc.py:185-187`). Not settable | **Hard** |
| `lambda_iou` | `0.5` | Weight on the *additional* confidence-scaled overlap term, not on the base overlap above | **Set** |
| `lambda_mhd` | `0.25` | Weight on motion agreement | **Set** |
| `lambda_shape` | `0.25` | Weight on width/height similarity — but the term is `lambda_shape × conf × shape`, and `conf` is already zeroed below `iou_threshold` | **Set** |
| confidence fusion | `det_score × track_conf` | How the two confidences combine into the `conf` factor (`assoc.py:189-193`) | **Hard** |
| shape/Mahalanobis coupling | structural | The shape term is added **inside** the branch requiring a non-empty Mahalanobis matrix (`assoc.py:198-203`). With no motion data, `lambda_shape` does nothing however large | **Hard** |
| appearance weight | **derived**, `3.0` | `1.5 × (1 + lambda_iou + lambda_shape + lambda_mhd)`. Cannot be set independently | **Hard** |
| the `1.5` multiplier | `1.5` | The only handle on the appearance weight, and it is a literal | **Hard** |
| appearance gating | none, pre-assignment | Appearance is added to every pair's cost before any overlap test (`assoc.py:205-209`), so zero-overlap pairs still influence which assignment the solver picks | **Hard** |
| `iou_threshold` | `0.3` | Four jobs — see cross-stage coupling 2 | **Set** |
| shortcut condition | ≤1 edge per row and column | If the above-threshold **composite-cost** graph is already one-to-one, pairs are taken directly; otherwise LAPJV runs (`assoc.py:116-126`) | **Hard** |
| `extend_cost` | `True` | Permits a rectangular cost matrix. With no `cost_limit`, LAPJV solves a maximum-cardinality assignment up to the smaller dimension; quality filtering happens afterwards | **Hard** |
| normal overlap requirement | `iou_threshold` = `0.3` | Raw overlap a proposed pair needs to survive (`assoc.py:151`) | **Hard** |
| appearance rescue floor | `iou_threshold / 2` = `0.15` | Absolute minimum overlap for any match. The divisor is a literal, so it cannot move independently | **Hard** |
| appearance rescue similarity | `0.75` | Cosine similarity required to match at only 0.15–0.30 overlap | **Hard** |
| confidence/shape mask | at `iou_threshold` | Below 0.30 overlap the confidence-weighted overlap and shape terms are forced to zero | **Hard** |
| no re-solve after rejection | structural | Rejected pairs go to the unmatched lists; the assignment is not recomputed (`assoc.py:138-163`) | **Hard** |

## Stage 8 — Update, birth, death

`tracker/boost_track.py:229-251`.

| Parameter | Current | What it does | Status |
| --- | --- | --- | --- |
| `af` | `0.95` | Appearance memory. A track keeps at least 95% of its previous appearance vector per update | **Hard** |
| `trust` formula | `(score - det_thresh) / (1 - det_thresh)` | Detections near the cutoff get memory weight approaching 1.0, so they update appearance essentially not at all. Deliberate: rescued boxes cannot poison identity. Note both rescues feed this (coupling 3) | **Hard** |
| EMA renormalisation | L2, every update | The blended vector is renormalised to unit length (`boost_track.py:119-121`), separately from the input vectors' own normalisation | **Hard** |
| track birth | unconditional | Every unmatched detection starts a track. The `>= det_thresh` guard at `:238` is vacuous because the array was filtered at `:200`. There is no confidence bar for birth | **Hard** |
| ID allocation | class-global counter | `KalmanBoxTracker.count` is a class attribute never reset in `BoostTrack.__init__`, so IDs keep climbing across sequences in one process; exported with `+1` (`boost_track.py:57-69,246-247`). Harmless for per-sequence evaluation, but ID values are not reproducible across run groupings | **Hard** |
| `max_age` | `50` | Frames a track survives unseen, normally `max(fps × 2, 30)`; the dictionary entry is ignored and our runner rewires the helper. The runner also mirrors each frame rate onto its split-qualified name and aborts if `max_age` moves (`run_boosttrack.py:300-308`), so 50 holds on every split | **Set†** |
| death condition | `time_since_update > max_age` | Association runs before removal, so a track can still reconnect on the step where its counter reaches 51 — the largest reconnectable frame difference is `max_age + 1` | **Hard** |

## Stage 9 — Output gate

`tracker/boost_track.py:241-251` and `utils.py:29-46`.

| Parameter | Current | What it does | Status |
| --- | --- | --- | --- |
| `min_hits` | `3` | Matched frames in a row required before a track is written. `hit_streak` starts at 0 and only `update()` increments it, so birth does not count: a new track needs 3 *subsequent* matched frames, normally 4 detection-bearing frames including its birth frame. The streak is zeroed during the prediction step following a miss | **Set** |
| matched-this-frame rule | required | `time_since_update < 1`. A track is never written on a frame it was not matched; there is no coasting output. This is why lowering `min_hits` recovers at most 2 of the 3 blank frames after a miss | **Hard** |
| startup exemption | `frame_count <= min_hits` | The first 3 frames of a sequence bypass the streak requirement | **Hard** |
| `aspect_ratio_thresh` | `1.6` | Drops boxes whose width exceeds 1.6× their height. The local variable is named `vertical`, but the filter removes **wide** boxes | **Set** |
| `min_box_area` | `10` | Drops boxes of 10 px² or smaller (strict `>` comparison) | **Set** |
| output quantisation | 1 dp box, 2 dp conf | `write_results_no_score` rounds coordinates to one decimal and confidence to two (`utils.py:17-25`). Applies to our tracks too | **Hard** |

## Stage 10 — Postprocess A: linear interpolation (not run here)

`utils.py:49-115`, invoked from `main.py:133`. Our runner writes raw tracks and
stops; it has no postprocessing invocation at all. Running these stages means
writing our own script against the artifact layout — **no BoostTrack source edit
is required to call these functions.**

| Parameter | Current | What it does | Status |
| --- | --- | --- | --- |
| run at all | not invoked | Upstream gates on `--no_post` (`main.py:124-125`). Our runner has no equivalent | Ours |
| `n_min` | `25` | Only tracks with **more than** 25 rows are eligible. On our train tracks this is the difference between 54,842 gap frames and 51,749 eligible ones | **Hard** |
| `n_dti` | `1000` | Maximum gap length. The function default is 20; `main.py` passes 1000. The test is exclusive, so it admits at most a 999-frame endpoint difference — unbounded in practice, since output gaps cannot exceed `max_age + 1` | **Hard** |
| gap condition | `1 < gap < n_dti` | Single-missing-frame gaps are filled; a zero-length gap is not a gap | **Hard** |
| interpolation | linear | Boxes are slid linearly between the rows either side of the gap, on all four of x, y, w, h | **Hard** |
| output confidence | `-1` | The confidence column is overwritten and destroyed | **Hard** |
| `n_conf` | computed, unused | Dead code in the upstream function | **Hard** |

## Stage 11 — Postprocess B: gradient-boosting smoothing (not run here)

`tracker/GBI.py`, invoked from `main.py:146`.

| Parameter | Current | What it does | Status |
| --- | --- | --- | --- |
| `n_estimators` | `115` | Trees in the smoother. Upstream leaves 71 in a comment as an alternative | **Hard** |
| `learning_rate` | `0.065` | How aggressively each tree corrects. Lower is smoother | **Hard** |
| `min_samples_split` | `6` | Minimum samples before the smoother will branch. Higher is smoother | **Hard** |
| unpinned sklearn defaults | `loss=squared_error`, `max_depth=3`, `subsample=1.0`, `min_samples_leaf=1`, `max_features=None`, `random_state=None` | Everything else is left to the installed scikit-learn (1.9.0 here). These materially define the smoother, so its behaviour is a function of the environment. **Pin them explicitly before reporting any GBI result** | **Hard** |
| what it fits | frame → x, y, w, h | Four independent regressions per track ID. It replaces **every** box, including ones the detector actually observed | **Hard** |
| `interval` argument | accepted, unused | Passed by `main.py` and never read inside `GBInterpolation` | **Hard** |
| interpolation inside GBI | **none** | `GBI.py` defines `LinearInterpolation` and `GBInterpolation` never calls it. GBI only sorts and smooths, so it is a smoother stacked on Stage 10's output, not an interpolator | **Hard** |
| output confidence | `1` | Overwritten again | **Hard** |
| output quantisation | int frame/ID, 2 dp box | `%d,%d,%.2f,...` (`GBI.py:66`) | **Hard** |

## Confirmed dead, dormant, or misleading

Recorded so they are not rediscovered:

- `dlo_boost_coef` — dormant at BoostTrack++ defaults, live only if `use_sb` and
  `use_vt` are both `False` (Stage 4).
- `GeneralSettings.values["max_age"]` — never read; the static frame-rate helper
  is used instead. Our runner patches it (Stage 8).
- The stretched box (`k1`, `k2`) is used for rescue **scoring** only. Association
  uses plain overlap, so a stale track gets a wider net for "is this box worth
  keeping" and a normal net for "is this box mine".
- `s_sim_corr = False` selects the uncorrected shape formula while the corrected
  one ships alongside it, disabled.
- The `>= det_thresh` guard on track birth is vacuous.
- `interval` in `GBInterpolation` and `n_conf` in `dti` are both unused.
- The ECC cache returns before refreshing `prev_image`, so a partially-cached
  sequence registers a miss against a stale frame (Stage 1).

## Audit findings rejected

Two findings from the 2026-09-09 `gpt-5.6-sol` audit were checked and rejected:

- **"`max_age` falls back to 30 on split-qualified sequence names."** True of
  upstream, false here. `run_boosttrack.py:300-308` mirrors each sequence's frame
  rate onto its qualified name and aborts the run if `max_age` differs, precisely
  to prevent this.
- **"`lambda_iou` is described as the weight of all overlap."** The row already
  said "confidence-scaled". The real omission was a separate row for the base
  overlap coefficient of 1.0, which has been added.
