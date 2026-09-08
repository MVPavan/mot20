# Fine-Tuning Experiments

This is the index for reproducible RF-DETR detector experiments. Detailed
receipts live under `finetuning/docs/experiments/`; generated artifacts and
logs remain under ignored `finetuning/artifacts/` directories.

## Completed Characterization: Byte65 Aspect-Preserving 3 Epochs

| Item | Value |
| --- | --- |
| Status | Completed normally on 2026-09-04 |
| Classification | `local_test_adapted`; not held-out MOT20 or leaderboard-comparable |
| Training mix | 4,468 MOT20 `train_half`, 19,370 CrowdHuman `train`/`val`, and 21 manually audited Byte65 MOT20-test images |
| Evaluation | Unchanged 4,463-image MOT20 `val_half` |
| Configuration | `finetuning/configs/rfdetr_2xl_byte65_test_adapted_ddp_batch8_lr5e5_aspect_characterization.toml` |
| Launch | External eight-rank `torchrun` after parent-only `--prepare-run` |
| Artifacts | `finetuning/artifacts/rfdetr-2xl-byte65-test-adapted-ddp-batch8-lr5e5-aspect-characterization-2026-09-04-r3/` |
| Duration | 2,690.02 seconds |
| Best checkpoint | `checkpoint_best_total.pth` from EMA |
| Final regular / EMA $mAP_{50:95}$ | 0.6063 / 0.6085 |

Detailed receipt:
`finetuning/docs/experiments/2026-09-04-rfdetr-2xl-byte65-aspect-characterization.md`.

## Completed Full Fine-Tuning: Byte65 Aspect-Preserving 50 Epochs

| Item | Value |
| --- | --- |
| Status | Completed normally on 2026-09-05 after starting 2026-09-04 |
| Classification | `local_test_adapted`; test-derived Byte65 supervision must not be reported as held-out MOT20 or official benchmark evidence |
| Training mix | 4,468 MOT20 `train_half`, 19,370 CrowdHuman `train`/`val`, and 21 manually audited Byte65 MOT20-test images |
| Evaluation | Unchanged 4,463-image MOT20 `val_half` |
| Model / geometry | RF-DETR 2XL, one pedestrian class, 1120px aspect-preserving, 920--1320px training multi-scale, no crop jitter |
| Capacity / distribution | $Q=390$, `group_detr=13`, BF16, eight RTX 3090 GPUs, DDP with SyncBatchNorm and `find_unused_parameters=True`, batch 8 per rank / 64 global |
| Optimization | AdamW, $lr=5\times10^{-5}$, encoder $lr=7.5\times10^{-5}$, weight decay $10^{-4}$, one warmup epoch, epoch-based `lr_drop=40` |
| Configuration | `finetuning/configs/rfdetr_2xl_byte65_test_adapted_ddp_batch8_lr5e5_aspect_full_50e.toml` |
| Durable launcher | `finetuning/scripts/run_rfdetr_2xl_byte65_full_50e.sh` |
| Artifacts | `finetuning/artifacts/rfdetr-2xl-byte65-test-adapted-ddp-batch8-lr5e5-aspect-full-50e-2026-09-04-r1/` |
| Duration | 42,703.70 seconds |
| Best checkpoint | `checkpoint_best_total.pth`, `best_total_source = regular`, `global_step = 2238` (end of epoch 5) |
| Best regular / EMA $mAP_{50:95}$ | 0.6202 at epoch 5 / 0.6198 at epoch 8 |
| Final epoch-49 regular / EMA $mAP_{50:95}$ | 0.5832 / 0.5844 |

Accuracy peaked around epochs 5--8 and then declined for the rest of the run;
`val/loss` reached its minimum at epoch 5 and returned to its epoch-0 level by
epoch 49. The promoted best-total checkpoint holds the epoch-5 peak weights,
not the final-epoch weights.

No `console.log` exists for this run. The launcher's host-side `tee` cannot
write into the container-created root-owned run directory, so `metrics.csv` is
the only surviving per-epoch record.

Detailed receipt:
`finetuning/docs/experiments/2026-09-04-rfdetr-2xl-byte65-aspect-full-50e.md`.

## Completed I5 Mix Ablation: Arms B and C

Single-variable arms against the 50-epoch run above (arm A), same base
checkpoint, same capacity, same `valid` split. Only the training mix changes.
Epoch counts differ because images per epoch differ up to 8x; the arms are
matched on optimisation steps, not epochs.

| Item | Arm B | Arm C |
| --- | --- | --- |
| Mix | MOT20 oversampled 4x to 48.0% of images | MOT20 only, no CrowdHuman, 100% |
| Train images / annotations | 37,263 / 2,908,881 | 4,489 / 587,538 |
| Configuration | `finetuning/configs/rfdetr_2xl_i5-arm-b-mot20-oversampled-50pct.toml` | `finetuning/configs/rfdetr_2xl_i5-arm-c-mot20-only.toml` |
| Dataset root | `datasets/finetuning/rfdetr-i5-armb-2026-09-07` | `datasets/finetuning/rfdetr-i5-armc-2026-09-07` |
| Artifacts | `finetuning/artifacts/rfdetr-2xl-i5-armb-2026-09-07-r1/` | `finetuning/artifacts/rfdetr-2xl-i5-armc-2026-09-07-r1/` |
| Duration | 34,541.3 s, 30 epochs | 16,480.1 s, 100 epochs |
| Best regular $mAP_{50:95}$ | 0.6147 at epoch 1 | 0.6118 at epoch 9 |
| Best EMA $mAP_{50:95}$ | **0.6168** at epoch 1 | **0.6139** at epoch 9 |
| Final $mAP_{50:95}$ | 0.5696 | 0.5579 |

Arm B repeats the same 4,468 MOT20 images four times via manifest duplication,
declared with `intentional_oversampling` metadata so the dataset audit accepts
the duplicate `file_name`s rather than rejecting them as an accident. It adds no
new imagery.

**Verdict: the mix hypothesis is falsified.** Sweeping the MOT20 share across
essentially its whole range, 18.8% -> 48.0% -> 100%, moves peak
$mAP_{50:95}$ by 0.0063, about 1% relative, and moves it the *wrong* way: the
unmodified arm A remains the best of the three. All three arms reproduce the same
shape, an early peak followed by monotone decline, so removing CrowdHuman
entirely does not fix it. Neither arm has been carried through to HOTA.

Interpretation: `docs/tracker-improvements.md`, section I5.

## Completed I6 Resolution Arm: Arm D

The only arm to beat arm A, and the first configuration in the project to beat
the ByteTrack YOLOX-X baseline on HOTA. Trained on the **same dataset as arm A**,
with byte-identical `train` and `valid` manifests
(`train_manifest_sha256 = 9986768f...38ace3`), so the input geometry is the only
variable.

| Item | Value |
| --- | --- |
| Status | Completed 2026-09-08 05:39 |
| Classification | `local_test_adapted` |
| Change against arm A | `max_size = 1600` instead of RF-DETR's `_COCO_MAX_SIZE` default of 1333 |
| Effective geometry at 1920x1080 | 1600x900, scale 0.833, matching YOLOX-X's 0.830. Arm A saw 1333x750, scale 0.694 |
| Capacity / distribution | $Q=390$, `group_detr=13`, BF16, eight RTX 3090 GPUs, micro-batch 4 x `grad_accum_steps` 2, global batch 64, 14 GB of 24 GB per card |
| Configuration | `finetuning/configs/rfdetr_2xl_i5-arm-d-maxsize1600-bs4.toml` |
| Artifacts | `finetuning/artifacts/rfdetr-2xl-i5-armd-2026-09-07-r1/` |
| Duration | 27,256.6 s, 30 epochs — `train_call_wall_seconds` from `launcher-result.json`, the trainer's own clock around the training call. `docs/tracker-improvements.md` quotes 27,795 s for the same run: that is the supervisor span in `finetuning/artifacts/overnight-supervisor.log`, from `preparing` at 21:55:56 to torchrun exit at 05:39:11, so it includes run preparation and checkpoint expansion. Both are correct and measure different things |
| Best regular $mAP_{50:95}$ | 0.6243 at epoch 13 |
| Best EMA $mAP_{50:95}$ | **0.6282 at epoch 9** |
| Best checkpoint | `checkpoint_best_total.pth`, `best_total_source = ema`, `global_step = 3730` |
| Final epoch-29 regular / EMA | 0.6126 / 0.6140 |

Carried through the full chain to HOTA on `val_half` with held-out OSNet ReID and
BoostTrack++ defaults: **HOTA 70.777** against the baseline's 70.208 and arm A's
68.809.

**The gain did not arrive by the predicted mechanism.** I6 was proposed to fix
box regression. LocA moved 85.190 -> 85.321, by 0.131, and remains 2.856 below
the baseline. The gain is association: AssA +2.810 over arm A. The route is
recall, DetRe 76.952 -> 78.256 and 8,082 fewer false negatives, which gives the
tracker more continuous evidence per identity.

Note the EMA branch won here, whereas arm A selected the regular branch. That is
partial evidence for I7: EMA was updating.

Interpretation: `docs/tracker-improvements.md` section I6, and
`docs/experiment-report.md`.

## In Progress: I4 Competition Build

Started 2026-09-08 09:54. Running in host tmux session `mot20`, window
`i4-competition`, so it survives the editor session that launched it. Launcher
`finetuning/scripts/run_rfdetr_2xl_i4_competition.sh`; host-side log
`finetuning/artifacts/i4-competition-launch.log`; per-epoch record
`finetuning/artifacts/rfdetr-2xl-i4-competition-2026-09-08-r1/metrics.csv`.

| Item | Value |
| --- | --- |
| Configuration | `finetuning/configs/rfdetr_2xl_i4-competition-maxsize1600-7gpu.toml` |
| Run directory | `finetuning/artifacts/rfdetr-2xl-i4-competition-2026-09-08-r1/` |
| Initialization | `weights/rfdetr/rf-detr-xxlarge.pth`, sha256 `bf418652...c5d553ae` — the same base as every other arm, **not** arm D's checkpoint |
| Recipe | Arm D's, `max_size = 1600` |
| Distribution | Seven GPUs, devices 0-6; **device 7 deliberately left free** for concurrent inference work. Micro-batch 4 x `grad_accum_steps` 2 x 7 = effective batch 56 |
| Epochs | 8, **506 optimisation steps per epoch, 4,048 total** — see the correction below |
| Measured footprint | About 14 GB of 24 GB per card, matching arm D |
| Checkpoint selection | **`last_ema.pth`** — the final EMA weights. See the deliverable note below; do not accept `checkpoint_best_*.pth` from this run |

#### Correction: the config header's step arithmetic is wrong

The config's header comment computes "505.75 steps per epoch, so 8 epochs is
about 4,046 steps". That division is the right idea but the wrong number, and
the reasoning attached to it is looser than it was stated.

The true count is **506 steps per epoch, 4,048 in eight epochs.** RF-DETR's
`GradAccumAlignedDataset` pads the dataset up to a multiple of
`effective_batch_size x world_size` so that gradient accumulation never fires on
a partial window, so 28,322 images become 28,336 and divide exactly by 56. It
rounds up; it does not truncate. The live `metrics.csv` is consistent with 506.

The stated rationale also conflates two things. If the target were literally the
nearest step count to arm D's 3,730-step peak, **seven** epochs (3,542) is
closer than eight (4,048). Eight was chosen to land just *past* the peak and
inside arm D's observed plateau, which ran to step ~5,967 — a different and
defensible criterion, but not "step-matched". Warmup is likewise not
step-matched: `warmup_epochs = 1.0` is 506 steps here against arm D's 373.

The config file itself is **left unmodified on purpose.** Its sha256
`7d9d58cd...` is recorded as `config_sha256` in the run's `run-provenance.json`;
editing the file, even only its comments, would break that link for a run that
is still executing. The corrected arithmetic lives here instead.

#### Deliverable: which checkpoint

With a formally empty `valid` split there is no validation metric, so
"best checkpoint" has no meaning for this run — and RF-DETR's
`best_model.py` will still emit plausible-looking files: `checkpoint_best_ema.pth`
is documented to be *backfilled with the final EMA weights if the EMA metric
never improved*, which is exactly this case. A file named `best` here would be a
backfill, not a selection.

Declared in advance of the run finishing, so it cannot be chosen post hoc:

- **The deliverable is `last_ema.pth`**, the final EMA weights.
- EMA rather than regular because arm D's ablation selected the EMA branch
  (`best_total_source = ema`, peak 0.6282 EMA against 0.6243 regular), and this
  build exists to enact arm D's recipe.
- `checkpoint_best_total.pth`, `checkpoint_best_ema.pth` and
  `checkpoint_best_regular.pth` from this run are to be treated as artifacts of
  the callback, not as selections, and must not be reported as "best".
- Record the sha256 of `last_ema.pth` in this receipt when the run completes.

The empty `valid` split had never been exercised by the training loop. It works:
Lightning emits `Total length of DataLoader across ranks is zero` as a warning
and proceeds. That is the intended behaviour, not a failure.

Progress bars do not reach the host log because tqdm suppresses them when stdout
is not a terminal, so `metrics.csv` is the per-epoch record of this run.

### Dataset

The ByteTrack-parity dataset was **already built and audited** before this run;
only the training was pending.

| Item | Value |
| --- | --- |
| Dataset root | `datasets/finetuning/rfdetr-mot20-fulltrain-crowdhuman-byte65-test-adapted-2026-09-07` |
| Train | 28,322 images, 1,830,550 annotations: 4,468 MOT20 `train_half` + 4,463 MOT20 `val_half` + 19,370 CrowdHuman + 21 Byte65 |
| MOT20 share | 31.6%, up from arm A's 18.8% |
| Valid | Formally empty, 0 images. `metadata.formal_empty_validation = true`, `checkpoint_selection = final_epoch_selected_by_ablation` |
| Audit | `cross_split_duplicate_images: []`, `mot20_temporal_overlap: []`, `held_out_benchmark_comparable: false` |

**This build has no held-out yardstick.** Because `val_half` is in its training
mix, no number in `docs/results-reference.md` can be re-measured against *this
model*. It does not invalidate the existing figures: arms A–D held `val_half`
out and their weights are untouched, so comparative work on them stays valid
while this runs. Per the 2026-09-06 decision, the
competition build takes the **final** checkpoint at an epoch count chosen on the
ablation build; its `valid` split is a formality for the training loop and must
not drive checkpoint selection.

Note this build also lifts MOT20 from 18.8% to 31.6% of training images, so it
partially confounds I5. The two effects cannot be attributed separately from this
run alone.

## Pending Work

**Tracked in Beads, not here.** Run `bd ready` for available work,
`bd list --status=in_progress` for active runs, and `bd list --status=open` for
everything planned. Epic `RF-DETR 2XL detector training` holds the training
arms; `Evaluation, comparison, and root-cause analysis` holds the measurement
work.

This file keeps the receipts: what each run was, what it measured, and where its
artifacts live. `docs/results-reference.md` is authoritative for every number and
is generated, never hand-edited. `docs/experiment-report.md` carries the
cross-detector comparison and its caveats.
