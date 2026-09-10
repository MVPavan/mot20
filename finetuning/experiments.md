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

## Completed: I4 Competition Build

Started 2026-09-08 09:54, **completed 12:28:47**, `train_call_wall_seconds`
8,983.98 (2 h 30 m). Ran in host tmux session `mot20`, window `i4-competition`,
so it survived the editor session that launched it. Launcher
`finetuning/scripts/run_rfdetr_2xl_i4_competition.sh`; host-side log
`finetuning/artifacts/i4-competition-launch.log`; per-epoch record
`finetuning/artifacts/rfdetr-2xl-i4-competition-2026-09-08-r1/metrics.csv`.

Clean exit: `Trainer.fit stopped: max_epochs=8 reached`, and
`launcher-result.json` was written, which is the only trustworthy completion
marker for this workstream.

**Delivered checkpoint**

| Item | Value |
| --- | --- |
| File | `last_ema.pth` — as pre-declared before the run finished |
| sha256 | `fb3f2f0dba7d3b236652d0ae02c1d19874ee4c5f26145d853fe4ac63ec25ceb6` |
| `epoch` / `global_step` | 8 / **4,048** |

**The predicted checkpoint-naming trap occurred exactly as described.** The
console log's final line reads:

```
EMA metric never improved; saved final EMA weights as checkpoint_best_ema.pth
```

So `checkpoint_best_ema.pth` (sha256 `98ba273f…`) is a **backfill of the final
EMA weights, not a validation-selected best**. No `checkpoint_best_total.pth` or
`checkpoint_best_regular.pth` was produced at all, because there was no metric
to select on.

Verified directly rather than assumed: the two files hold **identical weights** —
532 tensors compared, zero differing, both reporting `global_step = 4048`. So
the risk here was mislabelled provenance, not wrong weights. Either file yields
the same model; only `last_ema.pth` describes truthfully how it was obtained.
Do not report `checkpoint_best_ema.pth` as "best" anywhere.

**Step count confirmed empirically.** The checkpoint's own `global_step` is
4,048, matching 506 x 8 and refuting the config header's "about 4,046 at 505.75
steps per epoch". See the correction below.

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

The true count is **506 steps per epoch, 4,048 in eight epochs** — since
confirmed by the finished run's own `global_step = 4048`. RF-DETR's
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

**Outcome, 2026-09-08 12:28.** All three predictions held: no
`checkpoint_best_total.pth` or `checkpoint_best_regular.pth` was written,
`checkpoint_best_ema.pth` was explicitly backfilled per the console log, and
`last_ema.pth` is the honest name for the weights. Because the two EMA files
turned out to be weight-identical, declaring in advance changed the *record*
rather than the *model* — which is the outcome to hope for, and is only knowable
because the declaration was made first.

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

## Completed I7 Diagnostic: Arm E (contaminated validation)

**Every `val/mAP_*` from arm E and arm E2 is a TRAINING-SET FIT measurement.**
The `valid` split is MOT20 `val_half` and all 4,463 of those frames are also in
`train`; the dataset audit measured the overlap and recorded
`contaminated_validation.duplicate_image_count = 4463`. These numbers cannot be
compared with arms A-D and must never share a table with them.

| | |
| --- | --- |
| Configuration | `finetuning/configs/rfdetr_2xl_i5-arm-e-contaminated-val-diagnostic.toml` |
| Launcher | `finetuning/scripts/run_rfdetr_2xl_arm_e_diagnostic.sh` |
| Dataset root | `datasets/finetuning/rfdetr-i5-arme-2026-09-08` (28,322 train / 4,463 valid) |
| Artifacts | `finetuning/artifacts/rfdetr-2xl-arme-diagnostic-2026-09-08-r1/` |
| Geometry | 7 GPUs, micro-batch 4 x grad_accum 2 = effective 56, 506 steps/epoch — I4 parity |
| Wall time | 26,643 s (7.40 h) for 20 epochs, 22.2 min/epoch |

Purpose was I7: arms A-D all peak in single-digit epochs on held-out data and
then decline, and nothing explains it. Arm E measures the *training-set* side of
the same curve to fork the question — classic overfitting predicts training-set
fit keeps rising, mixture interference predicts it falls too.

**Result: no turnover through epoch 19.** EMA mAP@50:95 rose monotonically
0.6025 → 0.7588 across all 20 epochs (regular 0.6106 → 0.7533). The curve never
peaked, so 20 epochs was not enough to answer the question. Hence arm E2.

Checkpoint retention: run at `checkpoint_interval = 1`, then pruned on
2026-09-09 to epochs 7 (the I4-parity anchor), 13, and 19 (arm E2's resume
point), plus the best/EMA `.pth` files. No published detection came from a
`.ckpt`.

## Active I7 Diagnostic: Arm E2 (arm E resumed to 100 epochs)

Started 2026-09-09 12:42 BST. Inherits arm E's contaminated validation split and
every caveat above.

| | |
| --- | --- |
| Configuration | `finetuning/configs/rfdetr_2xl_i5-arm-e2-continue-100e-6gpu.toml` |
| Launcher | `finetuning/scripts/run_rfdetr_2xl_arm_e2_continue.sh` |
| Artifacts | `finetuning/artifacts/rfdetr-2xl-arme2-continue-2026-09-09-r1/` |
| Resumed from | `rfdetr-2xl-arme-diagnostic-2026-09-08-r1/checkpoint_19.ckpt` |
| Geometry | 6 GPUs, effective batch 48, 590 steps/epoch. Devices 6-7 left free |

**This is a true resume, not a warm start.** RF-DETR's `TrainConfig.resume` is
forwarded to Lightning's `trainer.fit(ckpt_path=...)`, and
`train_rfdetr_2xl.py` already passes the whole `[training]` table through to
`model.train()`, so no launcher change was needed. Optimizer state, EMA buffers,
LR-scheduler position and the epoch counter all continue from epoch 19; the run
log confirms `Restored all states` and `Epoch 20`.

**Arm E2 epoch N is not step-comparable with arm E epoch N or I4 epoch N.** The
7 → 6 GPU change moves steps/epoch from 506 to 590, forfeiting the I4 parity arm
E was built to preserve. Report arm E2 against optimizer steps: arm E's 20
epochs are 10,120 steps, arm E2's epoch 20 ends at 10,710.

**LR schedule changed from step to cosine**, floored at 5% of base. Arm E
carried `lr_drop = 24` inherited from I4, where `epochs = 8` meant it never
fired; at 100 epochs it would fire for the first time, and not at epoch 24 — the
boundary is `lr_drop * steps_per_epoch` = 14,160 global steps, which a run
resuming at 10,120 reaches at epoch ~26.9, leaving 73 of 80 new epochs at 5e-6.
A 10x cliff mid-diagnostic would make the fit curve unreadable. Measured cosine
values: 4.694e-5 at the resume step (6% below arm E's constant 5e-5, so the
handoff is effectively continuous), 4.228e-5 at epoch 30, 2.877e-5 at epoch 50,
8.474e-6 at epoch 80, 2.500e-6 at the end. The first logged step confirms
`train/lr = 4.694037e-05`.

The decay sharpens the I7 fork rather than blunting it: under a monotonically
falling LR, training-set fit should rise monotonically if the model is merely
fitting its data. A *decline* could then be neither overfitting nor an
excessive LR, leaving mixture interference — the 68.4% CrowdHuman share — as the
live explanation.

Checkpointing is `checkpoint_interval = 10`, RF-DETR's default (~16 GB over 80
epochs, against ~160 GB at arm E's interval of 1). Nothing is lost: `last.ckpt`
is written every epoch — and only exists when the interval is not 1, so arm E2
has per-epoch crash recovery that arm E did not — `BestModelCallback` is always
on and independent of the interval, and `metrics.csv` logs every epoch at
`eval_interval = 1`, which is where the diagnostic curve actually comes from.

Two caveats on "best" here. It is best-on-training-data, so it selects nothing.
And because arm E2's `output_dir` differs from the resumed checkpoint's
directory, PyTorch Lightning does not restore `best_model_score` (it logs this
explicitly), so `checkpoint_best_*.pth` is the best of epochs 20-99, not the
best overall. Arm E's own peak is in its `metrics.csv`.

Ignore the `Val (Epoch 20/100)` table printed before training begins: that is
Lightning's 2-batch sanity check, not a full evaluation, which is why its result
is absent from `metrics.csv`.

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
