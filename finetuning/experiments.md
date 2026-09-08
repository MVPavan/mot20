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
