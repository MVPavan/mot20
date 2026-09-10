# RF-DETR 2XL Byte65 Aspect-Preserving Characterization

## Scope

This completed three-epoch run characterizes the separately named Byte65
`local_test_adapted` overlay. It is not held-out MOT20 evaluation and must not
be used for leaderboard or clean-baseline model selection.

## Contract

| Item | Value |
| --- | --- |
| Dataset | 23,859 training images: 4,468 MOT20 `train_half`, 19,370 CrowdHuman, and 21 Byte65; unchanged 4,463-image MOT20 `val_half` evaluation path |
| Classification | `local_test_adapted` |
| Model | RF-DETR 2XL, one class, gradient checkpointing |
| GPUs / distribution | 8 x RTX 3090, external `torchrun`, DDP with SyncBatchNorm and `find_unused_parameters=True` |
| Geometry | 1120px aspect-preserving, 920--1320px train multi-scale, no crop jitter, 40px collator padding |
| Batch / precision | 8 per rank / 64 global, BF16 AMP |
| Capacity | `num_queries=num_select=eval_max_dets=390`, `group_detr=13` |
| Optimization | AdamW, $lr=5\times10^{-5}$, encoder $lr=7.5\times10^{-5}$, weight decay $10^{-4}$, one warmup epoch, epoch-based decay at 100 |

## Result

The immutable run receipt is
`finetuning/artifacts/rfdetr-2xl-byte65-test-adapted-ddp-batch8-lr5e5-aspect-characterization-2026-09-04-r3/`.
It completed normally at `max_epochs=3` in 2,690.02 seconds. The run produced
`last.ckpt`, `last_ema.pth`, `checkpoint_2.ckpt`, best regular/EMA/total
checkpoints, `metrics.csv`, provenance, training config, and `launcher-result.json`.

| Epoch | Regular $mAP_{50:95}$ | EMA $mAP_{50:95}$ | Regular $mAP_{50}$ | EMA $mAP_{50}$ |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.5736 | 0.5762 | 0.8981 | 0.8965 |
| 2 | 0.5963 | 0.5949 | 0.9121 | 0.9092 |
| 3 | 0.6063 | 0.6085 | 0.8963 | 0.9244 |

The final best-total checkpoint came from EMA ($mAP_{50:95}=0.6085$). These
metrics are characterization evidence for this explicitly test-adapted mix
only. A clean three-epoch run, independent resume validation, detector export,
and ByteTrack handoff remain open.