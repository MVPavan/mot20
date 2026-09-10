# RF-DETR 2XL Aspect-Preserving Capacity Probe

## Scope

This bounded clean-baseline experiment measured trainability at the selected
RF-DETR 2XL input envelope. It is capacity evidence only; it does not measure
detector quality, full-epoch throughput, checkpoint/resume behavior, or MOT20
metrics.

## Contract

| Item | Value |
| --- | --- |
| Dataset | Clean MOT20 `train_half` plus CrowdHuman `train`/`val`; MOT20 `val_half` for evaluation path |
| GPUs | 8 x NVIDIA RTX 3090 |
| Model | RF-DETR 2XL, one class, gradient checkpointing |
| Runtime | RF-DETR 1.9.4, RF-DETR+ 1.0.2, PyTorch 2.8.0+cu129 |
| Distributed mode | DDP with SyncBatchNorm and `find_unused_parameters=True`, matching installed RF-DETR `strategy="ddp"` |
| Precision | BF16 AMP |
| Per-rank / global batch | 8 / 64 |
| Query contract | `num_queries=num_select=eval_max_dets=390`, `group_detr=13` |
| Geometry | Aspect-preserving loader-side multi-scale config, then an upward-only `1360 x 1360` padded envelope |
| Train work | Three repeated forward, ignore-aware loss, backward, all-reduce, AdamW, and EMA steps |
| Evaluation work | One EMA forward and postprocess step retaining all 390 predictions per image |

The train batch repeated the audited densest image (377 ordinary and 70 ignored
boxes) eight times, for 3,016 ordinary and 560 ignored boxes per rank. The
evaluation batch repeated the densest validation image (220 ordinary and 28
ignored boxes) eight times. The extra pixels added to reach the envelope were
marked as padding; real image pixels and transformed targets were unchanged.

## Result

The completed immutable receipt is
`finetuning/artifacts/rfdetr-2xl-clean-ddp-batch8-aspect-1360-capacity-2026-09-04-r5/`.
All eight ranks completed all three training steps with finite loss and finite
gradients. Every training and evaluation batch had shape `[8, 3, 1360, 1360]`.
Every evaluation result retained exactly 390 detections per image.

| Measurement | Value |
| --- | ---: |
| Largest peak allocated memory | 18,618,556,928 bytes (17.34 GiB) |
| Largest peak reserved memory | 20,229,128,192 bytes (18.84 GiB) |
| Smallest free memory after probe | 4,583,260,160 bytes (4.27 GiB) |
| Train-step wall time, mean across all ranks/steps | 18.58 seconds |
| Train-step wall-time range | 16.99--20.62 seconds |
| Rank-0 EMA evaluation wall time | 0.23 seconds |

The selected batch 8 per rank / global batch 64 fits this bounded worst-case
shape and target-density workload. It does not establish an upper batch limit
or a full-training memory margin.

## Remaining Gates

- Run the full three-epoch characterization with the selected configuration.
- Validate full held-out `val_half` evaluation, checkpoint writing, and resume.
- Compare the conservative and installed-default learning-rate arms only after
  the first clean characterization has a complete receipt.
- Implement raw detector export and ByteTrack handoff validation.