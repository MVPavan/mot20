# RF-DETR 2XL Hyperparameter Review Packet

## Decision Requested

Approve or revise the proposed first full local test-adapted run before it is
unblocked:

| Setting | Proposed value | Evidence |
| --- | --- | --- |
| Devices | 8 RTX 3090 GPUs, DDP, synchronized batch norm | Full launcher verified. |
| Batch | 8 per GPU, global 64, accumulation 1 | Matches the user's conservative global-64 target; batch 9 with checkpointing sustained successfully. |
| Memory control | Gradient checkpointing enabled | Non-checkpointed batches 8 and 9 OOMed on the sustained dense multi-scale corpus. |
| Precision | BF16 AMP | Verified in all DDP probes. |
| Input | Aspect-preserving RF-DETR resolution with multi-scale and padding | RF-DETR 2XL rejects ByteTrack's 896x1600 shape contract; square resize is disabled. |
| Capacity | $Q=390$, `group_detr=13`, `num_select=390`, `eval_max_dets=390` | Audited maximum is 377 loss-participating pedestrians. |
| Data | MOT20 `train_half` plus CrowdHuman train/val plus audited Byte65 | Explicitly `local_test_adapted`; not held-out or leaderboard comparable. |
| Validation | Immutable MOT20 `val_half`, eval every epoch | Prevents Byte65/MOT20-test-derived data from entering selection metrics. |

The proposed global batch is 64. RF-DETR's `batch_size` is per device, so that
is `batch_size=8` with eight devices. It is not a claim that batch eight is
sustained-safe without gradient checkpointing.

## ByteTrack Comparison

The pinned reference is `repos/ByteTrack` commit
`d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`, especially
`exps/example/mot/yolox_x_mix_mot20_ch.py`. ByteTrack is a semantic and
workflow reference, not a source of RF-DETR optimizer values.

| Area | ByteTrack YOLOX experiment | RF-DETR proposal | Relevance |
| --- | --- | --- | --- |
| Batch semantics | `tools/train.py` defaults to global `-b 64`; distributed loader divides it by world size. The MOT20 experiment itself does not hard-code 48. | Per-device batch 8, global 64 on eight GPUs. | Directly relevant; values match only after translating semantics. |
| Model/optimizer | YOLOX-X, SGD, momentum 0.9, weight decay `5e-4`; base LR scales as `0.001 / 64 * global_batch`. | RF-DETR 2XL, AdamW, LR `1e-4`, encoder LR `1.5e-4`, weight decay `1e-4`, layer-wise decay defaults. | Architecture-specific; do not port SGD or linear LR scaling without RF-DETR evidence. |
| Schedule | 80 epochs, 1 warmup epoch, inherited YOLOX warm-cosine schedule, final 10 no-augmentation epochs, evaluation every 5 epochs. | Current probes use 1 epoch, no warmup, RF-DETR default step scheduler, evaluation every epoch. | Needs a reviewed full-run schedule; probe settings do not justify an epoch budget. |
| Geometry | 896x1600 with random size range and fixed rectangular inference. | Aspect-preserving RF-DETR native input with multi-scale/padding. | Necessary RF-DETR departure; 896 is invalid for its 40-pixel window constraint. |
| Augmentation | Mosaic and mixup enabled, then disabled for final 10 epochs. | RF-DETR direct aspect-preserving multi-scale resize; no crop-to-square scale jitter, YOLOX mosaic, or mixup translation. | Preserves pedestrian geometry without a multi-image transform. |
| Label capacity | `max_labels=600` before mosaic and 1200 after mosaic. | Query capacity $Q=390$ from the observed maximum 377 ordinary loss targets; ignored boxes use project loss masking. | Necessary model-specific difference. |
| EMA | Enabled. | Enabled. | Aligned in intent. |
| Data | MOT20 temporal `train_half` plus CrowdHuman train and val. | Same clean sources, plus audited Byte65 only in a separately classified local test-adapted run. | Byte65 requires the existing non-comparable classification. |
| Detector evaluation | `test_conf=0.001`, NMS IoU `0.7`. | Evaluate raw top-$Q$ outputs without score floor/NMS; tune ByteTrack handoff separately on `val_half`. | Necessary because RF-DETR outputs are query predictions, not YOLOX dense boxes. |

## Observed Loss And Metrics

The DDP runs verify finite forward loss, backward propagation, optimizer steps,
EMA updates, validation, and checkpoint writing. They do not establish detector
quality or convergence.

The three-epoch batch-four launcher run had exactly one global training step per
epoch. Its reported train loss fell from `15.694` to `13.547` to `12.474`, while
validation loss fell from `11.341` to `11.248` to `11.224`. This is only three
updates on a tiny density-selected corpus. Validation `mAP`, F1, precision, and
recall remained zero, and validation class error remained 100. It is not
meaningful evidence of learning quality.

The 720-image checkpointed batch-nine run performed ten global updates. Its
terminal validation record was AP/pedestrian `0.4334`, F1 `0.7474`, mAP@0.50
`0.7264`, mAP@0.50:0.95 `0.4334`, precision `0.6474`, and recall `0.8838`.
These are detection metrics, not accuracy. They are not decision-grade because
the corpus is intentionally density-selected, only 72 validation images,
contains test-adapted training data, and has just ten updates. The logger did
not emit per-step losses at its 50-step interval, so no loss trend can be
inferred from that run.

## Reviewer Questions

1. Is global batch 64, implemented as eight images per GPU with gradient
   checkpointing, an appropriately conservative operational baseline given the
   observed sustained OOMs at non-checkpointed batches eight and nine?
2. Does RF-DETR 2XL have an evidence-based alternative to the probe's AdamW
   LR/weight-decay/scheduler/warmup defaults for this one-class dense-detection
   mixture, or should those be characterized before a long run?
3. Should multi-scale and scale jitter remain enabled initially, and what
   evaluation/checkpoint cadence balances signal with approximately 57-minute
   epochs?
4. What first-run epoch budget, stopping rule, selection metric, and seed policy
   are defensible without transferring YOLOX's 80-epoch recipe mechanically?
5. What additional short characterization run would make the chosen schedule
   and augmentation policy defensible before committing the full budget?

## Evidence

- `finetuning/docs/experiments/2026-09-04-rfdetr-2xl-ddp-capacity-probe.md`
- `finetuning/artifacts/rfdetr-2xl-byte65-test-adapted-ddp-batch4-2026-09-04/metrics.csv`
- `finetuning/artifacts/rfdetr-2xl-byte65-test-adapted-ddp-batch9-checkpointed-throughput-timed-2026-09-04/metrics.csv`
- `finetuning/artifacts/rfdetr-2xl-byte65-test-adapted-ddp-batch9-checkpointed-throughput-timed-2026-09-04/training_config.json`