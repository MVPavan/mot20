# RF-DETR 2XL Eight-GPU DDP Capacity Probe

## Scope

This bounded local experiment exercised the full RF-DETR launcher on the
user-authorized Byte65 `local_test_adapted` baseline. It is a trainability and
capacity check, not a detector-quality result or a leaderboard submission.

## Environment And Fixed Contract

| Item | Value |
| --- | --- |
| GPUs | 8 x NVIDIA GeForce RTX 3090, 24,576 MiB each |
| Model | `RFDETR2XLarge`, 126M parameters; gradient checkpointing for sustained batch-nine run |
| Precision | BF16 AMP |
| Input | Historical RF-DETR multi-scale square path, resolved 1080px scale |
| Query contract | `num_queries=num_select=eval_max_dets=390`, `group_detr=13` |
| Distributed mode | Lightning DDP, 8 devices, synchronized batch norm |
| Optimizer | RF-DETR AdamW with LR `1e-4`, encoder LR `1.5e-4`, weight decay `1e-4` |
| EMA | enabled |
| Source classification | `local_test_adapted`; Byte65 source required in every probe corpus |

Each corpus is a new linked, never-overwritten subset selected by positive-box
density while retaining an ignored-region image and at least one Byte65 image.
Validation uses eight linked MOT20 `val_half` images. Every subset audit found
no cross-split duplicate bytes or MOT20 temporal overlap. The maximum train
image contains 377 positive and 163 ignored boxes; $Q=390$ remains sufficient.

## Launcher Repair

The first DDP run found that Lightning re-executes the Python launcher for each
child rank. The children attempted to recreate the parent-created immutable run
directory and failed. `train_rfdetr_2xl.py` now creates the expanded query
checkpoint and provenance only in the parent; DDP children require and reuse
those artifacts. A focused regression test covers that behavior.

RF-DETR's `batch_size="auto"` probe ran successfully in the parent but is not
safe under its current DDP re-execution behavior: every child repeats the probe
against the default CUDA device. Do not use automatic batch selection for the
multi-GPU full run. Use an explicitly recorded per-device batch size instead.

## Results

| Experiment | Train images | Per-GPU batch | Global batch | Epochs | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Auto-batch parent probe | 32 | 4 recommended | 32 | n/a | Passed worst-case synthetic 377-target probe with 0.70 EMA headroom. |
| DDP launcher regression | 32 | 4 | 32 | 3 | Passed; all eight ranks ran train/validation and wrote `last.ckpt`, regular, EMA, and total best checkpoints. |
| Stress | 40 | 5 | 40 | 1 | Passed. |
| Stress | 48 | 6 | 48 | 1 | Passed. |
| Stress | 56 | 7 | 56 | 1 | Passed. |
| Stress | 64 | 8 | 64 | 1 | Passed. |
| Stress | 72 | 9 | 72 | 1 | Passed. |
| Stress | 80 | 10 | 80 | 1 | Failed: CUDA OOM during decoder deformable attention; GPU 3 had 245 MiB free before a requested 398 MiB allocation. |
| Sustained, no checkpointing | 720 | 9 | 72 | 1 | Failed with CUDA OOM during decoder deformable attention. |
| Sustained, no checkpointing | 720 | 8 | 64 | 1 | Failed with CUDA OOM during decoder deformable attention. |
| Sustained, gradient checkpointing | 720 | 9 | 72 | 1 | Passed; ten global train steps, validation, and all expected checkpoints completed in 103.64 seconds. |

The highest observed one-step setting without gradient checkpointing is **batch 9
per GPU**, global batch 72, with `grad_accum_steps=1`. It is not sustained-safe:
the larger corpus OOMed at both batches nine and eight because multi-scale input
eventually reached a more expensive shape. The highest observed sustained setting
is **batch 9 per GPU / global batch 72 with gradient checkpointing enabled**.
Batch 10 is not viable for this specific 8x3090, 1080px multi-scale, EMA-enabled,
377-target test envelope. This capacity evidence is limited to the historical
square-resize path. On 2026-09-04, the selected MOT20 path changed to
aspect-preserving resize; it requires a new capacity check before training.

The timed sustained run processed 720 training images in ten global steps and 72
validation images in 103.64 seconds, an end-to-end throughput of 6.95 images per
second. At the same overhead-inclusive rate, a 23,859-image train epoch is
projected to take about 57 minutes. This is an operational estimate for this
hardware/software envelope, not a convergence result or a guaranteed budget.

## Artifacts

- Successful 3-epoch DDP run:
  `finetuning/artifacts/rfdetr-2xl-byte65-test-adapted-ddp-batch4-2026-09-04/`
- Highest successful capacity run:
  `finetuning/artifacts/rfdetr-2xl-byte65-test-adapted-ddp-batch9-stress-2026-09-04/`
- Timed sustained successful run:
  `finetuning/artifacts/rfdetr-2xl-byte65-test-adapted-ddp-batch9-checkpointed-throughput-timed-2026-09-04/`
- Sustained non-checkpointed OOM evidence:
  `finetuning/artifacts/rfdetr-2xl-byte65-test-adapted-ddp-batch9-throughput-2026-09-04/`
- Expected batch-10 OOM evidence:
  `finetuning/artifacts/rfdetr-2xl-byte65-test-adapted-ddp-batch10-stress-2026-09-04/`

The test-adapted source and every generated checkpoint retain their run-local
provenance. No metric from these tiny subsets is comparable to held-out MOT20.