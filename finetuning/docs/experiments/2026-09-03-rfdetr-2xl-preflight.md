# RF-DETR 2XL Retrieval and Trainability Preflight

## Scope

This report records retrieval and bounded environment evidence only. It does
not report detector quality and does not authorize a full training run.

The source plan is `finetuning/docs/plans/2026-09-03-rfdetr-2xl-mot20-training.md`.

## Verified Retrieval

| Artifact | Provenance | Verification |
| --- | --- | --- |
| RF-DETR 2XL checkpoint | RF-DETR+ 1.0.2 registry | Downloaded to `finetuning/weights/rfdetr/rf-detr-xxlarge.pth`; MD5 is `e3204689c1f0280427e4c33e6a2ac6cd`, matching the publisher value. |
| CrowdHuman train and validation data | `sshao0516/CrowdHuman` revision `d97203da87e348ea69f7a7633a57c21a956120a6` | Four ZIP archives passed integrity checks: 15,000 training images and 4,370 validation images. |
| CrowdHuman annotations | Same pinned revision | `annotation_train.odgt` has 15,000 records; `annotation_val.odgt` has 4,370 records. |

The CrowdHuman source layout uses the three training archives, validation
archive, and two ODGT files expected by ByteTrack's conversion script. All
retrieved data and weights remain ignored local artifacts.

## Environment

The preflight used the running `nvpt-dm` Docker container with the repository
mounted at its normal repo-relative path. A project-local `.venv` was created
with UV, owned by the workspace user.

| Component | Verified value |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu129 |
| Torchvision | 0.23.0+cu129 |
| RF-DETR | 1.9.4 |
| RF-DETR+ | 1.0.2 |
| GPU | NVIDIA RTX 3090, 24 GiB each; eight visible |

The container's global Python cannot import PyTorch without a corrected
HPC-X library path. Every RF-DETR invocation must include:

```bash
export LD_LIBRARY_PATH=/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
```

RF-DETR 1.9.4 also fails when `train(device="cuda:0")` passes an internal
device list to a string-only code path. Select one GPU with
`CUDA_VISIBLE_DEVICES=0` and use `device="cuda"` until the dependency is
upgraded and revalidated.

## 2XL Smoke Evidence

The downloaded checkpoint loaded successfully and completed an 880px synthetic
inference on a single GPU. A bounded fine-tuning smoke run then used a temporary
one-image, one-class COCO fixture with the model configured for `num_classes=1`.
RF-DETR's small-dataset sampler ran its built-in 20 training batches, followed
by validation, and emitted its regular, total, and trainer checkpoints.

| Measurement | Result |
| --- | --- |
| Trainable parameters | 126 million |
| Precision | bfloat16 automatic mixed precision |
| Physical batch | 1 |
| Wall time | 19.81 seconds |
| Peak allocated VRAM | 4.12 GiB |
| Peak reserved VRAM | 4.36 GiB |

No RF-DETR+ license, package, checkpoint, account, or model-loading gate
prevented training. The released checkpoint's 90-class detection head was
intentionally reinitialized for the one-class pedestrian model.

The synthetic fixture reported zero AP/AR. That result has no detector-quality
meaning and must not be compared with real-data experiments.

## Real Data Construction and Ignore Tracer

On 2026-09-04, the guarded extractor published
`datasets/crowdhuman/extracted/2026-09-04-clean` after resolving every ODGT ID:
15,000 training images and 4,370 validation images. The resulting clean RF-DETR
dataset is `datasets/finetuning/rfdetr-mot20-crowdhuman-clean-2026-09-04`.
Its `audit.json` records 23,838 train images and 4,463 validation images, with
no duplicate bytes or MOT20 temporal overlap. The maximum loss-participating
pedestrian count is 377, so the selected capacity is $Q=390$, the smallest
multiple of `group_detr=13` satisfying $Q > 377$.

The one-batch real ignored-region tracer at this capacity used RF-DETR's
installed COCO datamodule, transforms, collator, ignore-aware criterion, BF16
forward/loss, backward pass, and AdamW step. It observed 33 positive and two
ignored boxes after transforms; loss was finite at `14.84772777557373`; peak
reserved GPU memory was 3,437,232,128 bytes. The run used a trusted published
300-query checkpoint, preserving each pretrained group and retaining model-
initialized query rows for the required extra 90 queries. This is trainability
evidence only, not detector-quality evidence.

## Remaining Gates

- RF-DETR-2XL rejects ByteTrack's exact 896x1600 explicit shape: its installed
  configuration has `patch_size: 20` and `num_windows: 2`, requiring each
  dimension be divisible by 40. The approved first tracer run uses RF-DETR's
  native `resolution: 880` aspect-preserving pipeline, with its 1333px long-side
  cap and collator padding; validate it on real MOT20 data before treating the
  YOLOX input size as transferable.
- Convert the real MOT20, CrowdHuman, and Byte65 inputs to the validated
  one-class COCO contract.
- Audit the complete label density, then select a query and evaluation capacity
  above every image's pedestrian count.
- Test a real-data batch at the chosen resolution, batch size, augmentations,
  and query capacity; the smoke run only proves that batch 1 is feasible.
- Validate multi-GPU distributed training before using all eight GPUs.
- Keep any run containing Byte65 labels marked as local, test-adapted, and not
  leaderboard-comparable under `docs/MOTPolicy.md`.