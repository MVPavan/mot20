# Fine-Tuning Blockers Checklist

## Tracking

This file is the temporary durable checklist for RF-DETR fine-tuning blocker
work while the `bd` command is unavailable. The authoritative requirements and
decisions remain in `finetuning/docs/plans/2026-09-03-rfdetr-2xl-mot20-training.md`.

## Active Work

- [x] Correct and verify the RF-DETR query/group capacity contract.
- [x] Add a versioned RF-DETR 2XL configuration and run launcher.
- [x] Assemble a reproducible mixed image root matching manifest prefixes.
- [x] Add real-manifest auditing: source hashes, complete conversion accounting,
  temporal split checks, image-byte duplicates, and label-density statistics.
- [x] Extract CrowdHuman to an explicit non-overwriting local location and run
  the real clean conversion/audit.
- [x] Run an end-to-end RF-DETR ignored-region tracer through the installed
  dataloader, transforms, criterion, backward pass, and optimizer step.
- [x] Select $Q$, `eval_max_dets`, physical batch, and accumulation from real
  density and capacity evidence.
- [x] Exercise the full RF-DETR launcher across all eight GPUs on an immutable
  dense Byte65 test-adapted subset. Batch 9 per GPU / global batch 72 passed;
  batch 10 per GPU OOMed. The launcher now safely reuses parent-created run
  artifacts in DDP children.
- [x] Run a longer measured-throughput probe at batch 9. Batch 9 without
  gradient checkpointing was not sustained-safe; batch 9 with checkpointing
  completed 720 train and 72 validation images in 103.64 seconds. The
  overhead-inclusive 23,859-image epoch estimate is about 57 minutes.
- [x] Audit every clean train/valid image and sequential batch through the
  selected aspect-preserving loader configuration. The audit verified target
  retention and minimal 40px padding; installed training resize can round the
  nominal 1333px cap up to 1335px, yielding a 1360x1360 padded envelope.
- [x] Run the fresh clean eight-GPU DDP capacity probe at the audited 1360x1360
  envelope. Per-rank batch 8 / global batch 64 completed three finite
  ignore-aware backward/optimizer/EMA steps and an EMA evaluation postprocess
  retaining 390 predictions per image, with 4.27 GiB minimum free VRAM.
- [x] Complete the separate Byte65 `local_test_adapted` three-epoch 1120px
  aspect-preserving batch-8/global-64 characterization under external
  `torchrun` DDP. The immutable `r3` receipt completed in 2,690.02 seconds;
  final regular/EMA $mAP_{50:95}$ was 0.6063/0.6085 and is not held-out
  MOT20 evidence.
- [ ] Implement detector evaluation, raw export, and ByteTrack handoff checks.
- [ ] Record final optimization, schedule, augmentation, and evaluation values.

## Deferred Follow-Up

- [x] Record the user-confirmed exhaustive human audit for the 21 Byte65 images
  and permit their separately named local test-adapted baseline.
- [x] Materialize and audit
  `datasets/finetuning/rfdetr-mot20-crowdhuman-byte65-test-adapted-2026-09-04`
  without changing the clean baseline. It contains 23,859 train images and
  1,154,031 annotations; validation remains 4,463 MOT20 images. Its audit found
  no cross-split duplicate bytes or temporal overlap, and retained $Q=390$.
- [x] Run the loader-audit follow-up through GPT-5.6 Sol High. It recommended
  keeping the deterministic 1335px cap-rounding behavior through a fresh
  1360x1360 capacity probe, with per-rank memory and numerical telemetry.

## Constraints

- Keep the clean baseline to MOT20 `train_half` plus CrowdHuman `train` and
  `val`; reserve MOT20 `val_half` for detector selection.
- Preserve ByteTrack source-label handling unless a material change is reviewed.
- Keep Byte65 separate and label every use as local test-adapted work.
- Create only new, never-overwritten local data or run artifact directories.