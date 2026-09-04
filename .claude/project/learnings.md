# Durable Learnings

Add entries only after a fact, fix, or pattern is verified and likely to recur. Never store secrets, raw credentials, or machine-local absolute paths.

## Entry Format

### YYYY-MM-DD - Short title

- Scope:
- Trigger:
- Rule:
- Evidence:
- Related docs:

---

### 2026-08-27 - Initial MOT20 repository scope

- Scope: repository-wide.
- Trigger: planning project structure or describing current capabilities.
- Rule: this repository targets MOT20 work across detection, ReID, association, tracking, and evaluation, but currently contains no implementation stack. Do not infer a framework or working command until committed files establish it.
- Evidence: repository bootstrap request and current file inventory.
- Related docs: `AGENTS.md`, `.claude/project/brief.md`.

### 2026-08-27 - Separate work state from experimental evidence

- Scope: Beads and experiment reporting.
- Trigger: creating issues for training, inference, or evaluation.
- Rule: Beads tracks the work item and blockers; durable metrics and run provenance belong in maintained experiment documentation or a future experiment tracker.
- Evidence: adopted tracking policy.
- Related docs: `.beads/beads.md`, `.claude/project/tracking.md`.

### 2026-09-03 - RF-DETR 2XL preflight works in nvpt-dm

- Scope: RF-DETR detector retrieval, environment setup, and smoke training.
- Trigger: preparing RF-DETR 2XL training on the mounted MOT20 repository.
- Rule: use a project-local UV `.venv` inside `nvpt-dm`, not the host Python. Before running PyTorch, prepend the container's HPC-X UCC and UCX libraries to `LD_LIBRARY_PATH`; the global container environment otherwise imports an incompatible system UCX library and PyTorch fails.
- Rule: with RF-DETR 1.9.4, select a single GPU through `CUDA_VISIBLE_DEVICES` and pass `device="cuda"`. Do not pass `device="cuda:0"` to `train()` until its list/string device-selector defect is revalidated as fixed.
- Rule: RF-DETR+ 1.0.2 loaded the verified 2XL checkpoint and completed a one-class, batch-1 bfloat16 smoke training run on one RTX 3090. This proves basic single-GPU trainability, not real-data capacity, convergence, or multi-GPU readiness.
- Evidence: checkpoint MD5 match, UV import/CUDA check, 880px inference, and 20-batch temporary COCO smoke run with 4.36 GiB peak reserved VRAM.
- Related docs: `finetuning/docs/experiments/2026-09-03-rfdetr-2xl-preflight.md`, `finetuning/docs/plans/2026-09-03-rfdetr-2xl-mot20-training.md`.

### 2026-09-03 - ByteTrack is the detector-workflow reference

- Scope: MOT20 detector conversion, training, evaluation, and ByteTrack handoff.
- Trigger: choosing a behavior that affects RF-DETR training or detection export.
- Rule: use the pinned local ByteTrack YOLOX MOT20 workflow as the default reference and proceed without user review only when following it directly or applying a common practice it also uses. Before implementing a material RF-DETR departure in data semantics, split, transforms, training, evaluation, score/NMS policy, or tracker handoff, review the deviation with the user and record the reference, reason, expected impact, and validation.
- Evidence: user direction after review of `repos/ByteTrack` commit `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`.
- Related docs: `finetuning/docs/plans/2026-09-03-rfdetr-2xl-mot20-training.md`, `repos/ByteTrack/exps/example/mot/yolox_x_mix_mot20_ch.py`.

### 2026-09-03 - RF-DETR 2XL uses native 880px geometry

- Scope: RF-DETR 2XL training/inference geometry on MOT20.
- Trigger: translating ByteTrack's YOLOX 896x1600 input shape.
- Rule: use RF-DETR's native aspect-preserving `resolution: 880` pipeline for the first tracer run. RF-DETR 2XL has `patch_size: 20` and `num_windows: 2`; explicit dimensions must each be divisible by 40, so 896x1600 is rejected. Do not distort inputs to 880x880 without a separate reviewed decision.
- Evidence: installed RF-DETR+ 1.0.2 `RFDETR2XLargeConfig` and `_validate_shape_dims((896, 1600), 40, 20, 2)` in the verified `nvpt-dm` environment.
- Related docs: `finetuning/docs/experiments/2026-09-03-rfdetr-2xl-preflight.md`, `finetuning/docs/plans/2026-09-03-rfdetr-2xl-mot20-training.md`.

### 2026-09-04 - Byte65 is a separate test-adapted source

- Scope: Byte65 detector fine-tuning data.
- Trigger: materializing the post-modification Byte65 YOLO archive for local use.
- Rule: use only `post_modification_annotations/`; `byte65nms_68seq` is a separate local, test-adapted source and must not alter the clean MOT20/CrowdHuman baseline or be reported as held-out MOT20 evidence. The generated audit records seven boundary clips; all output boxes remain valid.
- Evidence: 21 archive images and labels, 21 matching CVAT post-modification records, and 2,588 matching annotations after materialization.
- Related docs: `finetuning/docs/experiments/2026-09-04-byte65nms-68seq-audit.md`, `docs/MOTPolicy.md`.

### 2026-09-04 - Byte65 test-adapted baseline is distinct from clean validation

- Scope: RF-DETR dataset assembly and run provenance.
- Trigger: including the user-confirmed exhaustively human-audited Byte65 labels in the baseline requested for local running.
- Rule: assemble Byte65 only into a new `local_test_adapted` dataset and use a run config with the same classification. Preserve the clean MOT20/CrowdHuman root and its MOT20 `val_half`; the launcher rejects a config/audit classification mismatch.
- Evidence: immutable `rfdetr-mot20-crowdhuman-byte65-test-adapted-2026-09-04` audit: 23,859 train images, 1,154,031 annotations, no cross-split duplicate image bytes or temporal overlap, and $Q=390$ remains sufficient for the observed maximum of 377 labels.
- Related docs: `finetuning/docs/experiments/2026-09-04-byte65nms-68seq-audit.md`, `finetuning/configs/rfdetr_2xl_byte65_test_adapted.toml`, `docs/MOTPolicy.md`.

### 2026-09-04 - RF-DETR DDP capacity requires checkpointing at batch nine

- Scope: RF-DETR 2XL eight-GPU DDP training on dense MOT20/CrowdHuman/Byte65 test-adapted data.
- Trigger: selecting an operational physical batch and projecting runtime for the full local baseline.
- Rule: use an explicit batch size, not RF-DETR `batch_size="auto"`, under Lightning DDP because child relaunches repeat the auto probe. At 1080px multi-scale resolution, BF16, EMA, synchronized batch norm, $Q=390$, and 377 target envelope, batch 9 per GPU / global 72 is sustained-safe only with gradient checkpointing enabled. Preserve the parent-created run checkpoint/provenance for DDP children.
- Evidence: non-checkpointed batches 8 and 9 OOMed on a 720-image dense subset; a checkpointed batch-nine run completed ten global train steps plus 72-image validation in 103.64 seconds across eight 24 GiB RTX 3090 GPUs. This gives an overhead-inclusive 23,859-image epoch estimate of about 57 minutes.
- Related docs: `finetuning/docs/experiments/2026-09-04-rfdetr-2xl-ddp-capacity-probe.md`, `finetuning/scripts/train_rfdetr_2xl.py`, `finetuning/configs/rfdetr_2xl_byte65_test_adapted_ddp_batch9_checkpointed_throughput.toml`.

### 2026-09-04 - Aspect-preserving loader geometry has a cap-rounding envelope

- Scope: clean RF-DETR 2XL MOT20/CrowdHuman data loading at the selected 1120px base resolution.
- Trigger: changing from the historical square-resize path before any further capacity measurement.
- Rule: use `square_resize_div_64=false`, `scale_jitter=false`, and `do_random_resize_via_padding=true` for the selected direct, aspect-preserving loader-side 920--1320px multi-scale path. The 2XL collator minimally rounds each batch maximum to 40px. Do not assume the installed nominal 1333px training cap is strict: its two-stage integer rounding produced a 1335px long edge and a 1360x1360 padded batch envelope.
- Evidence: all 23,838 clean train images / 2,980 sequential batches and 4,463 validation images / 558 sequential batches passed exact transform, source target-count, and minimal padding checks without model construction.
- Related docs: `finetuning/docs/experiments/2026-09-04-rfdetr-loader-geometry-audit.md`, `finetuning/artifacts/rfdetr-loader-geometry-aspect-preserving-1120-2026-09-04-r2.json`.
