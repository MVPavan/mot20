# RF-DETR 2XL Hyperparameter Reviews

## Scope

Independent read-only reviews were requested through Copilot CLI after the
eight-GPU capacity evidence was recorded. GPT-5.6 Sol ran at high reasoning
effort and Claude Opus 5 ran at medium effort. Both reviewed the same packet,
pinned ByteTrack experiment, installed RF-DETR behavior, run configurations,
and generated probe artifacts. Neither reviewer edited the repository or
launched training.

## Shared Conclusion

Both reviews reject starting an uncharacterized long fine-tune, but support the
following first characterization envelope:

- Eight GPUs with DDP and a global batch of 64, expressed as eight images per
  GPU rather than as RF-DETR `batch_size=64`.
- BF16, gradient checkpointing, native aspect-preserving input,
  multi-scale without crop-to-square scale jitter, EMA, and $Q=390$ with
  `group_detr=13`.
- RF-DETR AdamW settings and schedules must be characterized rather than copied
  from ByteTrack's YOLOX SGD recipe.
- Hyperparameter selection should use the clean MOT20/CrowdHuman mix. Byte65
  remains restricted to the separately labeled local test-adapted baseline.

## Findings Applied Before Training

| Finding | Resolution |
| --- | --- |
| Batch semantics differed across projects. | Treat ByteTrack `-b 64` as global batch. Use RF-DETR `batch_size=8`, `devices=8`, global batch 64. |
| Sustained batch-eight/nine without checkpointing OOMed. | Enable RF-DETR-supported gradient checkpointing. |
| The installed Roboflow COCO path defaulted to square resize. | Set `square_resize_div_64=false` and `scale_jitter=false` for aspect-preserving MOT20 preprocessing; rerun capacity checks before training. |
| Query expansion was unseeded. | Launcher now seeds Python and PyTorch before query checkpoint expansion when the configuration supplies `training.seed`; first arms use seed 42. |
| A ten-step run did not characterize LR or warmup. | Start with a clean three-epoch arm at lower LR (`5e-5`, encoder `7.5e-5`), one warmup epoch, then constant LR. |
| Probe metrics were overinterpretable. | Treat them as finite-loss/evaluator wiring evidence only. Use full immutable `val_half` evaluation every epoch and select by AP@[.50:.95], maxDets 390. |

## Rejected Mechanical Transfers From ByteTrack

Do not copy YOLOX's SGD, momentum, LR scaling constant, 80 epochs, 896x1600
geometry, mosaic/mixup, confidence floor, or NMS setting. The transferable
principles are global-batch accounting, strict temporal separation, source-data
semantics, EMA use, and keeping a held-out selection split.

## Remaining Gates

The first three-epoch clean arm is a characterization run, not a final detector.
Compare it against an identical `1e-4` / `1.5e-4` arm from the same seeded
expanded checkpoint, then review the schedule and full-run budget. Resume,
per-rank VRAM telemetry, raw detector export, and ByteTrack handoff validation
remain required before a production detector claim.

## Review Records

- GPT-5.6 Sol High review session: `522d574b-3ef9-4d15-962d-802b8ea45fcf`
- Claude Opus 5 Medium review session: `b1b7a1ba-90c6-45b3-8b50-60e4061e4d85`
- Review packet: `finetuning/docs/plans/2026-09-04-rfdetr-hyperparameter-review-packet.md`

## GPT-5.6 Sol High Follow-Up: Loader Audit

The follow-up review used GPT-5.6 Sol at high reasoning effort through Copilot
CLI. It was read-only: the reviewer received the current configuration, the
full model-free loader-audit result, the 38-test and compile results, and the
historical-capacity limitation. It did not inspect data, execute commands,
construct a model, or run training.

### Evidence Given

- Clean MOT20/CrowdHuman root: 23,838 train images, 4,463 held-out MOT20
  `val_half` images; one pedestrian class; maximum 377 ordinary labels/image.
- `Q = num_select = eval_max_dets = 390`, `group_detr = 13`.
- Selected geometry: `resolution=1120`, direct aspect-preserving loader-side
  multi-scale from 920 through 1320px, `square_resize_div_64=false`,
  `scale_jitter=false`, `do_random_resize_via_padding=true`, and 40px collator
  alignment.
- Complete CPU-only audit: all 28,301 images retained ordinary and ignored
  target counts; all 3,538 sequential batches received minimal 40px-aligned
  padding with no crop. The largest train image was 1335px on its long edge and
  the maximum padded train tensor was 1360x1360.
- `make -C finetuning test` passed 38 tests and `make -C finetuning compile`
  passed. Historical square-path capacity results were explicitly excluded.

### Opinion

Sol judged the configuration coherent enough to proceed only to a fresh
eight-GPU capacity probe. This is an external recommendation, not empirical
capacity evidence.

1. It confirmed that, in the installed RF-DETR version,
   `do_random_resize_via_padding=true` selects loader-side,
   aspect-preserving multi-scale resizing; it cautioned that the flag name is
   misleading because the per-image operation is resize and collator padding is
   a separate later operation.
2. It recommended not patching the 1335px cap overshoot before the capacity
   probe: the behavior is deterministic, exhaustively audited, and does not
   cross the observed 40px padded bucket. A patch would require new exact-size
   regression coverage, the full loader audit, and a new capacity probe. It
   should be revisited only for a strict interoperability or cross-version
   contract, or if another dataset crosses a padding/memory boundary.
3. It considered batch 8 per rank / global 64 with BF16, checkpointing,
   `Q=390`, and a 1360x1360 worst envelope conservative as the next capacity
   experiment, while stressing it is not a proof of fit. It requested repeated
   forward/loss/backward/sync/optimizer/EMA steps, worst geometry with dense and
   ignored targets, per-rank peak allocated/reserved/free memory, shapes, target
   counts, iteration time, finite loss/gradients, explicit headroom, a separate
   evaluation-path check, and pinned provenance from a fresh process.
4. It interpreted the three-epoch `lr=5e-5`, `lr_encoder=7.5e-5`, one-epoch
   warmup, `step`/`lr_drop=100` arm as one warmup epoch plus two constant-LR
   epochs. It found no evidence-based reason to change the rates before that
   characterization, but recommended documenting the effective schedule rather
   than implying a decay will occur.

### Resulting Decision

Keep the installed resize behavior unchanged for the capacity probe. Freeze the
audited configuration and run a fresh eight-GPU, per-rank-telemetry capacity and
evaluation-path probe at the observed 1360x1360 envelope before any three-epoch
characterization or detector-quality conclusion.