# RF-DETR 2XL Byte65 Aspect-Preserving Full 50-Epoch Fine-Tuning

## Scope

This completed fifty-epoch run is the full-length counterpart to the three-epoch
characterization in
`finetuning/docs/experiments/2026-09-04-rfdetr-2xl-byte65-aspect-characterization.md`.
It trains on the separately named Byte65 `local_test_adapted` overlay, so it is
not held-out MOT20 evaluation and must not be used for leaderboard or
clean-baseline model selection.

## Contract

| Item | Value |
| --- | --- |
| Dataset | 23,859 training images (4,468 MOT20 `train_half`, 19,370 CrowdHuman, 21 Byte65) with 1,154,031 annotations; unchanged 4,463-image / 676,519-annotation MOT20 `val_half` evaluation path |
| Classification | `local_test_adapted` |
| Model | RF-DETR 2XL, one class, gradient checkpointing |
| GPUs / distribution | 8 x RTX 3090, external `torchrun --standalone --nproc_per_node=8`, DDP with SyncBatchNorm and `find_unused_parameters=True` |
| Geometry | 1120px aspect-preserving, 920--1320px train multi-scale, no crop jitter |
| Batch / precision | 8 per rank / 64 global, BF16 AMP |
| Capacity | `num_queries=num_select=eval_max_dets=390`, `group_detr=13` |
| Optimization | AdamW, $lr=5\times10^{-5}$, encoder $lr=7.5\times10^{-5}$, weight decay $10^{-4}$, one warmup epoch, step schedule with `lr_drop=40`, EMA enabled, `seed=42` |
| Configuration | `finetuning/configs/rfdetr_2xl_byte65_test_adapted_ddp_batch8_lr5e5_aspect_full_50e.toml` (sha256 `8dc3e22a9ab1e54e97a5b11a14677bd1851927d0fd78cb82efc3c6d6614fe610`) |
| Launcher | `finetuning/scripts/run_rfdetr_2xl_byte65_full_50e.sh` |
| Base checkpoint | `finetuning/weights/rfdetr/rf-detr-xxlarge.pth` (sha256 `bf418652d5e07ad441599acfadc65bb1767029079e7a343a6d61675ec5d553ae`), expanded to $Q=390$ as `rfdetr-2xl-q390-initialization.pth` (sha256 `4a1e9dc1e61008e3e193e61b50017516b1e9a243d9679658f93da4ce9ff56b94`) |

## Result

The immutable run receipt is
`finetuning/artifacts/rfdetr-2xl-byte65-test-adapted-ddp-batch8-lr5e5-aspect-full-50e-2026-09-04-r1/`.
It completed normally at `max_epochs=50` in 42,703.70 seconds (about 11 hours
52 minutes), finishing on 2026-09-05. `metrics.csv` records all fifty
validation epochs through step 18,649, and the run produced `last.ckpt`,
`last_ema.pth`, ten `checkpoint_<n>.ckpt` interval checkpoints, best
regular/EMA/total checkpoints, provenance, training config, and
`launcher-result.json`.

Epoch numbers below are the zero-indexed values stored in `metrics.csv`.

| Epoch | Regular $mAP_{50:95}$ | EMA $mAP_{50:95}$ | Regular $mAP_{50}$ | EMA $mAP_{50}$ | Regular $mAR$ | `val/loss` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.5738 | 0.5759 | 0.8986 | 0.8973 | 0.6770 | 6.6686 |
| 2 | 0.6082 | 0.6091 | 0.9274 | 0.9269 | 0.6943 | 6.1832 |
| 4 | 0.6127 | 0.6177 | 0.9326 | 0.9326 | 0.6933 | 6.0899 |
| 5 | **0.6202** | 0.6192 | 0.9337 | 0.9334 | 0.6985 | **6.0166** |
| 8 | 0.6149 | **0.6198** | 0.9319 | 0.9338 | 0.6930 | 6.0638 |
| 10 | 0.6144 | 0.6174 | 0.9325 | 0.9322 | 0.6940 | 6.0760 |
| 15 | 0.6063 | 0.6106 | 0.9220 | 0.9287 | 0.6883 | 6.1795 |
| 20 | 0.6008 | 0.6057 | 0.9185 | 0.9226 | 0.6864 | 6.2568 |
| 30 | 0.6009 | 0.5979 | 0.9239 | 0.9218 | 0.6837 | 6.4180 |
| 40 | 0.5882 | 0.5888 | 0.9213 | 0.9214 | 0.6741 | 6.5861 |
| 49 | 0.5832 | 0.5844 | 0.9194 | 0.9195 | 0.6723 | 6.6603 |

## Checkpoint Selection

`checkpoint_best_total.pth` carries `best_total_source = regular` and
`global_step = 2238`, which at 373 optimizer steps per epoch corresponds to the
end of epoch 5. The promoted checkpoint therefore holds the peak
$mAP_{50:95}=0.6202$ weights, not the final-epoch weights. Its file
modification time is the end of the run because RF-DETR copies the winner in
`on_fit_end`; the timestamp does not indicate which epoch it came from.

## Interpretation

Accuracy peaked at epoch 5 (regular, $mAP_{50:95}=0.6202$) and epoch 8 (EMA,
$mAP_{50:95}=0.6198$), then declined for the remaining forty-plus epochs to
0.5832 / 0.5844 at epoch 49. `val/loss` agrees: its minimum is 6.0166 at epoch
5, rising back to 6.6603 by epoch 49, essentially its epoch-0 value. Precision
kept rising (0.8746 to 0.9095) while recall and $mAR$ fell, so the late epochs
traded recall for precision without a net detection gain.

The extra runtime past roughly epoch 8 produced no usable improvement on this
mix. The best checkpoint exceeds the three-epoch characterization result
($mAP_{50:95}=0.6085$ from EMA) by about 1.2 points, and all of that gain was
earned within the first six epochs. A future run on this mix should use a much
shorter budget, and `lr_drop=40` never took effect in any useful region of the
curve.

These metrics are characterization evidence for this explicitly test-adapted
mix only.

## Known Gaps

- `console.log` was never created for this run. The launcher pipes container
  output through `tee "$run_dir/console.log"` on the host as the invoking user,
  but `--prepare-run` creates the run directory inside the container as `root`
  with mode `drwxr-xr-x`, so the host `tee` cannot open the file. The same gap
  applies to the characterization run. `metrics.csv` is the only surviving
  per-epoch record; no training console output was retained.
- The training tmux window closed when the launcher exited, so no live session
  remains to inspect. Completion evidence comes from the artifact directory.
- Detector export and ByteTrack handoff for the selected checkpoint remain
  open.
