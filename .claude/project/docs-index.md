# Documentation Index

Prefer current repository reality over assumptions or inherited harness text.

| Path | Purpose | Authority |
| --- | --- | --- |
| `AGENTS.md` | Shared operating rules and repository conventions | authoritative |
| `.beads/beads.md` | Beads policy and work/knowledge/experiment separation | authoritative |
| `.claude/project/brief.md` | Current scope, state, and constraints | authoritative |
| `.claude/project/repo-map.md` | Current and intended physical layout | authoritative |
| `.claude/project/verification.md` | Commands allowed to support completion claims | authoritative |
| `.claude/project/invariants.md` | Mechanically checkable repository facts | authoritative |
| `.claude/project/tools.md` | Tool routing and environment guidance | authoritative |
| `.claude/project/tracking.md` | Work, knowledge, and experiment tracking policy | authoritative |
| `.claude/project/learnings.md` | Verified recurring facts | supporting |
| `.claude/project/adoption-report.md` | Harness adoption decisions and deviations | supporting |
| `docs/MOTPolicy.md` | Required language for local test-adapted work and benchmark separation | authoritative for any MOT20 test-derived review or report |
| `docs/tracker-experiments.md` | Index of BoostTrack++ tracking experiments using this repository's RF-DETR detector, with external weight provenance, verified integration contracts, and the layered artifact contract | authoritative for tracker weight checksums, detection-handoff contracts, and artifact/variant naming; read before editing the tracker or exporting detections |
| `docs/tracker-parameters.md` | Stage-wise inventory of every BoostTrack++ parameter and hardcoded literal, including both postprocessing passes, marked by whether it is reachable via `--set`, dead, or requires a source edit | authoritative for which tracker knobs exist and are reachable; read before designing any tracker sweep. Records the derived appearance weight and the four-way overload of `iou_threshold`, both of which confound naive sweeps |
| `docs/tracker-improvements.md` | Analysis and measured evidence behind the detector-gap work; **task status lives in Beads, not here**. Closing the measured detector gap against the ByteTrack YOLOX-X baseline, covering tracker-side tuning, dataset mix, and retraining | authoritative for the two-build evaluation-integrity policy; read before adding `val_half` to training or starting a detector retrain |
| `docs/experiment-report.md` | Narrative summary of every detector experiment, with the detection matrix and the TrackEval tracking matrix compared side by side against the ByteTrack YOLOX-X baseline, and the conclusions drawn from them | supporting; read for interpretation, but `docs/results-reference.md` is authoritative for any figure it quotes |
| `docs/results-reference.md` | Every measured detection, localization, tracking, and detector-training number in one place: RF-DETR arms A-D, the ByteTrack YOLOX-X baseline, the association sweep, and the dataset builds | authoritative for reported metric values; **generated** by `tracking/scripts/build_results_reference.py`, so never edit it by hand — change the generator and re-run |
| `docs/mot20-train-evaluation.md` | Detection mAP and full TrackEval metrics for I4, arm E, and the official ByteTrack YOLOX-X over the entire MOT20 train split (8,931 frames) | authoritative for full-train figures, which `docs/results-reference.md` does not cover. **Every row is training-set fit, not generalization** — all three detectors trained on 100% of this split, so no row may be pooled with a `val_half` figure. Also records why the supplied `yoloxx20` bundle is not the official release |
| `finetuning/docs/plans/2026-09-03-rfdetr-2xl-mot20-training.md` | Approved gated plan for RF-DETR-2XL one-class detector work | authoritative before MOT20/CrowdHuman/Byte65 conversion or RF-DETR training |
| `finetuning/docs/experiments/2026-09-03-rfdetr-2xl-preflight.md` | Verified RF-DETR 2XL retrieval and bounded trainability evidence | supporting; read before selecting the detector training environment or 2XL configuration |
| `finetuning/docs/experiments/2026-09-04-byte65nms-68seq-audit.md` | Byte65 test-adapted source audit and generated dataset record | supporting; read before using `byte65nms_68seq` in any training overlay |
| `finetuning/docs/experiments/2026-09-04-rfdetr-2xl-ddp-capacity-probe.md` | Eight-GPU RF-DETR launcher and capacity evidence | supporting; read before selecting physical batch, GPU count, or full-data runtime budget |
| `finetuning/docs/experiments/2026-09-04-rfdetr-loader-geometry-audit.md` | Full clean-manifest RF-DETR data transform, target-retention, and collator-padding evidence | supporting; read before interpreting 2XL input geometry or running an aspect-preserving capacity probe |
| `finetuning/docs/experiments/2026-09-04-rfdetr-2xl-aspect-capacity-probe.md` | Eight-GPU clean RF-DETR 2XL batch-8 capacity evidence at the audited 1360x1360 envelope | supporting; read before launching the clean three-epoch characterization |
| `finetuning/docs/experiments/2026-09-04-rfdetr-2xl-byte65-aspect-characterization.md` | Completed Byte65 local-test-adapted RF-DETR 2XL 1120px characterization receipt | supporting; read before interpreting or comparing this test-adapted run |
| `finetuning/docs/experiments/2026-09-04-rfdetr-2xl-byte65-aspect-full-50e.md` | Completed Byte65 local-test-adapted RF-DETR 2XL 1120px fifty-epoch fine-tuning receipt, including the epoch-5 peak and post-peak decline | supporting; read before interpreting this run, selecting its checkpoint, or choosing an epoch budget for the same mix |
| `finetuning/experiments.md` | Index of completed and active RF-DETR detector experiments, configurations, launchers, receipts, and logs | supporting; read before monitoring or comparing detector experiments |
| `finetuning/docs/status/2026-09-04-finetuning-readiness.md` | **Historical snapshot, superseded.** RF-DETR 2XL readiness as it stood before any real fine-tuning run; its gates have all been passed | supporting; read for the environment, geometry, and ignore-label evidence only. Its "Remaining Fine-Tuning Run" and "Open Gates" sections are stale — current status is in Beads, current results in `finetuning/experiments.md` |
| `track-viz/README.md` | Viewer setup, source contracts, controls, capabilities, paths, and known limitations | authoritative for viewer operation |
| `track-viz/docs/HANDOFF.md` | Concise viewer architecture, module map, invariants, and new-session reading order | authoritative orientation for viewer development |
| `track-viz/docs/performance.md` | Latest accepted browser performance, cache, accessibility, and source-integrity evidence | authoritative for viewer release measurements |
| `track-viz/docs/exports.md` | Export API/CLI safety, provenance, codec, and exercised artifacts | authoritative for viewer exports |
| `track-viz/docs/filmstrip-sampling.md` | Deterministic bounded track-filmstrip algorithm | authoritative for filmstrip sampling |

When architecture, dataset contracts, evaluation procedures, or experiment registries are added under `docs/`, add them here with a clear authority level and read condition.
