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
| `finetuning/docs/plans/2026-09-03-rfdetr-2xl-mot20-training.md` | Approved gated plan for RF-DETR-2XL one-class detector work | authoritative before MOT20/CrowdHuman/Byte65 conversion or RF-DETR training |
| `finetuning/docs/experiments/2026-09-03-rfdetr-2xl-preflight.md` | Verified RF-DETR 2XL retrieval and bounded trainability evidence | supporting; read before selecting the detector training environment or 2XL configuration |
| `finetuning/docs/experiments/2026-09-04-byte65nms-68seq-audit.md` | Byte65 test-adapted source audit and generated dataset record | supporting; read before using `byte65nms_68seq` in any training overlay |
| `finetuning/docs/experiments/2026-09-04-rfdetr-2xl-ddp-capacity-probe.md` | Eight-GPU RF-DETR launcher and capacity evidence | supporting; read before selecting physical batch, GPU count, or full-data runtime budget |
| `finetuning/docs/experiments/2026-09-04-rfdetr-loader-geometry-audit.md` | Full clean-manifest RF-DETR data transform, target-retention, and collator-padding evidence | supporting; read before interpreting 2XL input geometry or running an aspect-preserving capacity probe |
| `finetuning/docs/experiments/2026-09-04-rfdetr-2xl-aspect-capacity-probe.md` | Eight-GPU clean RF-DETR 2XL batch-8 capacity evidence at the audited 1360x1360 envelope | supporting; read before launching the clean three-epoch characterization |
| `finetuning/docs/experiments/2026-09-04-rfdetr-2xl-byte65-aspect-characterization.md` | Completed Byte65 local-test-adapted RF-DETR 2XL 1120px characterization receipt | supporting; read before interpreting or comparing this test-adapted run |
| `finetuning/experiments.md` | Index of completed and active RF-DETR detector experiments, configurations, launchers, receipts, and logs | supporting; read before monitoring or comparing detector experiments |
| `finetuning/docs/status/2026-09-04-finetuning-readiness.md` | Current RF-DETR 2XL fine-tuning readiness, planned execution, and open gates | supporting; read before review or authorizing a real fine-tuning run |
| `finetuning/docs/status/2026-09-04-finetuning-blockers-todo.md` | Temporary RF-DETR blocker execution checklist while `bd` is unavailable | supporting; update during active fine-tuning blocker work |
| `track-viz/README.md` | Viewer setup, source contracts, controls, capabilities, paths, and known limitations | authoritative for viewer operation |
| `track-viz/docs/HANDOFF.md` | Concise viewer architecture, module map, invariants, and new-session reading order | authoritative orientation for viewer development |
| `track-viz/docs/performance.md` | Latest accepted browser performance, cache, accessibility, and source-integrity evidence | authoritative for viewer release measurements |
| `track-viz/docs/exports.md` | Export API/CLI safety, provenance, codec, and exercised artifacts | authoritative for viewer exports |
| `track-viz/docs/filmstrip-sampling.md` | Deterministic bounded track-filmstrip algorithm | authoritative for filmstrip sampling |

When architecture, dataset contracts, evaluation procedures, or experiment registries are added under `docs/`, add them here with a clear authority level and read condition.
