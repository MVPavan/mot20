# Repository Map

Physical layout and navigation. Prefer repo-relative paths.

## Current Files

| Path | Role |
| --- | --- |
| `AGENTS.md` | Canonical cross-agent repository guide |
| `CLAUDE.md` | Claude bridge to `AGENTS.md` |
| `.beads/` | Beads store metadata, hooks, policy, and JSONL mirror |
| `.claude/project/` | Repository-specific facts, verification, and durable learnings |
| `.agents/skills/` | Canonical project skills for Codex and Copilot |
| `.claude/skills/` | Claude-compatible mirror of project skills |
| `.codex/` | Codex project settings and lifecycle hooks |
| `.github/` | GitHub Copilot instructions, prompts, and hooks |

## Viewer Implementation

| Path | Role |
| --- | --- |
| `track-viz/` | Self-contained viewer project: Python backend, React UI, tests, configuration, scripts, docs, and ignored derived artifacts |
| `Makefile` | Thin repository entry point delegating viewer commands to `track-viz/` and retaining repository-wide test composition |

## Intended Code Layout

| Path | Intended role |
| --- | --- |
| `src/mot20/detection/` | Detector adapters, inference, and post-processing |
| `src/mot20/reid/` | Appearance models, embeddings, and distance functions |
| `src/mot20/association/` | Cost construction, gating, and assignment |
| `src/mot20/tracking/` | Track state and lifecycle management |
| `src/mot20/evaluation/` | MOT export, validation, and metric integration |
| `configs/` | Versioned pipeline and experiment configuration |
| `scripts/` | Thin training, inference, conversion, and evaluation entry points |
| `tests/` | Unit and integration tests mirroring source layout |
| `docs/` | Architecture, contracts, roadmaps, and experiment reports |

## Local-Only Content

`data/`, `datasets/`, `weights/`, `checkpoints/`, `embeddings/`, `predictions/`, `outputs/`, `runs/`, `results/`, and `artifacts/` are ignored. Treat them as potentially expensive or irreplaceable experiment records.
