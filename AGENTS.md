# Repository Guidelines

## Project Mission

This repository supports participation in the **MOT20 Multiple Object Tracking Challenge**. Work may span object detection, ReID embeddings, motion and appearance association, track management, evaluation, and reproducible experiment analysis.

The repository is currently a scaffold. Do not claim that a framework, package manager, model family, or verification command is established until the corresponding manifest, code, or CI configuration exists.

## Project Structure

Use this intended layout unless the implementation establishes a better documented pattern:

- `track-viz/` — self-contained MOT20 viewer backend, React UI, tests, configuration, scripts, documentation, and ignored derived artifacts.
- `src/mot20/` — production packages, with focused modules such as `detection/`, `reid/`, `association/`, `tracking/`, and `evaluation/`.
- `tests/` — automated tests mirroring source paths; keep lightweight fixtures under `tests/fixtures/`.
- `configs/` — versioned experiment and pipeline configuration. Put tunable values here, not in Python literals.
- `scripts/` — thin, repeatable entry points for training, inference, conversion, and evaluation.
- `docs/` — architecture, dataset contracts, workstream roadmaps, and experiment interpretation.
- `data/`, `weights/`, `outputs/`, `artifacts/` — local or generated content; these are git-ignored.

## Engineering Rules

- Read the relevant caller, configuration, test, and data contract before editing.
- Keep changes surgical. Do not mix feature work with unrelated cleanup.
- Use repo-relative paths in committed code and docs. Machine-specific dataset locations belong in local config or environment variables.
- Preserve official sequence, frame, detection, and result identifiers during conversion. Never silently renumber or drop records.
- Treat metrics as experimental evidence: report the dataset split, detector/checkpoint, tracker configuration, evaluation command, and artifact path. Never fabricate or extrapolate results.
- Do not overwrite datasets, checkpoints, embeddings, predictions, or run directories without explicit approval.
- Secrets belong in ignored `.env` files. Never print or commit tokens.
- Do not commit, push, force-push, or sync Beads remotes unless explicitly asked. Stage explicit files only when a commit is requested.

## Build, Test, and Verification

No application build or test stack exists yet. Until real tooling is added, harness-only changes use structural checks from `.claude/project/verification.md`. When Python tooling is introduced, expose stable root commands such as `make test`, `make lint`, and `make run`, or document the package-manager equivalents.

Tests should be deterministic and prefer small synthetic MOT sequences over full datasets. Separate CPU unit tests from GPU or long-running integration tests. A behavior change should include a test or a documented real invocation that produces the expected artifact.

## Beads Issue Tracker

This project uses **bd (Beads)** for durable work items. Run `bd prime` for runtime guidance and read [`.beads/beads.md`](.beads/beads.md) for repository policy.

- Beads tracks durable tasks, bugs, features, decisions, epics, dependencies, and blockers.
- The current turn's plan/checklist tracks execution steps; do not create one issue per step.
- Durable verified knowledge belongs in `.claude/project/learnings.md`, not `bd remember`.
- Experiment metrics and provenance belong in maintained experiment documentation or a future tracker, not only in an issue description.

## Agent Harness

Read in this order: `AGENTS.md`, `.claude/project/brief.md`, `.claude/project/repo-map.md`, `.claude/project/docs-index.md`, `.claude/project/verification.md`, then relevant rules or skills.

- `.claude/project/` is the repository-specific factual overlay.
- `.agents/skills/` is the canonical Codex/Copilot skill set; `.claude/skills/` is its Claude-compatible mirror.
- `.codex/`, `.claude/`, and `.github/` contain tool-specific hooks and entry points. Keep shared policy here rather than duplicating it across tool files.

## Commits and Pull Requests

Use concise imperative subjects, preferably Conventional Commits, for example `feat: add cosine association` or `fix: preserve empty MOT frames`. Pull requests should explain scope, experiment or behavior impact, validation performed, linked Beads/issues, and any required datasets or checkpoints. Include sample output or visual comparisons when tracker behavior changes.
