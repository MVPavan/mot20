# Tracking Policy

## Work Tracking

Beads is the durable work tracker.

- Use it for tasks, bugs, features, epics, decisions, blockers, and follow-up work that must survive a session.
- Use the active agent's plan/checklist for the current work item's execution steps.
- Keep one issue per coherent unit of work, not one issue per shell command or implementation step.
- Use lowercase labels. Initial domain labels are `detection`, `reid`, `association`, `tracking`, `evaluation`, `dataset`, `experiment`, `harness`, `idea`, and `backlog`.
- Link epics to a roadmap, specification, or design document with `--spec-id` when one exists.

## Knowledge Tracking

- Put verified, recurring repository facts in `.claude/project/learnings.md`.
- Put architecture and data contracts in maintained documents under `docs/` when created.
- Do not use `bd remember`; Beads holds work state rather than the canonical knowledge base.

## Experiment Tracking

Beads may track the work required to run an experiment, but measured evidence needs separate provenance. Record at least the code revision, dataset/split, detector and checkpoint, ReID model and checkpoint, tracker configuration, seed/environment, evaluation command, metrics, and artifact path.

Do not keep the only copy of a result in an issue comment or terminal transcript.
