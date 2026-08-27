# Harness Adoption Report

Date: 2026-08-27

## Reference

The harness was selectively adopted from the user-provided reference repository. Its project-specific detection, door-keypoint, dataset, experiment, model, and issue content was not transferred.

## Decisions

- Root `AGENTS.md` is the canonical cross-agent guide.
- `.claude/project/` contains the MOT20-specific factual overlay.
- Portable skills are stored canonically in `.agents/skills/` and mirrored to `.claude/skills/`.
- Claude, Codex, and Copilot receive native hook configuration rather than sharing incompatible hook response formats.
- Complex RF-DETR-specific Copilot orchestration agents and wrapper scripts were not adopted.
- No model, Python version, package manager, or verification command was invented for the empty scaffold.

## Beads

- Initialized a fresh embedded-Dolt project with prefix `mot`; no reference issues or metadata were copied.
- Generic `bd setup` agent instructions were skipped so repository policy can preserve the work-item versus in-turn-plan distinction.
- `.beads/issues.jsonl` is tracked as an inspectable mirror for review and interchange, not described as a full backup.
- No Dolt sync remote or backup destination was configured because the Git repository has no remote.
- The adoption was tracked and closed as Beads issue `mot-buk`.

## Verification

- All 12 skills passed the skill validator and the `.agents/skills/` and `.claude/skills/` trees match.
- Claude, Codex, and Copilot JSON/TOML configuration parsed successfully.
- Hook scripts passed shell syntax, executable-bit, safe-command, destructive-command, and Beads-context checks.
- Ignore rules, reference-project leakage checks, Beads prefix, Git hook installation, `bd stats`, and `bd lint` passed.
- `bd config validate` reports only the intentionally missing Dolt sync remote.
- No files were staged, committed, or pushed.

## Follow-up

Update the project overlay and verification commands when source code, manifests, dataset contracts, CI, and evaluation tooling are introduced. Configure a Git and Dolt remote only when the repository destination is known and the user authorizes it.
