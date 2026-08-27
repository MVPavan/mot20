# Beads Issue Tracker Policy

This repository uses **bd (Beads)** for durable work tracking. Run `bd prime` for the current command catalogue and session workflow. This file defines MOT20-specific policy that runtime guidance does not own.

## Responsibility Boundaries

- **Beads:** durable tasks, bugs, features, epics, decisions, blockers, dependencies, and handoff state.
- **Active plan/checklist:** the current work item's implementation steps. One Beads issue per coherent unit of work, never one per step.
- **`.claude/project/learnings.md`:** verified recurring project knowledge. Do not use `bd remember`.
- **Experiment records:** measured metrics, run provenance, comparisons, and artifact locations. Beads may track the work, but must not hold the only copy of experimental evidence.

## Repository Rules

- Issue prefix is `mot`; IDs look like `mot-<hash>`.
- Every automated write passes `--actor "<workflow>:<session-or-purpose>"`. Examples: `workflow:harness-adoption`, `codex:association-review`, or `copilot:pr-review`.
- Labels are lowercase and kebab-case. Use domain labels such as `detection`, `reid`, `association`, `tracking`, `evaluation`, `dataset`, `experiment`, and `harness`.
- Before creating an epic, check for a roadmap, specification, or design document and attach it with `--spec-id`. If none exists, explain why in the issue description.
- Keep child work flat when practical. Add dependencies only when one issue genuinely requires another issue's output.
- Durable work discovered during an issue becomes a new issue with `--deps discovered-from:<current-id>` before handoff.
- Close work only after verification, and put concise evidence in `--reason`.

## Ideas and Backlog

Raw, untriaged possibilities use the deferred `idea` label:

```bash
bd create "idea in one line" -l idea --defer 2099-01-01 -q --actor "<workflow>:<purpose>"
```

Accepted later work uses `backlog`, a real type, and a priority:

```bash
bd create "later work" -l backlog -t task -p 3 --defer 2099-01-01 -q --actor "<workflow>:<purpose>"
```

Do not use ephemeral issues for ideas or backlog items that must survive cleanup.

## Durability and Git Authority

- The embedded Dolt database under `.beads/` is the local source of truth.
- `.beads/issues.jsonl` is a committed, inspectable export for review, interchange, and issue-level recovery. It is **not** a full Dolt backup and does not replace Dolt remote sync or `bd backup`.
- `export.auto: true` is best-effort. After issue changes, refresh explicitly with `bd export -o .beads/issues.jsonl`.
- No Git remote, Dolt remote, or backup destination is configured yet. Do not invent one.
- Do not commit, push, run `bd dolt push`, or configure/execute backups unless the user explicitly authorizes it.
- On a fresh clone, run `bd hooks install` to activate the committed Git hook shims. Never use `bd init --force` or `--reinit-local` without explicit recovery approval.

## Handoff

Before reporting durable work complete: close finished issues, run relevant quality gates, export `issues.jsonl`, inspect `git status`, and report changed files, validation, open blockers, and proposed Git/sync commands. Do not execute those Git/sync commands without authority.
