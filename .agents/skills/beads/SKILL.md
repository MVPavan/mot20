---
name: beads
description: Use for durable project work tracking with bd, including ready work, claiming or closing issues, dependencies, blockers, epics, ideas, backlog, and multi-session handoff.
---

# Beads

Beads stores durable work state for this repository. Run `bd prime` for the live command catalogue and read `.beads/beads.md` for project policy.

## Common Commands

| Goal | Command |
| --- | --- |
| Recover context | `bd prime` |
| Ready work | `bd ready` |
| Inspect | `bd show <id>` |
| Search | `bd search "<query>"` |
| Create | `bd create "title" -t task\|bug\|feature\|epic\|chore\|decision -p 0..4` |
| Claim | `bd update <id> --claim` |
| Close | `bd close <id> --reason="evidence"` |
| Dependencies | `bd dep add <id> <depends-on>`; `bd blocked`; `bd dep tree <id>` |
| Labels | `bd label add\|remove <id> <label>` |
| Epics | `bd epic status`; `bd list --parent <id>`; `bd list --spec <path>` |
| Health | `bd stats`; `bd lint`; `bd stale`; `bd orphans` |

## Repository Conventions

- Pass `--actor "<workflow>:<session-or-purpose>"` on every automated `create`, `update`, `close`, `dep`, or `label` write.
- Use one issue per coherent work item, never one issue per implementation step. The active agent's plan/checklist holds the current turn's steps.
- Use lowercase, kebab-case labels. Domain labels include `detection`, `reid`, `association`, `tracking`, `evaluation`, `dataset`, `experiment`, and `harness`.
- Capture raw possibilities as deferred `idea` issues and accepted later work as deferred `backlog` issues.
- Check for a roadmap, specification, or design doc before creating an epic; attach it with `--spec-id` when it exists.
- Add dependencies only when output is genuinely required. Do not chain independent work merely to encode preferred order.
- Durable knowledge belongs in `.claude/project/learnings.md`; experiment evidence belongs in maintained experiment records. Do not use `bd remember`.
- Do not commit, push, run `bd dolt push`, or configure backups/remotes unless explicitly authorized.
- If issue state changed, refresh the inspectable mirror with `bd export -o .beads/issues.jsonl`. This JSONL is not a full database backup.
