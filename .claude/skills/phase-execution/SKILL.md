---
name: phase-execution
description: Execute an approved MOT20 workstream phase from a roadmap using Beads epics and child tasks as durable state.
---

# Phase Execution

Use this when the user asks to start or execute a named phase from `docs/workstreams/<name>/roadmap.md`.

## Workflow

1. Read `AGENTS.md`, the roadmap phase, relevant specs, and `.claude/project/verification.md`.
2. Resolve the phase epic with `bd list --spec <roadmap> --json`; confirm it is open and its dependencies are satisfied.
3. Present the phase deliverables, risk, exit criterion, and ready direct-child tasks.
4. For a deep or ambiguous phase, use the planning and document-review skills and obtain approval before implementation.
5. Claim one ready child atomically with `bd ready --parent <epic> --claim --actor "<workflow>:<purpose>"`.
6. Execute the bounded task directly or through permitted delegation. Use test-driven-development for test-first work and systematic-debugging for unexpected failures.
7. Verify the task, then close it with concise evidence in `--reason`. Promote durable discoveries into new Beads children using `--deps discovered-from:<task-id>`.
8. Repeat until every direct child is closed.
9. Run the roadmap exit criterion and verification-before-completion, then close the epic with evidence.
10. Export `.beads/issues.jsonl`, inspect Git status, and report results without committing unless asked.

## Rules

- Beads is the phase-state authority. Do not maintain a competing handwritten status checklist.
- Never close a phase while a child remains open, blocked, or in progress.
- Every Beads write needs an actor.
- Do not run training, broad inference, downloads, external spend, or destructive data operations unless the user has authorized that operational step.
- If requirements change, stop and return to brainstorming or planning rather than silently expanding the phase.
