---
name: subagent-driven-development
description: Execute an approved multi-task plan through bounded implementer and reviewer roles when the active tool and user policy permit delegation.
---

# Subagent-Driven Development

Use delegation to reduce context pressure on standard or deep work, not as automatic ceremony. If delegation is unavailable or not permitted, execute the same bounded packets inline.

## Task Packet

Each task packet must include:

- one measurable objective;
- owned and forbidden files;
- source requirement or plan section;
- known facts and assumptions;
- relevant invariants and data-safety constraints;
- required tests and verification commands;
- test-first or characterization-first flag when applicable;
- commit and external-action authority.

## Workflow

1. Extract non-overlapping tasks from the approved plan.
2. Dispatch a fresh implementer per bounded task only when permitted.
3. Require a concise status: `DONE`, `DONE_WITH_CONCERNS`, `NEEDS_CONTEXT`, or `BLOCKED`.
4. Verify the resulting files and commands in the shared worktree; do not trust the report alone.
5. Run requirement/spec review before code-quality review.
6. Fix bounded findings and re-review. Route changed requirements or architecture decisions back to planning.
7. Run final repository verification before completion.

## Rules

- Do not run parallel writers on overlapping files, configs, datasets, checkpoints, or result directories.
- Tell every worker that other edits may exist and must not be reverted.
- Do not pass raw conversation history; provide only the task packet and necessary context.
- Do not let a worker make architecture, schema, dependency, security, data-loss, or experiment-validity decisions outside its packet.
- After two repeats of the same blocker, stop and request missing context or split the task; do not retry indefinitely.
