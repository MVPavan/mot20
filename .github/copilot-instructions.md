# Copilot Instructions - MOT20

Use the root `AGENTS.md` as the canonical shared repository guide.

- Read `.claude/project/brief.md`, `repo-map.md`, and `verification.md` before significant changes.
- Treat datasets, weights, embeddings, predictions, and run artifacts as local protected records.
- Do not invent a Python environment, ML framework, build command, or metric result.
- Keep changes small and preserve MOT sequence, frame, identity, and coordinate contracts.
- Use `.agents/skills/` when a matching workflow applies.
- Use Beads for durable work; policy is `.beads/beads.md`.
- Refresh `.beads/issues.jsonl` after Beads changes.
- Do not migrate machine-local Claude settings or configure remotes, commits, or pushes without explicit authority.

Apply relevant path-scoped instructions under `.github/instructions/` when present.
