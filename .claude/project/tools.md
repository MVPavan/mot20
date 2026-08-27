# Tools and Runtimes

## Beads (`bd`)

- Installed version during adoption: 1.1.2.
- Use for durable work items, blockers, dependencies, decisions, and multi-session handoff.
- Run `bd prime` for current operational guidance; repository policy is `.beads/beads.md`.
- Every automated write uses a meaningful actor such as `workflow:harness-adoption` or `codex:<purpose>`.
- Do not run `bd dolt push`, backups, Git commits, or pushes without explicit authority.

## Git

- The repository was initialized on branch `main` during harness adoption.
- Preserve unrelated work and stage explicit files only when requested.
- Large datasets and experiment outputs are deliberately ignored.

## Python and ML Stack

No Python version, environment manager, framework, linter, or test runner is selected yet. Read future manifests and CI before choosing commands. Prefer a single documented environment workflow once selected.

## Agent Surfaces

- Shared policy: `AGENTS.md` and `.claude/project/`.
- Codex/Copilot skills: `.agents/skills/`.
- Claude skills: `.claude/skills/` mirror.
- Claude hooks: `.claude/settings.json` and `.claude/hooks/`.
- Codex hooks: `.codex/config.toml`, `.codex/hooks.json`, and `.codex/hooks/`.
- Copilot hooks/instructions: `.github/`.
