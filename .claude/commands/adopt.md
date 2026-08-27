---
description: Refresh the repository-specific agent overlay from current repository facts.
---

# Adopt or Refresh the Harness

1. Read `AGENTS.md`, `.claude/project/*.md`, manifests, CI, maintained docs, source, and tests.
2. Use authority order: repository reality, current config/CI, maintained docs, older docs, explicit assumptions.
3. Update only the affected project overlay, shared guide, and tool-specific mirrors requested by the user.
4. Do not invent commands, dependencies, architecture, metrics, or dataset facts.
5. Keep `.agents/skills/` and `.claude/skills/` identical when skills change.
6. Never copy another repository's Beads issues, metadata, remote, project learnings, or machine-local settings.
7. Run structural verification and update `.claude/project/adoption-report.md`.
8. Stop for review before commit or push unless explicitly authorized.
