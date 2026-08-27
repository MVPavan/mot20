---
description: "Use when editing repository agent guidance, skills, hooks, Beads policy, or tool-specific harness files."
name: "Harness and Agent Customization"
applyTo: "AGENTS.md, CLAUDE.md, .agents/**, .claude/**, .codex/**, .github/**, .beads/**"
---

# Harness and Agent Customization

- Keep `AGENTS.md` as the canonical cross-agent guide.
- Keep `.claude/project/` factual and specific to the current repository.
- Keep `.agents/skills/` and `.claude/skills/` byte-for-byte equivalent.
- Use exact `SKILL.md` casing and validate skill frontmatter.
- Keep tool-specific entry points slim; point to shared guidance rather than duplicating it.
- Verify hook input/output semantics for the target agent before copying a hook.
- Do not copy another repository's Beads database, issues, metadata, remote, or learnings.
- Export `.beads/issues.jsonl` after issue changes and report Git status before completion.
