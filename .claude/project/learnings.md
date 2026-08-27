# Durable Learnings

Add entries only after a fact, fix, or pattern is verified and likely to recur. Never store secrets, raw credentials, or machine-local absolute paths.

## Entry Format

### YYYY-MM-DD - Short title

- Scope:
- Trigger:
- Rule:
- Evidence:
- Related docs:

---

### 2026-08-27 - Initial MOT20 repository scope

- Scope: repository-wide.
- Trigger: planning project structure or describing current capabilities.
- Rule: this repository targets MOT20 work across detection, ReID, association, tracking, and evaluation, but currently contains no implementation stack. Do not infer a framework or working command until committed files establish it.
- Evidence: repository bootstrap request and current file inventory.
- Related docs: `AGENTS.md`, `.claude/project/brief.md`.

### 2026-08-27 - Separate work state from experimental evidence

- Scope: Beads and experiment reporting.
- Trigger: creating issues for training, inference, or evaluation.
- Rule: Beads tracks the work item and blockers; durable metrics and run provenance belong in maintained experiment documentation or a future experiment tracker.
- Evidence: adopted tracking policy.
- Related docs: `.beads/beads.md`, `.claude/project/tracking.md`.
