# Verification Commands

Only use commands that exist and prove the claimed behavior. No application test suite exists yet.

## Harness and Documentation

```bash
git status --short --branch
jq empty .claude/settings.json .codex/hooks.json .github/hooks/project-hooks.json
python3 -c 'import tomllib; tomllib.load(open(".codex/config.toml", "rb"))'
for h in .claude/hooks/*.sh .codex/hooks/*.sh .github/hooks/*.sh; do bash -n "$h" && test -x "$h"; done
diff -qr .agents/skills .claude/skills
```

## Beads

```bash
bd where
bd stats
bd lint
bd hooks list
bd export -o .beads/issues.jsonl
```

`issues.jsonl` is an inspectable interchange mirror, not a full Dolt backup. Use a configured Dolt remote or `bd backup` for off-machine recovery.

`bd config validate` currently reports the intentionally missing Dolt sync remote. Run it as a diagnostic, not a passing gate, until a repository remote is selected and authorized.

## Future Python Work

When a manifest and test configuration exist, replace this section with exact commands. Expected categories are focused CPU tests, committed lint/format checks, marked GPU integration tests, and a real pipeline invocation that validates MOT-format output before metric claims.

Never present placeholder commands as completed validation.
