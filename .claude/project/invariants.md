# Mechanically Checkable Invariants

Run `.claude/commands/check-invariants.md` after harness or repository-structure changes.

## [INV-01] Canonical guidance exists

- Check: `test -s AGENTS.md && grep -q 'MOT20' AGENTS.md && grep -q '^@AGENTS.md$' CLAUDE.md`
- Expected: exit 0.

## [INV-02] Beads uses the MOT prefix

- Check: `grep -qE '^issue-prefix:[[:space:]]*mot$' .beads/config.yaml`
- Expected: exit 0.

## [INV-03] Large and secret paths are ignored

- Check: `for p in data/MOT20/train.txt weights/model.pth outputs/run.log .env; do git check-ignore -q "$p" || exit 1; done`
- Expected: exit 0.

## [INV-04] Skills are mirrored exactly

- Check: `diff -qr .agents/skills .claude/skills`
- Expected: exit 0.

## [INV-05] Hook configurations parse

- Check: `jq empty .claude/settings.json .codex/hooks.json .github/hooks/project-hooks.json && python3 -c 'import tomllib; tomllib.load(open(".codex/config.toml", "rb"))'`
- Expected: exit 0.

## [INV-06] Hook scripts are valid and executable

- Check: `for h in .claude/hooks/*.sh .codex/hooks/*.sh .github/hooks/*.sh; do bash -n "$h" && test -x "$h" || exit 1; done`
- Expected: exit 0.

## [INV-07] Beads JSONL mirror is not ignored

- Check: `! git check-ignore -q .beads/issues.jsonl`
- Expected: exit 0.

## [INV-08] No reference-project leakage

- Check: `! rg -n 'rf-det[r]-jci|door_kpt[s]|JCI doo[r]|rfd[-]' AGENTS.md CLAUDE.md .claude .agents .codex .github .beads/beads.md`
- Expected: exit 0.
