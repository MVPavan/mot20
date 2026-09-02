# Verification Commands

Only use commands that exist and prove the claimed behavior.

## Viewer Release Gates

```bash
make test
make lint
make build
make e2e
make smoke-local
make pip-check
make acceptance
```

`make test` composes backend, frontend, and CVAT unit suites. `make e2e` runs
deterministic desktop and narrow Chromium acceptance, including automated axe
scans of Explore, pinned chooser, and Focus. The local-data production-server
journeys and performance/cache measurements are explicit because they take
several minutes and require ignored MOT20 files:

```bash
make e2e-real
```

Read `artifacts/viewer/verification/browser-performance.json` and compare
`source-manifest-before.json` with `source-manifest-after.json` before making a
release claim. Source manifests cover every configured `seqinfo`, annotation,
and enumerated image file.

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

## Focused Viewer Commands

```bash
.venv/bin/python -m pytest -m "not local_data" tests/viewer
npm --prefix web run test
npm --prefix web run typecheck
.venv/bin/python -m unittest discover -s cvat/tests -v
PLAYWRIGHT_BROWSERS_PATH="$PWD/web/.playwright" npm --prefix web run e2e
```

Never present an unrun command or ignored derived report as completed
validation. Performance claims must include the actual machine/browser/display
environment and all three post-warm pointer-latency runs.
