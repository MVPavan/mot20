#!/usr/bin/env bash
# SessionStart: load Beads context when bd is available. Never fail the session.

if ! command -v bd >/dev/null 2>&1; then
  [ -x "$HOME/.local/bin/bd" ] && PATH="$HOME/.local/bin:$PATH"
fi

if command -v bd >/dev/null 2>&1; then
  bd prime --hook-json
fi

exit 0
