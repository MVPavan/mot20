#!/usr/bin/env bash
# SessionStart hook for GitHub Copilot agents.
set -euo pipefail

if ! command -v bd >/dev/null 2>&1; then
  [ -x "$HOME/.local/bin/bd" ] && PATH="$HOME/.local/bin:$PATH"
fi

if command -v bd >/dev/null 2>&1; then
  OUTPUT=$(bd prime --hook-json)
  if command -v jq >/dev/null 2>&1; then
    printf '%s\n' "$OUTPUT" | jq 'if .hookSpecificOutput.additionalContext? then . + {additionalContext: .hookSpecificOutput.additionalContext} else . end'
  else
    printf '%s\n' "$OUTPUT"
  fi
fi

exit 0
