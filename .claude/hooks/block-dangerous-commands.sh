#!/usr/bin/env bash
# PreToolUse(Bash): block destructive Git, Beads reinitialization, and broad recursive removal.
set -euo pipefail

INPUT=$(cat)

if command -v jq >/dev/null 2>&1; then
  COMMAND=$(printf '%s\n' "$INPUT" | jq -r '.tool_input.command // empty')
else
  COMMAND=$(printf '%s\n' "$INPUT" | grep -o '"command":"[^"]*"' | sed 's/"command":"//;s/"$//' || true)
fi

[ -z "${COMMAND:-}" ] && exit 0

DANGEROUS_PATTERNS=(
  "git push --force"
  "git push -f"
  "git reset --hard"
  "git clean -fd"
  "git clean -f"
  "git branch -D"
  "git checkout ."
  "git restore ."
  "--no-verify"
)

for pattern in "${DANGEROUS_PATTERNS[@]}"; do
  if printf '%s\n' "$COMMAND" | grep -qF -- "$pattern"; then
    echo "BLOCKED: command matches dangerous pattern '$pattern'. Ask the user before proceeding." >&2
    exit 2
  fi
done

if printf '%s\n' "$COMMAND" | grep -qE '\bbd\b' \
   && printf '%s\n' "$COMMAND" | grep -qE '\binit\b' \
   && printf '%s\n' "$COMMAND" | grep -qE -- '--force|--reinit'; then
  echo "BLOCKED: command can reinitialize and destroy the local Beads store. Ask the user before proceeding." >&2
  exit 2
fi

is_recursive_rm() {
  printf '%s\n' "$1" | grep -qE -- '(^|[[:space:]])-[A-Za-z]*[rR]' ||
    printf '%s\n' "$1" | grep -qiF -- '--recursive'
}

while IFS= read -r segment; do
  [ -z "${segment:-}" ] && continue
  is_recursive_rm "$segment" || continue
  found_target=0
  unsafe=0
  set -f
  set -- $segment
  set +f
  for token in "$@"; do
    [ "$token" = "rm" ] && continue
    case "$token" in -*) continue ;; esac
    found_target=1
    case "$token" in
      /tmp/*) case "$token" in *..*) unsafe=1 ;; esac ;;
      *) unsafe=1 ;;
    esac
  done
  if [ "$found_target" = 0 ] || [ "$unsafe" = 1 ]; then
    echo "BLOCKED: recursive removal outside /tmp, or a target that cannot be proven /tmp-only. Ask the user before proceeding." >&2
    exit 2
  fi
done <<EOF
$(printf '%s\n' "$COMMAND" | grep -oE '(^|[[:space:]])rm[[:space:]]+[^;&|<>]*' || true)
EOF

exit 0
