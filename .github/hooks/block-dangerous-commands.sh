#!/usr/bin/env bash
# PreToolUse hook for GitHub Copilot agents.
set -euo pipefail

INPUT=$(cat)

if command -v jq >/dev/null 2>&1; then
  COMMAND=$(printf '%s\n' "$INPUT" | jq -r '
    def decode_args:
      if type == "object" then .
      elif type == "string" then (fromjson? // {})
      else {}
      end;
    (.tool_input? // .toolArgs? // .toolCalls?[0]?.args? // {} | decode_args) as $args
    | $args.command // $args.commandLine // $args.script // empty
  ')
else
  COMMAND=$(printf '%s\n' "$INPUT" | grep -oE '"(command|commandLine|script)":"[^"]*"' | head -n 1 | sed 's/"[^"]*":"//;s/"$//' || true)
fi

[ -z "${COMMAND:-}" ] && exit 0

deny() {
  reason=$1
  if command -v jq >/dev/null 2>&1; then
    jq -n --arg reason "$reason" '{permissionDecision:"deny", permissionDecisionReason:$reason}'
  else
    printf '{"permissionDecision":"deny","permissionDecisionReason":"Blocked by repository safety policy."}\n'
  fi
  exit 0
}

for pattern in "git push --force" "git push -f" "git reset --hard" "git clean -fd" "git clean -f" "git branch -D" "git checkout ." "git restore ." "--no-verify"; do
  if printf '%s\n' "$COMMAND" | grep -qF -- "$pattern"; then
    deny "Command matches dangerous pattern '$pattern'. Ask the user before proceeding."
  fi
done

if printf '%s\n' "$COMMAND" | grep -qE '\bbd\b' \
   && printf '%s\n' "$COMMAND" | grep -qE '\binit\b' \
   && printf '%s\n' "$COMMAND" | grep -qE -- '--force|--reinit'; then
  deny "Command can reinitialize and destroy the local Beads store. Ask the user before proceeding."
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
    deny "Recursive removal outside /tmp, or a target that cannot be proven /tmp-only, is blocked."
  fi
done <<EOF
$(printf '%s\n' "$COMMAND" | grep -oE '(^|[[:space:]])rm[[:space:]]+[^;&|<>]*' || true)
EOF

exit 0
