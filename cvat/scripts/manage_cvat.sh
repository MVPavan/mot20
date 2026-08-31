#!/usr/bin/env bash
# Lifecycle commands for the isolated MOT20 CVAT stack. Never removes volumes.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
stack_dir="$(cd "$script_dir/.." && pwd)"
env_file="$stack_dir/.env"
compose_file="$stack_dir/docker-compose.yml"

if [[ ! -f "$env_file" ]]; then
  echo "Missing $env_file. Copy config/stack.env.example to .env and set CVAT_ADMIN_PASSWORD." >&2
  exit 2
fi

set -a
# shellcheck source=/dev/null
source "$env_file"
set +a
compose=(docker compose --env-file "$env_file" -f "$compose_file")

status() {
  "${compose[@]}" ps
  if curl -fsS "http://${CVAT_HOST:-127.0.0.1}:${CVAT_HOST_PORT:-8082}/api/server/about" >/dev/null; then
    echo "CVAT is reachable at http://${CVAT_HOST:-127.0.0.1}:${CVAT_HOST_PORT:-8082}"
  else
    echo "CVAT API is not reachable yet" >&2
    return 1
  fi
}

wait_for_api() {
  local attempt
  for attempt in $(seq 1 30); do
    if curl -fsS "http://${CVAT_HOST:-127.0.0.1}:${CVAT_HOST_PORT:-8082}/api/server/about" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  echo "CVAT did not become reachable within 60 seconds" >&2
  return 1
}

case "${1:-status}" in
  start) "${compose[@]}" up -d; wait_for_api; status ;;
  stop) "${compose[@]}" down ;;
  restart) "${compose[@]}" restart "${@:2}" ;;
  status) status ;;
  ps) "${compose[@]}" ps ;;
  logs) "${compose[@]}" logs -f --tail=200 "${2:?service required}" ;;
  create-admin)
    : "${CVAT_ADMIN_USER:?set CVAT_ADMIN_USER in .env}"
    : "${CVAT_ADMIN_EMAIL:?set CVAT_ADMIN_EMAIL in .env}"
    : "${CVAT_ADMIN_PASSWORD:?set a non-empty CVAT_ADMIN_PASSWORD in .env}"
    "${compose[@]}" exec -T -e DJANGO_SUPERUSER_PASSWORD="$CVAT_ADMIN_PASSWORD" cvat_server python3 /home/django/manage.py createsuperuser --noinput --username "$CVAT_ADMIN_USER" --email "$CVAT_ADMIN_EMAIL"
    ;;
  backup)
    destination="$stack_dir/backups/$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$destination"
    "${compose[@]}" exec -T cvat_db pg_dump --clean --if-exists -U root -d cvat > "$destination/cvat.sql"
    echo "PostgreSQL backup written to $destination/cvat.sql"
    ;;
  *) echo "usage: $0 {start|stop|restart|status|ps|logs SERVICE|create-admin|backup}" >&2; exit 2 ;;
esac
