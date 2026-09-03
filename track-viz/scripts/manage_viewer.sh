#!/usr/bin/env bash
# Start, stop, or inspect a loopback Track-Viz server by port.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
track_viz_dir="$(cd "$script_dir/.." && pwd)"
repository_root="$(cd "$track_viz_dir/.." && pwd)"
runtime_dir="$track_viz_dir/artifacts/service"

action="${1:-status}"
if [[ $# -gt 0 ]]; then
  shift
fi
port=8000
config="$track_viz_dir/configs/viewer.toml"

usage() {
  echo "usage: $0 {start|stop|restart|status} [--port PORT] [--config PATH]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      port="$2"
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      config="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ ! "$port" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
  echo "Port must be an integer from 1 through 65535." >&2
  exit 2
fi

pid_file="$runtime_dir/viewer-$port.pid"
log_file="$runtime_dir/viewer-$port.log"
python="$repository_root/.venv/bin/python"

viewer_health() {
  curl --fail --silent --max-time 1 "http://127.0.0.1:$port/api/health" 2>/dev/null \
    | grep --quiet '"status":"ok"'
}

port_pids() {
  fuser "$port/tcp" 2>/dev/null || true
}

process_is_viewer() {
  local pid="$1"
  local command
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  command="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$command" == *"run_viewer.py"* ]] || return 1
  [[ "$command" == *"--port $port"* || ("$port" == "8000" && "$command" != *"--port "*) ]]
}

viewer_pid() {
  local pid
  if [[ -f "$pid_file" ]]; then
    pid="$(<"$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && process_is_viewer "$pid"; then
      echo "$pid"
      return 0
    fi
    rm -f "$pid_file"
  fi
  for pid in $(port_pids); do
    if process_is_viewer "$pid"; then
      echo "$pid"
      return 0
    fi
  done
  return 1
}

show_status() {
  local pid
  if viewer_health; then
    echo "Track-Viz is running at http://127.0.0.1:$port"
    if pid="$(viewer_pid)"; then
      echo "PID: $pid"
    fi
    echo "Log: $log_file"
    return 0
  fi
  if [[ -n "$(port_pids)" ]]; then
    echo "Port $port is occupied by a process that is not a healthy Track-Viz server." >&2
    return 2
  fi
  echo "No Track-Viz server is running on port $port."
  return 1
}

start_viewer() {
  local pid attempt
  if viewer_health; then
    echo "Track-Viz is already running at http://127.0.0.1:$port"
    return 0
  fi
  if [[ -n "$(port_pids)" ]]; then
    echo "Port $port is already occupied; refusing to replace that process." >&2
    return 1
  fi
  if [[ ! -x "$python" ]]; then
    echo "Missing Python environment at $python. Follow track-viz/README.md setup first." >&2
    return 2
  fi
  if [[ "$config" != /* ]]; then
    config="$repository_root/$config"
  fi
  if [[ ! -f "$config" ]]; then
    echo "Viewer configuration does not exist: $config" >&2
    return 2
  fi

  make -C "$track_viz_dir" build
  mkdir -p "$runtime_dir"
  nohup env PYTHONPATH="$track_viz_dir/src" \
    "$python" "$script_dir/run_viewer.py" \
    --config "$config" --host 127.0.0.1 --port "$port" \
    >"$log_file" 2>&1 </dev/null &
  pid=$!
  echo "$pid" > "$pid_file"

  for attempt in $(seq 1 40); do
    if viewer_health; then
      echo "Track-Viz started at http://127.0.0.1:$port"
      echo "PID: $pid"
      echo "Log: $log_file"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 0.25
  done

  echo "Track-Viz did not become healthy on port $port. Recent log output:" >&2
  tail -n 30 "$log_file" >&2 || true
  kill -TERM "$pid" 2>/dev/null || true
  rm -f "$pid_file"
  return 1
}

stop_viewer() {
  local pid attempt
  if ! pid="$(viewer_pid)"; then
    if [[ -n "$(port_pids)" ]]; then
      echo "Port $port is occupied by a non-Track-Viz process; it was not stopped." >&2
      return 1
    fi
    echo "No Track-Viz server is running on port $port."
    return 0
  fi

  kill -TERM "$pid"
  for attempt in $(seq 1 40); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$pid_file"
      echo "Track-Viz stopped on port $port."
      return 0
    fi
    sleep 0.25
  done
  kill -KILL "$pid"
  rm -f "$pid_file"
  echo "Track-Viz stopped on port $port after a forced shutdown."
}

case "$action" in
  start) start_viewer ;;
  stop) stop_viewer ;;
  restart)
    stop_viewer
    start_viewer
    ;;
  status) show_status ;;
  *) usage; exit 2 ;;
esac
