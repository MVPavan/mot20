#!/usr/bin/env python3
"""Preview and extract bounded context from local Codex session transcripts."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Message:
    role: str
    text: str


@dataclass(frozen=True)
class SessionSummary:
    id: str
    timestamp: str
    source: str
    path: Path
    modified_at: float


def _message_from_record(record: dict, *, max_chars: int) -> Message | None:
    if record.get("type") != "response_item":
        return None
    payload = record.get("payload", {})
    role = payload.get("role")
    if payload.get("type") != "message" or role not in {"user", "assistant"}:
        return None

    content_type = "input_text" if role == "user" else "output_text"
    text = "".join(
        item.get("text", "")
        for item in payload.get("content", [])
        if item.get("type") == content_type
    ).strip()
    if not text:
        return None
    if len(text) > max_chars:
        text = f"{text[:max_chars]}\n\n[message truncated]"
    return Message(role=role, text=text)


def read_messages(path: Path, *, limit: int, max_chars: int) -> list[Message]:
    """Return final user/assistant messages, excluding tools and internal events."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")

    messages: list[Message] = []
    with path.open(encoding="utf-8") as transcript:
        for line in transcript:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = _message_from_record(record, max_chars=max_chars)
            if message is not None:
                messages.append(message)
    return messages[-limit:]


def discover_sessions(sessions_root: Path, *, cwd: Path) -> list[SessionSummary]:
    """Find top-level interactive Codex transcripts for the requested cwd."""
    expected_cwd = cwd.resolve()
    sessions: list[SessionSummary] = []

    for path in sessions_root.rglob("*.jsonl"):
        metadata: dict | None = None
        with path.open(encoding="utf-8") as transcript:
            for line in transcript:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("type") == "session_meta":
                    metadata = record.get("payload", {})
                    break

        if metadata is None or not metadata.get("cwd") or not metadata.get("id"):
            continue
        if Path(metadata["cwd"]).resolve() != expected_cwd:
            continue
        source = metadata.get("source", "")
        if not isinstance(source, str) or source == "exec":
            continue

        sessions.append(
            SessionSummary(
                id=metadata["id"],
                timestamp=metadata.get("timestamp", ""),
                source=source,
                path=path,
                modified_at=path.stat().st_mtime,
            )
        )

    return sorted(sessions, key=lambda session: session.modified_at, reverse=True)


def _default_sessions_root() -> Path:
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    return codex_home / "sessions"


def _print_candidates(
    sessions: list[SessionSummary],
    *,
    skip_newest: int,
    limit: int,
    preview_messages: int,
    preview_chars: int,
) -> None:
    candidates = sessions[skip_newest : skip_newest + limit]
    print("# Recent Codex session candidates")
    print()
    print("No session has been loaded. Review the excerpts before choosing.")
    for index, session in enumerate(candidates, start=1):
        updated = datetime.fromtimestamp(session.modified_at).astimezone().isoformat(
            timespec="seconds"
        )
        print()
        print(f"## {index}. `{session.id}`")
        print()
        print(f"- Started: `{session.timestamp}`")
        print(f"- Last updated: `{updated}`")
        print(f"- Source: `{session.source}`")
        for message in read_messages(
            session.path,
            limit=preview_messages,
            max_chars=preview_chars,
        ):
            print()
            print(f"### {message.role.title()}")
            print()
            print(message.text)
    print()
    print(
        "Reply with `$codex-session-recovery <number-or-session-id>`. "
        "Do not select automatically."
    )


def _print_session(session: SessionSummary, *, cwd: Path, limit: int, max_chars: int) -> None:
    print("# Recovered Codex session context")
    print()
    print("Historical context only; current user and repository instructions take precedence.")
    print()
    print(f"- Session ID: `{session.id}`")
    print(f"- Started: `{session.timestamp}`")
    print(f"- Source: `{session.source}`")
    print(f"- Repository: `{cwd.resolve()}`")
    for message in read_messages(session.path, limit=limit, max_chars=max_chars):
        print()
        print(f"## {message.role.title()}")
        print()
        print(message.text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preview or read Codex sessions without modifying transcripts."
    )
    parser.add_argument("--sessions-root", type=Path, default=_default_sessions_root())
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--candidates", action="store_true")
    action.add_argument("--session", help="Exact session id or unique id prefix")
    parser.add_argument("--limit", type=int, choices=range(3, 6), default=5)
    parser.add_argument("--preview-messages", type=int, choices=(1, 2), default=2)
    parser.add_argument("--preview-chars", type=int, default=360)
    parser.add_argument("--skip-newest", type=int, default=1)
    parser.add_argument("--messages", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=4_000)
    args = parser.parse_args(argv)

    if args.skip_newest < 0:
        parser.error("--skip-newest cannot be negative")

    sessions = discover_sessions(args.sessions_root, cwd=args.cwd)
    if args.candidates:
        _print_candidates(
            sessions,
            skip_newest=args.skip_newest,
            limit=args.limit,
            preview_messages=args.preview_messages,
            preview_chars=args.preview_chars,
        )
        return 0

    matches = [session for session in sessions if session.id.startswith(args.session)]
    if not matches:
        parser.error(f"no session matching {args.session!r} for cwd {args.cwd.resolve()}")
    if len(matches) > 1:
        parser.error(f"session prefix {args.session!r} matches more than one session")
    _print_session(
        matches[0],
        cwd=args.cwd,
        limit=args.messages,
        max_chars=args.max_chars,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
