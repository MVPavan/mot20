import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = SKILL_ROOT / "scripts" / "session_context.py"


def write_session(
    path: Path,
    *,
    session_id: str,
    cwd: Path,
    messages: list[tuple[str, str]],
) -> None:
    records = [
        {
            "type": "session_meta",
            "payload": {
                "id": session_id,
                "timestamp": "2026-09-04T10:00:00Z",
                "cwd": str(cwd),
                "source": "vscode",
            },
        }
    ]
    for role, text in messages:
        content_type = "input_text" if role == "user" else "output_text"
        records.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": role,
                    "content": [{"type": content_type, "text": text}],
                },
            }
        )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class CandidateCliTests(unittest.TestCase):
    def test_shows_three_prior_sessions_with_two_recent_messages_each(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cwd = root / "repo"
            cwd.mkdir()
            sessions = [
                ("current-id", 500),
                ("candidate-one", 400),
                ("candidate-two", 300),
                ("candidate-three", 200),
                ("too-old", 100),
            ]
            for session_id, modified_at in sessions:
                path = root / f"{session_id}.jsonl"
                write_session(
                    path,
                    session_id=session_id,
                    cwd=cwd,
                    messages=[
                        ("user", f"{session_id} old message"),
                        ("assistant", f"{session_id} assistant tail"),
                        ("user", f"{session_id} user tail"),
                    ],
                )
                os.utime(path, (modified_at, modified_at))

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--sessions-root",
                    str(root),
                    "--cwd",
                    str(cwd),
                    "--candidates",
                    "--limit",
                    "3",
                    "--preview-messages",
                    "2",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("current-id", result.stdout)
            self.assertIn("candidate-one", result.stdout)
            self.assertIn("candidate-two", result.stdout)
            self.assertIn("candidate-three", result.stdout)
            self.assertNotIn("too-old", result.stdout)
            self.assertNotIn("candidate-one old message", result.stdout)
            self.assertIn("candidate-one assistant tail", result.stdout)
            self.assertIn("candidate-one user tail", result.stdout)
            self.assertIn("$codex-session-recovery <number-or-session-id>", result.stdout)

    def test_selected_session_loads_a_bounded_conversation_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cwd = root / "repo"
            cwd.mkdir()
            selected = root / "selected.jsonl"
            write_session(
                selected,
                session_id="selected-session-id",
                cwd=cwd,
                messages=[
                    ("user", "old message that must be outside the tail"),
                    ("assistant", "selected assistant tail"),
                    ("user", "selected user tail"),
                ],
            )
            with selected.open("a", encoding="utf-8") as transcript:
                transcript.write(
                    json.dumps(
                        {
                            "type": "response_item",
                            "payload": {
                                "type": "function_call_output",
                                "output": "private tool output",
                            },
                        }
                    )
                    + "\n"
                )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--sessions-root",
                    str(root),
                    "--cwd",
                    str(cwd),
                    "--session",
                    "selected-session",
                    "--messages",
                    "2",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("selected-session-id", result.stdout)
            self.assertNotIn("old message that must be outside the tail", result.stdout)
            self.assertIn("selected assistant tail", result.stdout)
            self.assertIn("selected user tail", result.stdout)
            self.assertNotIn("private tool output", result.stdout)


if __name__ == "__main__":
    unittest.main()
