from __future__ import annotations

import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


class ViewerServiceScriptTest(unittest.TestCase):
    def test_status_reports_that_an_unused_port_is_stopped(self) -> None:
        repository_root = Path(__file__).resolve().parents[3]
        script = repository_root / "track-viz" / "scripts" / "manage_viewer.sh"
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        result = subprocess.run(
            [str(script), "status", "--port", str(port)],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn(f"No Track-Viz server is running on port {port}.", result.stdout)

    def test_start_status_and_stop_manage_the_selected_port(self) -> None:
        repository_root = Path(__file__).resolve().parents[3]
        script = repository_root / "track-viz" / "scripts" / "manage_viewer.sh"
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = Path(temporary_directory) / "viewer.toml"
            config.write_text("sources = []\n", encoding="utf-8")
            command = [str(script), "--port", str(port), "--config", str(config)]
            try:
                started = subprocess.run(
                    [str(script), "start", *command[1:]],
                    cwd=repository_root,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(started.returncode, 0, started.stderr)
                self.assertIn(f"Track-Viz started at http://127.0.0.1:{port}", started.stdout)

                status = subprocess.run(
                    [str(script), "status", "--port", str(port)],
                    cwd=repository_root,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(status.returncode, 0, status.stderr)
                self.assertIn(f"Track-Viz is running at http://127.0.0.1:{port}", status.stdout)
            finally:
                stopped = subprocess.run(
                    [str(script), "stop", "--port", str(port)],
                    cwd=repository_root,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            self.assertEqual(stopped.returncode, 0, stopped.stderr)
            self.assertIn(f"Track-Viz stopped on port {port}.", stopped.stdout)

    def test_stop_does_not_terminate_an_unrelated_process_on_the_port(self) -> None:
        repository_root = Path(__file__).resolve().parents[3]
        script = repository_root / "track-viz" / "scripts" / "manage_viewer.sh"
        python = repository_root / ".venv" / "bin" / "python"
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        unrelated = subprocess.Popen(
            [str(python), "-m", "http.server", str(port), "--bind", "127.0.0.1"],
            cwd=repository_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            for _ in range(50):
                with socket.socket() as client:
                    if client.connect_ex(("127.0.0.1", port)) == 0:
                        break
                self.assertIsNone(unrelated.poll())
                time.sleep(0.02)
            else:
                self.fail("unrelated test server did not start")
            stopped = subprocess.run(
                [str(script), "stop", "--port", str(port)],
                cwd=repository_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(stopped.returncode, 1)
            self.assertIn("occupied by a non-Track-Viz process", stopped.stderr)
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
