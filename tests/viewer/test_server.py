from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from mot20.viewer.server import build_app, main


class ViewerServerTest(unittest.TestCase):
    def test_build_app_loads_registry_from_fixed_repository_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "configs" / "viewer.toml"
            config_path.parent.mkdir()
            config_path.write_text("sources = []\n", encoding="utf-8")

            app = build_app(repository_root=root)
            response = TestClient(app, base_url="http://127.0.0.1").get("/api/health")

        self.assertEqual(
            response.json(),
            {"status": "ok", "source_count": 0, "unavailable_count": 0},
        )

    def test_build_app_registers_internal_track_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "configs" / "viewer.toml"
            config_path.parent.mkdir()
            config_path.write_text("sources = []\n", encoding="utf-8")

            client = TestClient(
                build_app(repository_root=root),
                base_url="http://127.0.0.1",
            )
            responses = (
                client.get("/api/sequences/not-present/tracks/1"),
                client.get(
                    "/api/sequences/not-present/tracks/1/filmstrip?current_row_index=1"
                ),
                client.get(
                    "/api/sequences/not-present/observations/1/crop?source_hash=missing"
                ),
                client.get(
                    "/api/sequences/not-present/tracks/1/context?frame=1"
                ),
                client.get(
                    "/api/sequences/not-present/tracks/1/events"
                ),
                client.post(
                    "/api/sequences/not-present/exports",
                    headers={"Origin": "http://127.0.0.1:8000"},
                    json={
                        "source_hash": "missing",
                        "track_id": 1,
                        "start_frame": 1,
                        "end_frame": 1,
                    },
                ),
            )
            colors = client.get("/api/contracts/track-colors")

        for response in responses:
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json()["error"]["code"], "unknown_source")
        self.assertEqual(colors.status_code, 200)
        self.assertEqual(colors.json()["version"], "fnv1a32-hsv-integer-v1")

    @patch("mot20.viewer.server.uvicorn.run")
    @patch("mot20.viewer.server.build_app")
    def test_main_uses_loopback_defaults(self, app_builder: Mock, uvicorn_run: Mock) -> None:
        app = Mock()
        app_builder.return_value = app

        main([])

        app_builder.assert_called_once_with(
            development_origin=None,
            application_origin="http://127.0.0.1:8000",
            trusted_hosts=("127.0.0.1", "localhost"),
        )
        uvicorn_run.assert_called_once_with(app, host="127.0.0.1", port=8000)

    @patch("mot20.viewer.server.uvicorn.run")
    @patch("mot20.viewer.server.build_app")
    def test_main_refuses_implicit_wide_bind_and_warns_for_explicit_bind(
        self,
        app_builder: Mock,
        uvicorn_run: Mock,
    ) -> None:
        with self.assertRaises(SystemExit):
            main(["--host", "0.0.0.0"])
        app_builder.assert_not_called()

        app = Mock()
        app_builder.return_value = app
        with self.assertLogs("mot20.viewer.server", level="WARNING") as logs:
            main(
                [
                    "--host",
                    "0.0.0.0",
                    "--allow-non-loopback",
                    "--trusted-host",
                    "viewer.local",
                    "--dev-origin",
                    "http://127.0.0.1:5173",
                    "--port",
                    "8123",
                ]
            )

        self.assertIn("non-loopback", logs.output[0])
        app_builder.assert_called_once_with(
            development_origin="http://127.0.0.1:5173",
            application_origin="http://127.0.0.1:5173",
            trusted_hosts=("viewer.local",),
        )
        uvicorn_run.assert_called_once_with(app, host="0.0.0.0", port=8123)


if __name__ == "__main__":
    unittest.main()