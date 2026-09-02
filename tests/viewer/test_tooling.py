from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


class ToolingContractTest(unittest.TestCase):
    def test_generated_environment_artifact_and_frontend_paths_are_ignored(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        generated_paths = (
            ".venv/bin/python",
            "artifacts/viewer/cache/item.jpg",
            "web/node_modules/package/index.js",
            "web/dist/index.html",
            "web/coverage/index.html",
            "web/test-results/result.json",
            "web/playwright-report/index.html",
            "web/.playwright/browser/chrome",
            "web/.vite/deps/module.js",
        )
        for generated_path in generated_paths:
            with self.subTest(path=generated_path):
                result = subprocess.run(
                    ["git", "check-ignore", "--quiet", generated_path],
                    cwd=repository_root,
                    check=False,
                )
                self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()