import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  testMatch: "real.spec.ts",
  outputDir: "./test-results/real",
  reporter: [["html", { outputFolder: "playwright-report/real", open: "never" }], ["list"]],
  timeout: 600_000,
  expect: { timeout: 15_000 },
  use: {
    baseURL: "http://127.0.0.1:4180",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "real-desktop-chromium",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 1000 } },
    },
    {
      name: "real-narrow-chromium",
      use: { ...devices["Desktop Chrome"], viewport: { width: 390, height: 844 } },
    },
  ],
  webServer: {
    command: "PYTHONPATH=../src ../../.venv/bin/python ../scripts/run_viewer.py --config track-viz/configs/viewer.toml --port 4180",
    url: "http://127.0.0.1:4180/api/health",
    reuseExistingServer: false,
    timeout: 120_000,
  },
});