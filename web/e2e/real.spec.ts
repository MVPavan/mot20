import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { arch, cpus, platform, release, totalmem } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test, type Page } from "@playwright/test";

test.skip(process.env.MOT20_REAL_E2E !== "1", "Requires configured local MOT20 sources");

const REPOSITORY_ROOT = fileURLToPath(new URL("../..", import.meta.url));
const REPORT_ROOT = path.join(REPOSITORY_ROOT, "artifacts", "viewer", "verification");

interface SourceMetadata {
  source_key: string;
  source_hash: string;
  width: number;
  height: number;
}

function percentile(samples: readonly number[], percentileValue: number): number {
  const sorted = [...samples].sort((left, right) => left - right);
  return sorted[Math.ceil(sorted.length * percentileValue) - 1];
}

function browserErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });
  page.on("requestfailed", (request) => {
    if (request.failure()?.errorText !== "net::ERR_ABORTED") errors.push(`request: ${request.url()}`);
  });
  page.on("response", (response) => {
    if (response.status() >= 400) errors.push(`response ${response.status()}: ${response.url()}`);
  });
  return errors;
}

async function sourceMetadata(page: Page, sourceKey: string): Promise<SourceMetadata> {
  const response = await page.request.get("/api/sequences");
  expect(response.ok()).toBe(true);
  const body = await response.json() as { sources: SourceMetadata[] };
  const source = body.sources.find((candidate) => candidate.source_key === sourceKey);
  expect(source).toBeDefined();
  return source!;
}

async function imagePoint(page: Page, source: SourceMetadata, imageX: number, imageY: number) {
  return page.getByTestId("frame-viewport").evaluate((viewport, input) => {
    const canvas = viewport.querySelector<HTMLCanvasElement>('[data-layer="overlay"]')!;
    const bounds = canvas.getBoundingClientRect();
    const [x, y, width, height] = (viewport.getAttribute("data-image-rect") ?? "").split(",").map(Number);
    const dpr = canvas.width / bounds.width;
    return {
      x: bounds.left + x / dpr + (input.imageX / input.sourceWidth) * (width / dpr),
      y: bounds.top + y / dpr + (input.imageY / input.sourceHeight) * (height / dpr),
    };
  }, { imageX, imageY, sourceWidth: source.width, sourceHeight: source.height });
}

async function expectRenderedFrame(page: Page): Promise<void> {
  await expect(page.getByTestId("frame-viewport")).toHaveAttribute("data-image-rect", /.+/);
  await expect(page.getByRole("status", { name: /loading exact frame/i })).toHaveCount(0);
  const pixels = await page.locator('[data-layer="image"]').evaluate((canvas: HTMLCanvasElement) => {
    const context = canvas.getContext("2d")!;
    const sample = context.getImageData(
      Math.floor(canvas.width * 0.25),
      Math.floor(canvas.height * 0.25),
      Math.max(1, Math.floor(canvas.width * 0.5)),
      Math.max(1, Math.floor(canvas.height * 0.5)),
    ).data;
    return sample.some((value, index) => index % 4 !== 3 && value > 0);
  });
  expect(pixels).toBe(true);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
    await page.evaluate(() => window.innerWidth),
  );
}

async function seekAndMeasure(page: Page, frame: number) {
  return page.evaluate((nextFrame) => new Promise<{
    durationMs: number;
    cacheSize: number;
    cacheCapacity: number;
    closedCount: number;
    heapBytes: number | null;
  }>((resolve) => {
    const input = document.querySelector<HTMLInputElement>('#frame-number')!;
    const viewport = document.querySelector<HTMLElement>('[data-testid="frame-viewport"]')!;
    const previousDrawCount = viewport.dataset.imageDrawCount;
    const started = performance.now();
    const observer = new MutationObserver(() => {
      if (viewport.dataset.imageDrawCount === previousDrawCount) return;
      observer.disconnect();
      requestAnimationFrame(() => {
        const memory = performance as Performance & { memory?: { usedJSHeapSize: number } };
        resolve({
          durationMs: performance.now() - started,
          cacheSize: Number(viewport.dataset.bitmapCacheSize),
          cacheCapacity: Number(viewport.dataset.bitmapCacheCapacity),
          closedCount: Number(viewport.dataset.bitmapCacheClosedCount),
          heapBytes: memory.memory?.usedJSHeapSize ?? null,
        });
      });
    });
    observer.observe(viewport, { attributes: true, attributeFilter: ["data-image-draw-count"] });
    const valueSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
    if (valueSetter === undefined) throw new Error("HTML input value setter is unavailable");
    valueSetter.call(input, String(nextFrame));
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }), frame);
}

test("real 06/08 detection ambiguity journeys remain aligned and track-disabled", async ({ page }, testInfo) => {
  const errors = browserErrors(page);
  const journeys = [
    { key: "mot20-06-joco", frame: 522, x: 1426.23, y: 378.825 },
    { key: "mot20-08-joco", frame: 29, x: 686.76, y: 196.53 },
  ];
  await page.goto("/");
  for (const journey of journeys) {
    const source = await sourceMetadata(page, journey.key);
    await page.getByLabel("Source", { exact: true }).selectOption(journey.key);
    await page.getByLabel("Frame number").fill(String(journey.frame));
    await expectRenderedFrame(page);
    await expect(page.getByText("Loading observations")).toHaveCount(0);
    const point = await imagePoint(page, source, journey.x, journey.y);
    await page.mouse.move(point.x, point.y);
    await expect(page.locator(".frame-stage")).toHaveAttribute("data-candidate-count", "5");
    await page.mouse.wheel(0, 100);
    await page.mouse.click(point.x, point.y);
    await expect(page.locator(".frame-stage")).toHaveAttribute("data-selection-mode", "pinned");
    await page.locator(".candidate-chooser").getByRole("option").first().click();
    await expect(page.locator(".frame-stage")).toHaveAttribute("data-selection-mode", "confirmed");
    await expect(page.getByRole("button", { name: "Follow" })).toBeDisabled();
    await page.screenshot({ path: testInfo.outputPath(`${journey.key}.png`), fullPage: true });
  }
  expect(errors).toEqual([]);
});

test("real MOT20-01 Focus, timeline, filmstrip, Context, and export readback remain coherent", async ({ page }, testInfo) => {
  const errors = browserErrors(page);
  const source = await sourceMetadata(page, "mot20-01-gt");
  await page.goto("/");
  await page.getByLabel("Source", { exact: true }).selectOption(source.source_key);
  await page.getByLabel("Frame number").fill("404");
  await page.getByLabel("Exact track ID").fill("72");
  await page.getByRole("button", { name: "Find track" }).click();
  await expect(page.getByRole("region", { name: "Focus review for track 72" })).toBeVisible();
  await expect(page.getByLabel("Track timeline")).toBeVisible();
  await expect(page.getByLabel(/Track filmstrip/)).toBeVisible();
  await expect(page.getByText(/Observed on exact frame 404/)).toBeVisible();
  await page.getByRole("radio", { name: "Context" }).check();
  await expect(page.locator('[data-layer="overlay"]')).toHaveAttribute("data-context-commands", "3");
  await expectRenderedFrame(page);
  await page.screenshot({ path: testInfo.outputPath("mot20-01-focus-context.png"), fullPage: true });

  const exportResponse = await page.request.post(`/api/sequences/${source.source_key}/exports`, {
    headers: { Origin: "http://127.0.0.1:4180" },
    data: {
      source_hash: source.source_hash,
      track_id: 72,
      start_frame: 404,
      end_frame: 413,
      context_count: 3,
      trace_length: 30,
    },
  });
  expect(exportResponse.ok()).toBe(true);
  const exported = await exportResponse.json() as {
    export_id: string;
    video_path: string;
    metadata_path: string;
  };
  const video = await readFile(path.join(REPOSITORY_ROOT, exported.video_path));
  const metadata = JSON.parse(await readFile(path.join(REPOSITORY_ROOT, exported.metadata_path), "utf8")) as {
    export_id: string;
    output: { sha256: string };
  };
  expect(video.byteLength).toBeGreaterThan(0);
  expect(metadata.export_id).toBe(exported.export_id);
  expect(createHash("sha256").update(video).digest("hex")).toBe(metadata.output.sha256);
  expect(errors).toEqual([]);
});

test("dense pointer latency, seek budgets, ETag, and 500-frame cache ceilings are measured", async ({ page, browser }, testInfo) => {
  test.skip(testInfo.project.name !== "real-desktop-chromium", "Performance acceptance uses the desktop display");
  const errors = browserErrors(page);
  const source = await sourceMetadata(page, "mot20-06-joco");
  await page.goto("/");
  await page.getByLabel("Source", { exact: true }).selectOption(source.source_key);
  await page.getByLabel("Frame number").fill("522");
  await expectRenderedFrame(page);
  await expect(page.getByText("Loading observations")).toHaveCount(0);

  const pointerRuns = await page.getByTestId("frame-viewport").evaluate(async (viewport, input) => {
    const canvas = viewport.querySelector<HTMLCanvasElement>('[data-layer="overlay"]')!;
    const bounds = canvas.getBoundingClientRect();
    const [x, y, width, height] = (viewport.getAttribute("data-image-rect") ?? "").split(",").map(Number);
    const dpr = canvas.width / bounds.width;
    const clientX = bounds.left + x / dpr + (input.imageX / input.sourceWidth) * (width / dpr);
    const clientY = bounds.top + y / dpr + (input.imageY / input.sourceHeight) * (height / dpr);
    const sample = (offset: number) => new Promise<number>((resolve) => {
      window.addEventListener("mot20-viewer:pointer-latency", (event) => {
        resolve((event as CustomEvent<{ durationMs: number }>).detail.durationMs);
      }, { once: true });
      canvas.dispatchEvent(new PointerEvent("pointermove", {
        bubbles: true,
        clientX: clientX + offset,
        clientY,
        pointerId: 1,
        pointerType: "mouse",
      }));
    });
    for (let index = 0; index < 120; index += 1) await sample(index % 2);
    const runs: number[][] = [];
    for (let run = 0; run < 3; run += 1) {
      const samples: number[] = [];
      for (let index = 0; index < 1000; index += 1) samples.push(await sample(index % 2));
      runs.push(samples);
    }
    return runs;
  }, { imageX: 1426.23, imageY: 378.825, sourceWidth: source.width, sourceHeight: source.height });
  const pointer = pointerRuns.map((samples, index) => ({
    run: index + 1,
    samples: samples.length,
    p50Ms: percentile(samples, 0.5),
    p95Ms: percentile(samples, 0.95),
  }));
  pointer.forEach((run) => expect(run.p95Ms).toBeLessThan(50));

  const coldFrames = Array.from({ length: 30 }, (_, index) => 600 + index * 5);
  const cold = [];
  for (const frame of coldFrames) cold.push(await seekAndMeasure(page, frame));
  const warm = [];
  for (const frame of [...coldFrames].reverse().slice(1)) warm.push(await seekAndMeasure(page, frame));

  const scrub = [];
  for (let frame = 1; frame <= 500; frame += 1) scrub.push(await seekAndMeasure(page, frame));
  const cacheCapacity = Math.max(...scrub.map((sample) => sample.cacheCapacity));
  const bitmapCountCeiling = Math.max(...scrub.map((sample) => sample.cacheSize));
  const closedBitmapCeiling = Math.max(...scrub.map((sample) => sample.closedCount));
  const heapSamples = scrub.flatMap((sample) => sample.heapBytes === null ? [] : [sample.heapBytes]);
  expect(cacheCapacity).toBe(150);
  expect(bitmapCountCeiling).toBeLessThanOrEqual(cacheCapacity);
  expect(closedBitmapCeiling).toBeGreaterThan(0);

  const frameUrl = `/api/sequences/${source.source_key}/frames/522?source_hash=${source.source_hash}`;
  const firstFrame = await page.request.get(frameUrl);
  const etag = firstFrame.headers().etag;
  expect(etag).toBeTruthy();
  const conditionalFrame = await page.request.get(frameUrl, { headers: { "If-None-Match": etag } });
  expect(conditionalFrame.status()).toBe(304);

  const screen = await page.evaluate(() => ({
    width: window.screen.width,
    height: window.screen.height,
    viewportWidth: window.innerWidth,
    viewportHeight: window.innerHeight,
    devicePixelRatio: window.devicePixelRatio,
    userAgent: navigator.userAgent,
  }));
  const report = {
    generatedAt: new Date().toISOString(),
    source: { key: source.source_key, frame: 522, observations: 161 },
    pointerMeasurement: {
      span: "pointermove event.timeStamp through completed overlay requestAnimationFrame draw",
      warmupSamples: 120,
      runs: pointer,
      requiredP95Ms: 50,
    },
    seekToRenderedFrame: {
      sampleCount: cold.length,
      coldP50Ms: percentile(cold.map((sample) => sample.durationMs), 0.5),
      coldP95Ms: percentile(cold.map((sample) => sample.durationMs), 0.95),
      warmP50Ms: percentile(warm.map((sample) => sample.durationMs), 0.5),
      warmP95Ms: percentile(warm.map((sample) => sample.durationMs), 0.95),
      threshold: null,
    },
    scrub: {
      frames: 500,
      configuredCacheBounds: { minimum: 100, maximum: 200, default: 150 },
      cacheCapacity,
      bitmapCountCeiling,
      closedBitmapCeiling,
      observedJsHeapBytesCeiling: heapSamples.length === 0 ? null : Math.max(...heapSamples),
    },
    httpCache: { etag, conditionalStatus: conditionalFrame.status() },
    environment: {
      cpu: cpus()[0]?.model ?? "unknown",
      logicalCpuCount: cpus().length,
      totalRamBytes: totalmem(),
      os: `${platform()} ${release()} ${arch()}`,
      browser: `Chromium ${browser?.version() ?? "unknown"}`,
      display: `${screen.width}x${screen.height}`,
      viewport: `${screen.viewportWidth}x${screen.viewportHeight}`,
      devicePixelRatio: screen.devicePixelRatio,
      userAgent: screen.userAgent,
    },
  };
  await mkdir(REPORT_ROOT, { recursive: true });
  await writeFile(path.join(REPORT_ROOT, "browser-performance.json"), `${JSON.stringify(report, null, 2)}\n`);
  await testInfo.attach("browser-performance", { body: JSON.stringify(report, null, 2), contentType: "application/json" });
  expect(errors).toEqual([]);
});