import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page, type TestInfo } from "@playwright/test";

const SOURCE_KEY = "synthetic-test-adapted";
const SOURCE_HASH = "6b6f5ae8386d8f34b56d7c1f0b56ae167dc40018d564235080d3f58bc4251710";
const JPEG_BY_FRAME: Record<number, string> = {
  1: "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA4KCw0LCQ4NDA0QDw4RFiQXFhQUFiwgIRokNC43NjMuMjI6QVNGOj1OPjIySGJJTlZYXV5dOEVmbWVabFNbXVn/2wBDAQ8QEBYTFioXFypZOzI7WVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVn/wAARCAAJABADASIAAhEBAxEB/8QAFgABAQEAAAAAAAAAAAAAAAAAAAQF/8QAHhAAAQQBBQAAAAAAAAAAAAAAAQACESEDBAUSFHH/xAAVAQEBAAAAAAAAAAAAAAAAAAAFBv/EABkRAAIDAQAAAAAAAAAAAAAAAAABAhESQf/aAAwDAQACEQMRAD8AozYdENsD2FnY4tNPkzU1PqzERGRjntlAlR//2Q==",
  2: "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA4KCw0LCQ4NDA0QDw4RFiQXFhQUFiwgIRokNC43NjMuMjI6QVNGOj1OPjIySGJJTlZYXV5dOEVmbWVabFNbXVn/2wBDAQ8QEBYTFioXFypZOzI7WVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVn/wAARCAAJABADASIAAhEBAxEB/8QAFgABAQEAAAAAAAAAAAAAAAAAAAQF/8QAHhAAAQQBBQAAAAAAAAAAAAAAAQACESEDBAUSFHH/xAAVAQEBAAAAAAAAAAAAAAAAAAAFBv/EABkRAAIDAQAAAAAAAAAAAAAAAAABAhESQf/aAAwDAQACEQMRAD8AozYdENsD2FnY4tNPkzU1PqzERGRjntlAlR//2Q==",
  3: "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA4KCw0LCQ4NDA0QDw4RFiQXFhQUFiwgIRokNC43NjMuMjI6QVNGOj1OPjIySGJJTlZYXV5dOEVmbWVabFNbXVn/2wBDAQ8QEBYTFioXFypZOzI7WVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVn/wAARCAAJABADASIAAhEBAxEB/8QAFgABAQEAAAAAAAAAAAAAAAAAAAIF/8QAHRAAAQQDAQEAAAAAAAAAAAAAAQACAwQFERJBcf/EABUBAQEAAAAAAAAAAAAAAAAAAAME/8QAGREAAwEBAQAAAAAAAAAAAAAAAQIRADGh/9oADAMBAAIRAxEAPwCcPXxstR7rrohJ2QOpeTrQ839WIiI1QqxNt807MCAJzf/Z",
  4: "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA4KCw0LCQ4NDA0QDw4RFiQXFhQUFiwgIRokNC43NjMuMjI6QVNGOj1OPjIySGJJTlZYXV5dOEVmbWVabFNbXVn/2wBDAQ8QEBYTFioXFypZOzI7WVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVn/wAARCAAJABADASIAAhEBAxEB/8QAFgABAQEAAAAAAAAAAAAAAAAAAAIF/8QAHRAAAQQDAQEAAAAAAAAAAAAAAQACAwQFERJBcf/EABUBAQEAAAAAAAAAAAAAAAAAAAME/8QAGREAAwEBAQAAAAAAAAAAAAAAAQIRADGh/9oADAMBAAIRAxEAPwCcPXxstR7rrohJ2QOpeTrQ839WIiI1QqxNt807MCAJzf/Z",
  5: "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA4KCw0LCQ4NDA0QDw4RFiQXFhQUFiwgIRokNC43NjMuMjI6QVNGOj1OPjIySGJJTlZYXV5dOEVmbWVabFNbXVn/2wBDAQ8QEBYTFioXFypZOzI7WVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVlZWVn/wAARCAAJABADASIAAhEBAxEB/8QAFgABAQEAAAAAAAAAAAAAAAAABAAD/8QAHxAAAQQCAgMAAAAAAAAAAAAAAQIDBBIAEQUxQXHh/8QAFQEBAQAAAAAAAAAAAAAAAAAABAX/xAAcEQABBQADAAAAAAAAAAAAAAABAAIDEjERQYH/2gAMAwEAAhEDEQA/AD8S3FdlKTMKQ3QkWVUb2PuZ8ihlua4mMUloarVWx0PPvC5YkRkSF9jmdKaXilOPV//Z",
};

interface FixtureObservation {
  source_key: string;
  sequence: string;
  frame: number;
  row_index: number;
  row_hash: string;
  source_hash: string;
  raw_track_id: -1;
  usable_track_id: null;
  raw_geometry: { x: number; y: number; width: number; height: number };
  display_geometry: { x1: number; y1: number; x2: number; y2: number };
  score: number;
  ground_truth: null;
  opaque_result_fields: [-1, -1, -1];
  score_semantics: "tracker_score";
  ground_truth_semantics: "not_defined";
}

function fixtureObservation(
  frame: number,
  rowIndex: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
): FixtureObservation {
  return {
    source_key: SOURCE_KEY,
    sequence: "SYNTHETIC-TEST",
    frame,
    row_index: rowIndex,
    row_hash: String(rowIndex).padStart(64, "0"),
    source_hash: SOURCE_HASH,
    raw_track_id: -1,
    usable_track_id: null,
    raw_geometry: { x: x1, y: y1, width: x2 - x1, height: y2 - y1 },
    display_geometry: { x1, y1, x2, y2 },
    score: 0.9 - rowIndex / 10_000,
    ground_truth: null,
    opaque_result_fields: [-1, -1, -1],
    score_semantics: "tracker_score",
    ground_truth_semantics: "not_defined",
  };
}

const OBSERVATIONS_BY_FRAME: Record<number, FixtureObservation[]> = {
  1: [
    fixtureObservation(1, 101, 120, 35, 220, 165),
    fixtureObservation(1, 102, 125, 40, 215, 160),
    fixtureObservation(1, 103, 130, 45, 210, 155),
    fixtureObservation(1, 104, 135, 50, 205, 150),
    fixtureObservation(1, 105, 140, 55, 200, 145),
    fixtureObservation(1, 106, 145, 60, 195, 140),
    fixtureObservation(1, 107, 150, 65, 190, 135),
  ],
  2: [],
  3: [fixtureObservation(3, 301, 0, 0, 24, 34)],
  4: [fixtureObservation(4, 401, 260, 130, 310, 178)],
  5: [fixtureObservation(5, 501, 20, 20, 60, 80)],
};

const metadata = {
  source_key: SOURCE_KEY,
  sequence: "SYNTHETIC-TEST",
  frame_numbering: "one_based",
  frame_count: 5,
  width: 320,
  height: 180,
  frame_rate: 25,
  source_hash: SOURCE_HASH,
  adapter: "mot_result_10",
  source_class: "tracker_result",
  policy_classification: "local_test_adapted_development_material",
  source_row_count: 10,
  observation_count: 10,
  capability: {
    id_status: "sentinel_only",
    track_features: false,
    usable_track_ids: [],
    diagnostics: [],
  },
  provenance: {
    producer: "deterministic-e2e-fixture",
    detector: "synthetic-color-blocks",
    checkpoint: null,
    tracker: null,
    post_processing: null,
    adaptation_iterations: 2,
    notes: "Browser tracer only",
  },
  diagnostics: [],
};

const secondaryMetadata = {
  ...metadata,
  source_key: "synthetic-secondary",
  source_hash: "a".repeat(64),
  provenance: { ...metadata.provenance, producer: "secondary-e2e-fixture" },
};

function trackedObservation(
  source: { source_key: string; sequence: string; source_hash: string },
  frame: number,
  rowIndex: number,
) {
  return {
    ...fixtureObservation(frame, rowIndex, 80 + frame * 8, 35, 145 + frame * 8, 160),
    source_key: source.source_key,
    sequence: source.sequence,
    source_hash: source.source_hash,
    raw_track_id: 8,
    usable_track_id: 8,
  };
}

const continuousMetadata = {
  ...metadata,
  source_key: "synthetic-tracked-continuous",
  sequence: "SYNTHETIC-CONTINUOUS",
  source_hash: "c".repeat(64),
  capability: { id_status: "tracked", track_features: true, usable_track_ids: [8], diagnostics: [] },
};
const gappedMetadata = {
  ...continuousMetadata,
  source_key: "synthetic-tracked-gapped",
  sequence: "SYNTHETIC-GAPPED",
  source_hash: "d".repeat(64),
};
const TRACKED_OBSERVATIONS = new Map([
  [continuousMetadata.source_key, [
    trackedObservation(continuousMetadata, 1, 1001),
    trackedObservation(continuousMetadata, 2, 1002),
    trackedObservation(continuousMetadata, 3, 1003),
  ]],
  [gappedMetadata.source_key, [
    trackedObservation(gappedMetadata, 1, 2001),
    trackedObservation(gappedMetadata, 3, 2003),
    trackedObservation(gappedMetadata, 5, 2005),
  ]],
]);

async function installTrackedFixtureRoutes(page: Page, requestedFrames: number[]) {
  const sources = [continuousMetadata, gappedMetadata, metadata];
  await page.route("**/api/sequences", (route) => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({ sources, unavailable: [], diagnostics: [] }),
  }));
  await page.route(/\/api\/sequences\/[^/]+\/tracks\/8\/filmstrip(?:\?.*)?$/, (route) => {
    const url = new URL(route.request().url());
    const sourceKey = url.pathname.split("/")[3];
    const source = sources.find((item) => item.source_key === sourceKey)!;
    const observations = TRACKED_OBSERVATIONS.get(sourceKey)!;
    const currentRowIndex = Number(url.searchParams.get("current_row_index"));
    return route.fulfill({ contentType: "application/json", body: JSON.stringify({
      source_key: sourceKey, sequence: source.sequence, source_hash: source.source_hash, track_id: 8,
      current_row_index: currentRowIndex, total_observations: observations.length,
      sampled_count: observations.length,
      samples: observations.map((observation) => ({ is_current: observation.row_index === currentRowIndex, observation })),
    }) });
  });
  await page.route(/\/api\/sequences\/[^/]+\/tracks\/8\/events(?:\?.*)?$/, (route) => {
    const url = new URL(route.request().url());
    const sourceKey = url.pathname.split("/")[3];
    const source = sources.find((item) => item.source_key === sourceKey)!;
    const observations = TRACKED_OBSERVATIONS.get(sourceKey)!;
    const meaningful = sourceKey === gappedMetadata.source_key;
    const displacementEnabled = url.searchParams.get("enable_displacement") === "true";
    const scaleEnabled = url.searchParams.get("enable_scale_change") === "true";
    const interactionEnabled = url.searchParams.get("enable_close_interaction") === "true";
    return route.fulfill({ contentType: "application/json", body: JSON.stringify({
      source_key: sourceKey, sequence: source.sequence, source_hash: source.source_hash, track_id: 8,
      geometry_basis: "raw_xywh",
      settings: {
        displacement_enabled: displacementEnabled, displacement_threshold: Number(url.searchParams.get("displacement_threshold") ?? 0.5), displacement_operator: "greater_than_or_equal",
        scale_change_enabled: scaleEnabled, scale_change_threshold: Number(url.searchParams.get("scale_change_threshold") ?? 0.5), scale_change_operator: "greater_than_or_equal",
        close_interaction_enabled: interactionEnabled, close_interaction_threshold: Number(url.searchParams.get("close_interaction_threshold") ?? 0.25), close_interaction_operator: "less_than_or_equal",
      },
      confidence: { status: meaningful ? "meaningful" : "constant", meaningful, score_semantics: "tracker_score", threshold: 0.85, threshold_operator: "less_than_or_equal", diagnostic: null },
      displacement_events: displacementEnabled ? [{ from_frame: 1, to_frame: 2, threshold: 0.5 }] : [],
      scale_change_events: [], close_interaction_events: [],
      low_confidence_observations: meaningful ? [observations[1]] : [],
    }) });
  });
  await page.route(/\/api\/sequences\/[^/]+\/tracks\/8\/context(?:\?.*)?$/, (route) => {
    const url = new URL(route.request().url());
    const sourceKey = url.pathname.split("/")[3];
    const source = sources.find((item) => item.source_key === sourceKey)!;
    const frame = Number(url.searchParams.get("frame"));
    const focal = TRACKED_OBSERVATIONS.get(sourceKey)!.find((item) => item.frame === frame)!;
    const count = Number(url.searchParams.get("count"));
    const competitors = Array.from({ length: 8 }, (_, index) => ({
      rank: index + 1,
      track_id: index + 9,
      best_iou: 0.2 - index / 100,
      best_normalized_edge_proximity: 0.1 + index / 100,
      comparison_count: 1,
      evidence: [{
        frame,
        focal_row_index: focal.row_index,
        competitor_row_index: 5000 + index,
        focal_raw_xywh: [focal.raw_geometry.x, focal.raw_geometry.y, focal.raw_geometry.width, focal.raw_geometry.height],
        competitor_raw_xywh: [130 + index * 8, 30 + index * 4, 58, 132],
        iou: 0.2 - index / 100,
        edge_distance_pixels: index,
        focal_box_height: focal.raw_geometry.height,
        normalized_edge_proximity: 0.1 + index / 100,
      }],
    }));
    return route.fulfill({ contentType: "application/json", body: JSON.stringify({
      source_key: sourceKey, sequence: source.sequence, source_hash: source.source_hash, track_id: 8,
      geometry_basis: "raw_xywh", window: { center_frame: frame, start_frame: Math.max(1, frame - 3), end_frame: Math.min(5, frame + 3), radius: 3 },
      requested_count: count, hard_cap: 8, total_competitors: competitors.length,
      competitors: competitors.slice(0, count),
    }) });
  });
  await page.route(/\/api\/sequences\/[^/]+\/tracks\/8(?:\?.*)?$/, (route) => {
    const url = new URL(route.request().url());
    const sourceKey = url.pathname.split("/")[3];
    const source = sources.find((item) => item.source_key === sourceKey)!;
    const observations = TRACKED_OBSERVATIONS.get(sourceKey)!;
    const currentRowIndex = Number(url.searchParams.get("current_row_index"));
    const currentPosition = observations.findIndex((item) => item.row_index === currentRowIndex);
    const gaps = sourceKey === gappedMetadata.source_key
      ? [{ start_frame: 2, end_frame: 2, length: 1 }, { start_frame: 4, end_frame: 4, length: 1 }]
      : [];
    return route.fulfill({ contentType: "application/json", body: JSON.stringify({
      source_key: sourceKey, sequence: source.sequence, source_hash: source.source_hash, track_id: 8,
      observation_frames: observations.map((item) => item.frame), gaps,
      first_observation: observations[0], last_observation: observations.at(-1),
      previous_observation: currentPosition > 0 ? observations[currentPosition - 1] : null,
      next_observation: currentPosition >= 0 && currentPosition + 1 < observations.length ? observations[currentPosition + 1] : null,
      observations,
    }) });
  });
  await page.route(/\/api\/sequences\/[^/]+\/tracks\?track_id=8(?:&.*)?$/, (route) => {
    const url = new URL(route.request().url());
    const sourceKey = url.pathname.split("/")[3];
    const source = sources.find((item) => item.source_key === sourceKey)!;
    const observations = TRACKED_OBSERVATIONS.get(sourceKey)!;
    return route.fulfill({ contentType: "application/json", body: JSON.stringify({
      source_key: sourceKey, sequence: source.sequence, source_hash: source.source_hash, track_id: 8,
      observation_frames: observations.map((item) => item.frame), gaps: [],
      first_observation: observations[0], last_observation: observations.at(-1),
      previous_observation: null, next_observation: null, observations,
    }) });
  });
  await page.route(/\/api\/sequences\/[^/]+\/frames\/\d+\/observations(?:\?.*)?$/, (route) => {
    const url = new URL(route.request().url());
    const sourceKey = url.pathname.split("/")[3];
    const frame = Number(url.pathname.split("/")[5]);
    const source = sources.find((item) => item.source_key === sourceKey)!;
    const tracked = TRACKED_OBSERVATIONS.get(sourceKey)?.filter((item) => item.frame === frame) ?? [];
    const unrelated = sourceKey === metadata.source_key ? (OBSERVATIONS_BY_FRAME[frame] ?? []) : [
      { ...fixtureObservation(frame, 9000 + frame, 200, 30, 250, 140), source_key: sourceKey, source_hash: source.source_hash, sequence: source.sequence },
    ];
    return route.fulfill({ contentType: "application/json", body: JSON.stringify({
      source_key: sourceKey, sequence: source.sequence, frame, frame_numbering: "one_based",
      source_hash: source.source_hash, observations: [...tracked, ...unrelated],
    }) });
  });
  await page.route(/\/api\/sequences\/[^/]+\/observations\/\d+\/crop(?:\?.*)?$/, (route) => {
    const url = new URL(route.request().url());
    const sourceKey = url.pathname.split("/")[3];
    const rowIndex = Number(url.pathname.split("/")[5]);
    const source = sources.find((item) => item.source_key === sourceKey)!;
    const observation = [...TRACKED_OBSERVATIONS.values()].flat().find((item) => item.row_index === rowIndex)
      ?? Object.values(OBSERVATIONS_BY_FRAME).flat().find((item) => item.row_index === rowIndex)!;
    return route.fulfill({ contentType: "application/json", body: JSON.stringify({
      source_key: sourceKey, sequence: source.sequence, source_hash: source.source_hash,
      frame: observation.frame, row_index: rowIndex, row_hash: observation.row_hash,
      media_type: "image/jpeg", image_base64: JPEG_BY_FRAME[observation.frame],
    }) });
  });
  await page.route(/\/api\/sequences\/[^/]+\/frames\/\d+(?:\?.*)?$/, (route) => {
    const url = new URL(route.request().url());
    const sourceKey = url.pathname.split("/")[3];
    const frame = Number(url.pathname.split("/")[5]);
    const source = sources.find((item) => item.source_key === sourceKey)!;
    expect(url.searchParams.get("source_hash")).toBe(source.source_hash);
    requestedFrames.push(frame);
    return route.fulfill({ contentType: "image/jpeg", body: Buffer.from(JPEG_BY_FRAME[frame], "base64") });
  });
}

async function installFixtureRoutes(page: Page, requestedFrames: number[]) {
  await page.route("**/api/sequences", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ sources: [metadata, secondaryMetadata], unavailable: [], diagnostics: [] }),
    }),
  );
  await page.route(/\/api\/sequences\/[^/]+\/frames\/\d+\/observations(?:\?.*)?$/, (route) => {
    const url = new URL(route.request().url());
    const parts = url.pathname.split("/");
    const sourceKey = parts[parts.indexOf("sequences") + 1];
    const frame = Number(parts[parts.indexOf("frames") + 1]);
    const source = sourceKey === SOURCE_KEY ? metadata : secondaryMetadata;
    const observations = (OBSERVATIONS_BY_FRAME[frame] ?? []).map((observation) => ({
      ...observation,
      source_key: source.source_key,
      source_hash: source.source_hash,
    }));
    return route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        source_key: source.source_key,
        sequence: source.sequence,
        frame,
        frame_numbering: "one_based",
        source_hash: source.source_hash,
        observations,
      }),
    });
  });
  await page.route(/\/api\/sequences\/[^/]+\/observations\/\d+\/crop(?:\?.*)?$/, (route) => {
    const url = new URL(route.request().url());
    const parts = url.pathname.split("/");
    const sourceKey = parts[parts.indexOf("sequences") + 1];
    const rowIndex = Number(parts[parts.indexOf("observations") + 1]);
    const source = sourceKey === SOURCE_KEY ? metadata : secondaryMetadata;
    const observation = Object.values(OBSERVATIONS_BY_FRAME).flat().find((row) => row.row_index === rowIndex)!;
    return route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        source_key: source.source_key,
        sequence: source.sequence,
        source_hash: source.source_hash,
        frame: observation.frame,
        row_index: rowIndex,
        row_hash: observation.row_hash,
        media_type: "image/jpeg",
        image_base64: JPEG_BY_FRAME[observation.frame],
      }),
    });
  });
  await page.route(/\/api\/sequences\/[^/]+\/frames\/\d+(?:\?.*)?$/, (route) => {
    const url = new URL(route.request().url());
    const parts = url.pathname.split("/");
    const sourceKey = parts[parts.indexOf("sequences") + 1];
    const frame = Number(parts[parts.indexOf("frames") + 1]);
    const source = sourceKey === SOURCE_KEY ? metadata : secondaryMetadata;
    expect(url.searchParams.get("source_hash")).toBe(source.source_hash);
    requestedFrames.push(frame);
    const body = JPEG_BY_FRAME[frame];
    if (body === undefined) {
      return route.fulfill({ status: 404, body: "unknown synthetic frame" });
    }
    return route.fulfill({
      contentType: "image/jpeg",
      body: Buffer.from(body, "base64"),
    });
  });
}

function collectBrowserErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });
  page.on("requestfailed", (request) => {
    if (request.failure()?.errorText !== "net::ERR_ABORTED") {
      errors.push(`request: ${request.url()}`);
    }
  });
  page.on("response", (response) => {
    if (response.status() >= 400) errors.push(`response ${response.status()}: ${response.url()}`);
  });
  return errors;
}

async function imagePoint(page: Page, imageX: number, imageY: number) {
  const viewport = page.getByTestId("frame-viewport");
  const geometry = await viewport.evaluate((element, point) => {
    const canvas = element.querySelector<HTMLCanvasElement>('[data-layer="overlay"]')!;
    const bounds = canvas.getBoundingClientRect();
    const [x, y, width, height] = (element.getAttribute("data-image-rect") ?? "").split(",").map(Number);
    const dpr = canvas.width / bounds.width;
    return {
      x: bounds.left + x / dpr + (point.imageX / 320) * (width / dpr),
      y: bounds.top + y / dpr + (point.imageY / 180) * (height / dpr),
    };
  }, { imageX, imageY });
  return geometry;
}

async function assertCanvasGeometryAndPixels(page: Page) {
  const result = await page.locator('[data-testid="frame-viewport"]').evaluate((viewport) => {
    const imageCanvas = viewport.querySelector<HTMLCanvasElement>('[data-layer="image"]')!;
    const overlayCanvas = viewport.querySelector<HTMLCanvasElement>('[data-layer="overlay"]')!;
    const imageContext = imageCanvas.getContext("2d")!;
    const overlayContext = overlayCanvas.getContext("2d")!;
    const rectangle = (viewport.getAttribute("data-image-rect") ?? "").split(",").map(Number);
    const [x, y, width, height] = rectangle;
    const center = imageContext.getImageData(
      Math.floor(x + width / 2),
      Math.floor(y + height / 2),
      1,
      1,
    ).data;
    const letterbox = imageContext.getImageData(1, 1, 1, 1).data;
    const overlay = overlayContext.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height).data;
    return {
      canvasWidth: imageCanvas.width,
      canvasHeight: imageCanvas.height,
      imageAspect: width / height,
      imageInsideCanvas: x >= 0 && y >= 0 && x + width <= imageCanvas.width && y + height <= imageCanvas.height,
      centerRgb: Array.from(center.slice(0, 3)),
      letterboxRgb: Array.from(letterbox.slice(0, 3)),
      overlayAlphaSum: overlay.reduce((sum, value, index) => sum + (index % 4 === 3 ? value : 0), 0),
    };
  });

  expect(result.canvasWidth).toBeGreaterThan(0);
  expect(result.canvasHeight).toBeGreaterThan(0);
  expect(result.imageAspect).toBeCloseTo(16 / 9, 5);
  expect(result.imageInsideCanvas).toBe(true);
  expect(result.centerRgb.some((channel) => channel > 30)).toBe(true);
  expect(result.letterboxRgb).toEqual([17, 23, 25]);
  expect(result.overlayAlphaSum).toBe(0);
}

async function overlayAlphaSum(page: Page): Promise<number> {
  return page.locator('[data-layer="overlay"]').evaluate((canvas: HTMLCanvasElement) => {
    const context = canvas.getContext("2d")!;
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    return pixels.reduce((sum, value, index) => sum + (index % 4 === 3 ? value : 0), 0);
  });
}

async function assertNoHorizontalOverflow(page: Page): Promise<void> {
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
    await page.evaluate(() => window.innerWidth),
  );
}

async function assertNoSeriousAccessibilityViolations(page: Page): Promise<void> {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  const violations = results.violations.filter(
    (violation) => violation.impact === "serious" || violation.impact === "critical",
  );
  expect(violations).toEqual([]);
}

test("selects a source and renders exact first, middle, and last JPEGs", async ({ page }, testInfo: TestInfo) => {
  const requestedFrames: number[] = [];
  const errors = collectBrowserErrors(page);
  await installFixtureRoutes(page, requestedFrames);

  await page.goto("/");
  await page.getByLabel("Source", { exact: true }).selectOption(SOURCE_KEY);
  await expect(page.getByText(/local test-adapted development material/i)).toBeVisible();
  await expect(page.getByText(/not a held-out benchmark result/i)).toBeVisible();

  for (const frame of [1, 3, 5]) {
    await page.getByLabel("Frame number").fill(String(frame));
    await expect(page.getByTestId("frame-viewport")).toHaveAttribute("data-image-rect", /.+/);
    await expect(page.getByRole("status", { name: /loading exact frame/i })).toHaveCount(0);
    await assertCanvasGeometryAndPixels(page);
  }

  await page.screenshot({ path: testInfo.outputPath(`${testInfo.project.name}-viewer.png`), fullPage: true });
  expect(requestedFrames).toEqual(expect.arrayContaining([1, 2, 3, 4, 5]));
  expect(errors).toEqual([]);
});

test("cycles, pins, reaches every candidate, confirms without Follow, and resets sources", async ({ page }, testInfo) => {
  const requestedFrames: number[] = [];
  const errors = collectBrowserErrors(page);
  await installFixtureRoutes(page, requestedFrames);
  await page.goto("/");
  await page.getByLabel("Source", { exact: true }).selectOption(SOURCE_KEY);
  const viewport = page.getByTestId("frame-viewport");
  const overlay = viewport.locator('[data-layer="overlay"]');
  await expect(viewport).toHaveAttribute("data-image-rect", /.+/);
  await expect(page.getByText("Loading observations")).toHaveCount(0);
  await expect(overlay).toHaveAttribute("data-overlay-strokes", "0");
  await assertCanvasGeometryAndPixels(page);

  const overlap = await imagePoint(page, 160, 90);
  const pointerLatency = page.evaluate(() => new Promise<number>((resolve) => {
    window.addEventListener("mot20-viewer:pointer-latency", (event) => {
      resolve((event as CustomEvent<{ durationMs: number }>).detail.durationMs);
    }, { once: true });
  }));
  await page.mouse.move(overlap.x, overlap.y);
  await expect(page.locator(".frame-stage")).toHaveAttribute("data-candidate-count", "7");
  await expect(page.locator(".frame-stage")).toHaveAttribute("data-active-row", "107");
  await expect(overlay).toHaveAttribute("data-overlay-strokes", "2");
  await expect(pointerLatency).resolves.toBeGreaterThanOrEqual(0);
  await expect(overlay).toHaveAttribute("data-last-pointer-latency-ms", /\d+(?:\.\d+)?/);
  const imageDrawCount = await viewport.getAttribute("data-image-draw-count");
  const scrollBeforeCycle = await page.evaluate(() => scrollY);
  await page.mouse.wheel(0, 100);
  await expect(page.locator(".frame-stage")).toHaveAttribute("data-active-row", "106");
  expect(await page.evaluate(() => scrollY)).toBe(scrollBeforeCycle);
  expect(await viewport.getAttribute("data-image-draw-count")).toBe(imageDrawCount);

  await overlay.focus();
  await page.keyboard.down("b");
  await expect(overlay).toHaveAttribute("data-overlay-commands", "8");
  await page.keyboard.up("b");
  await expect(overlay).toHaveAttribute("data-overlay-commands", "2");
  expect(await viewport.getAttribute("data-image-draw-count")).toBe(imageDrawCount);

  const pinPoint = await imagePoint(page, 160, 90);
  await page.mouse.click(pinPoint.x, pinPoint.y);
  await expect(page.locator(".frame-stage")).toHaveAttribute("data-selection-mode", "pinned");
  const cards = page.locator(".candidate-chooser").getByRole("option");
  await expect(cards).toHaveCount(7);
  await expect(page.locator(".candidate-card__crop")).toHaveCount(7);
  await expect(page.getByText("Current frame only")).toHaveCount(7);
  const visibleCardCount = await page.locator(".candidate-chooser__list").evaluate((list) => {
    const bounds = list.getBoundingClientRect();
    return Array.from(list.children).filter((child) => {
      const childBounds = child.getBoundingClientRect();
      return childBounds.top >= bounds.top && childBounds.bottom <= bounds.bottom;
    }).length;
  });
  expect(visibleCardCount).toBeLessThanOrEqual(5);
  expect(await page.locator(".candidate-chooser__list").evaluate((list) => list.scrollHeight > list.clientHeight)).toBe(true);

  await cards.first().focus();
  await page.keyboard.press("Tab");
  await page.keyboard.press("Tab");
  await expect(page.locator(".frame-stage")).toHaveAttribute("data-active-row", "105");
  await page.keyboard.press("Enter");
  await expect(page.locator(".frame-stage")).toHaveAttribute("data-selection-mode", "confirmed");
  await expect(page.getByRole("button", { name: "Follow" })).toBeDisabled();
  await expect(page.locator("#follow-disabled-reason")).toContainText(
    "sentinel-only IDs and no stable track identity",
  );

  await overlay.focus();
  await page.keyboard.press("Escape");
  const secondClickPoint = await imagePoint(page, 160, 90);
  await page.mouse.move(secondClickPoint.x, secondClickPoint.y);
  await page.mouse.click(secondClickPoint.x, secondClickPoint.y);
  await page.mouse.click(secondClickPoint.x, secondClickPoint.y);
  await expect(page.locator(".frame-stage")).toHaveAttribute("data-selection-mode", "confirmed");
  await overlay.focus();
  await page.keyboard.press("Escape");
  const cardClickPoint = await imagePoint(page, 160, 90);
  await page.mouse.move(cardClickPoint.x, cardClickPoint.y);
  await page.mouse.click(cardClickPoint.x, cardClickPoint.y);
  await page.locator(".candidate-chooser").getByRole("option").nth(3).click();
  await expect(page.locator(".frame-stage")).toHaveAttribute("data-selection-mode", "confirmed");

  await overlay.focus();
  await page.keyboard.press("Escape");
  const sourceSwitchPoint = await imagePoint(page, 160, 90);
  await page.mouse.move(sourceSwitchPoint.x, sourceSwitchPoint.y);
  await page.mouse.click(sourceSwitchPoint.x, sourceSwitchPoint.y);
  await page.getByLabel("Source", { exact: true }).selectOption(secondaryMetadata.source_key);
  await expect(page.locator(".frame-stage")).toHaveAttribute("data-selection-mode", "explore");
  await expect(page.locator(".candidate-chooser").getByRole("option")).toHaveCount(0);
  await page.screenshot({ path: testInfo.outputPath(`${testInfo.project.name}-stage0.png`), fullPage: true });
  expect(errors).toEqual([]);
});

test("handles empty, no-hit, edge, resize-pinned, and keyboard frame navigation", async ({ page }) => {
  const requestedFrames: number[] = [];
  const errors = collectBrowserErrors(page);
  await installFixtureRoutes(page, requestedFrames);
  await page.goto("/");
  await page.getByLabel("Source", { exact: true }).selectOption(SOURCE_KEY);

  await page.getByLabel("Frame number").fill("2");
  await expect(page.getByText("No observations on frame 2")).toBeVisible();
  await page.getByLabel("Frame number").fill("3");
  await expect(page.getByText("Loading observations")).toHaveCount(0);
  const edge = await imagePoint(page, 0.5, 0.5);
  await page.mouse.move(edge.x, edge.y);
  await expect(page.locator(".frame-stage")).toHaveAttribute("data-active-row", "301");

  await page.getByLabel("Frame number").fill("1");
  await expect(page.getByText("Loading observations")).toHaveCount(0);
  const noHit = await imagePoint(page, 20, 120);
  await page.mouse.move(noHit.x, noHit.y);
  await expect(page.locator(".frame-stage")).toHaveAttribute("data-candidate-count", "0");
  await expect(page.locator('[data-layer="overlay"]')).toHaveAttribute("data-overlay-strokes", "0");
  await page.evaluate(() => {
    const shell = document.querySelector<HTMLElement>(".app-shell")!;
    shell.style.minHeight = "200vh";
  });
  const scrollBeforeNoHitWheel = await page.evaluate(() => scrollY);
  await page.mouse.wheel(0, 100);
  await expect.poll(() => page.evaluate(() => scrollY)).toBeGreaterThan(scrollBeforeNoHitWheel);
  await page.evaluate(() => scrollTo(0, 0));

  const overlap = await imagePoint(page, 160, 90);
  await page.mouse.move(overlap.x, overlap.y);
  await page.mouse.click(overlap.x, overlap.y);
  await expect(page.locator(".frame-stage")).toHaveAttribute("data-selection-mode", "pinned");
  await page.setViewportSize({ width: 760, height: 780 });
  await expect(page.locator(".frame-stage")).toHaveAttribute("data-active-row", "107");
  await expect(page.locator('[data-layer="overlay"]')).toHaveAttribute("data-overlay-strokes", "7");

  const overlay = page.locator('[data-layer="overlay"]');
  await overlay.focus();
  await page.keyboard.press("Escape");
  await page.keyboard.press("ArrowRight");
  await expect(page.getByLabel("Frame number")).toHaveValue("2");
  await page.keyboard.press("Shift+ArrowRight");
  await expect(page.getByLabel("Frame number")).toHaveValue("5");
  await page.getByRole("button", { name: "Previous 10 frames" }).click();
  await page.getByRole("button", { name: "Start playback" }).click();
  await expect(page.getByLabel("Frame number")).toHaveValue("2");
  await page.getByRole("button", { name: "Pause playback" }).click();
  await page.getByLabel("Frame scrubber").fill("4");
  await expect(page.getByLabel("Frame number")).toHaveValue("4");
  expect(errors).toEqual([]);
});

test("Explore, pinned chooser, and Focus have no serious or critical accessibility violations", async ({ page }) => {
  const requestedFrames: number[] = [];
  const errors = collectBrowserErrors(page);
  await installTrackedFixtureRoutes(page, requestedFrames);
  await page.goto("/");

  await page.getByLabel("Source", { exact: true }).selectOption(metadata.source_key);
  await expect(page.getByText("Loading observations")).toHaveCount(0);
  await assertNoSeriousAccessibilityViolations(page);

  const overlap = await imagePoint(page, 160, 90);
  await page.mouse.move(overlap.x, overlap.y);
  await page.mouse.click(overlap.x, overlap.y);
  await expect(page.locator(".frame-stage")).toHaveAttribute("data-selection-mode", "pinned");
  await assertNoSeriousAccessibilityViolations(page);

  await page.getByLabel("Source", { exact: true }).selectOption(continuousMetadata.source_key);
  await page.getByLabel("Exact track ID").fill("8");
  await page.getByRole("button", { name: "Find track" }).click();
  await expect(page.getByRole("region", { name: "Focus review for track 8" })).toBeVisible();
  await assertNoSeriousAccessibilityViolations(page);
  expect(errors).toEqual([]);
});

test("tracked continuous candidate enters Focus with temporal crops and exact navigation", async ({ page }) => {
  const requestedFrames: number[] = [];
  const errors = collectBrowserErrors(page);
  await installTrackedFixtureRoutes(page, requestedFrames);
  await page.goto("/");
  await page.getByLabel("Source", { exact: true }).selectOption(continuousMetadata.source_key);
  await page.getByLabel("Frame number").fill("2");
  await expect(page.getByText("Loading observations")).toHaveCount(0);
  const point = await imagePoint(page, 110, 80);
  await page.mouse.click(point.x, point.y);
  const candidate = page.getByRole("option", { name: /observation row 1002/i });
  await expect(candidate).toContainText("Current / 2");
  await expect(candidate).toContainText("Earlier / 1");
  await expect(candidate).toContainText("Later / 3");
  await candidate.click();
  await expect(page.getByRole("region", { name: "Focus review for track 8" })).toBeVisible();
  await expect(page.locator('[data-layer="overlay"]')).toHaveAttribute("data-overlay-commands", "2");
  expect(await overlayAlphaSum(page)).toBeGreaterThan(0);
  await assertNoHorizontalOverflow(page);
  await page.getByRole("button", { name: "Next observation" }).click();
  await expect(page.getByLabel("Frame number")).toHaveValue("3");
  await page.getByRole("button", { name: "Previous observation" }).click();
  await expect(page.getByLabel("Frame number")).toHaveValue("2");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("region", { name: "Focus review for track 8" })).toHaveCount(0);
  await expect(page.locator(".frame-stage")).toHaveAttribute("data-selection-mode", "explore");
  expect(errors).toEqual([]);
});

test("reduced motion keeps the focal box and suppresses the trace", async ({ page }) => {
  const requestedFrames: number[] = [];
  const errors = collectBrowserErrors(page);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await installTrackedFixtureRoutes(page, requestedFrames);
  await page.goto("/");
  await page.getByLabel("Source", { exact: true }).selectOption(continuousMetadata.source_key);
  await page.getByLabel("Frame number").fill("2");
  await page.getByLabel("Exact track ID").fill("8");
  await page.getByRole("button", { name: "Find track" }).click();

  await expect(page.getByRole("region", { name: "Focus review for track 8" })).toBeVisible();
  await expect(page.locator('[data-layer="overlay"]')).toHaveAttribute("data-overlay-commands", "1");
  expect(await overlayAlphaSum(page)).toBeGreaterThan(0);
  expect(errors).toEqual([]);
});

test("keyboard-only tracked review enables restrained Context without layout overlap", async ({ page }, testInfo) => {
  const requestedFrames: number[] = [];
  const errors = collectBrowserErrors(page);
  await installTrackedFixtureRoutes(page, requestedFrames);
  await page.goto("/");

  await page.getByLabel("Source", { exact: true }).focus();
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("Enter");
  await page.getByLabel("Exact track ID").focus();
  await page.keyboard.type("8");
  await page.keyboard.press("Enter");
  await expect(page.getByRole("region", { name: "Focus review for track 8" })).toBeVisible();

  await page.getByRole("radio", { name: "Focus" }).focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("radio", { name: "Context" })).toBeChecked();
  const overlay = page.locator('[data-layer="overlay"]');
  await expect(overlay).toHaveAttribute("data-context-commands", "3");
  const drawPlan = await overlay.evaluate((canvas) => ({
    contextInk: Number(canvas.getAttribute("data-context-ink-area")),
    focalArea: Number(canvas.getAttribute("data-focal-area")),
    labelIntersections: Number(canvas.getAttribute("data-label-intersection-count")),
    focalStroke: Number(canvas.getAttribute("data-focal-stroke-width")),
    contextStroke: Number(canvas.getAttribute("data-context-stroke-width")),
  }));
  expect(drawPlan.contextInk).toBeLessThanOrEqual(drawPlan.focalArea * 0.05);
  expect(drawPlan.labelIntersections).toBe(0);
  expect(drawPlan.focalStroke).toBeGreaterThan(drawPlan.contextStroke);

  const count = page.getByRole("spinbutton", { name: "Context tracks" });
  await count.focus();
  await page.keyboard.press("Control+A");
  await page.keyboard.type("1");
  await expect(overlay).toHaveAttribute("data-context-commands", "1");
  await page.getByRole("checkbox", { name: "Abrupt displacement" }).focus();
  await page.keyboard.press("Space");
  await expect(page.getByRole("checkbox", { name: "Abrupt displacement" })).toBeChecked();
  await expect(page.getByRole("button", { name: "Enabled event at frame 2, seek frame 2" })).toBeVisible();

  await assertNoHorizontalOverflow(page);
  const overflowingControls = await page.locator(".focus-review").evaluate((review) => {
    const reviewRect = review.getBoundingClientRect();
    const controls = Array.from(review.querySelectorAll<HTMLElement>("button, input"))
      .filter((control) => control.closest(".filmstrip__list") === null);
    const boundedElements = [...controls, ...Array.from(review.querySelectorAll<HTMLElement>(".filmstrip__list"))];
    return boundedElements.flatMap((control) => {
      const rect = control.getBoundingClientRect();
      return rect.left >= reviewRect.left && rect.right <= reviewRect.right && rect.width > 0
        ? []
        : [{ name: control.getAttribute("aria-label") ?? control.textContent ?? control.tagName, left: rect.left, right: rect.right, width: rect.width, reviewLeft: reviewRect.left, reviewRight: reviewRect.right }];
    });
  });
  expect(overflowingControls).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath(`${testInfo.project.name}-context-keyboard.png`), fullPage: true });
  expect(errors).toEqual([]);
});

test("tracked gapped Focus shows exact evidence, no current box, and resets on sentinel source", async ({ page }) => {
  const requestedFrames: number[] = [];
  const errors = collectBrowserErrors(page);
  await installTrackedFixtureRoutes(page, requestedFrames);
  await page.goto("/");
  await page.getByLabel("Source", { exact: true }).selectOption(gappedMetadata.source_key);
  await page.getByLabel("Exact track ID").fill("8");
  await page.getByRole("button", { name: "Find track" }).click();
  await expect(page.getByRole("region", { name: "Focus review for track 8" })).toBeVisible();
  await expect(page.getByLabel("Track timeline")).toContainText("Low confidence at or below 0.85");
  await page.getByRole("button", { name: "Next gap" }).click();
  await expect(page.getByLabel("Frame number")).toHaveValue("2");
  await expect(page.getByText("Gap at frame 2. Previous observation 1; next observation 3.")).toBeVisible();
  await expect(page.locator('[data-layer="overlay"]')).toHaveAttribute("data-overlay-commands", "0");
  expect(await overlayAlphaSum(page)).toBe(0);
  await assertNoHorizontalOverflow(page);
  await page.getByRole("button", { name: "Next observation" }).click();
  await expect(page.getByLabel("Frame number")).toHaveValue("3");
  await page.getByRole("button", { name: "Gap 4-4, seek frame 4" }).click();
  await expect(page.getByLabel("Frame number")).toHaveValue("4");
  await page.getByLabel("Source", { exact: true }).selectOption(metadata.source_key);
  await expect(page.getByRole("region", { name: "Focus review for track 8" })).toHaveCount(0);
  await expect(page.locator(".frame-stage")).toHaveAttribute("data-selection-mode", "explore");
  await expect(page.getByText("Track ID search unavailable for this source.")).toBeVisible();
  expect(errors).toEqual([]);
});

test("source switch ignores delayed Focus evidence without recreating the prior track", async ({ page }) => {
  const requestedFrames: number[] = [];
  const errors = collectBrowserErrors(page);
  await installTrackedFixtureRoutes(page, requestedFrames);
  let releaseEvidence!: () => void;
  const evidenceGate = new Promise<void>((resolve) => { releaseEvidence = resolve; });
  await page.route(/\/api\/sequences\/synthetic-tracked-continuous\/tracks\/8\?(?:.*)$/, async (route) => {
    await evidenceGate;
    const observations = TRACKED_OBSERVATIONS.get(continuousMetadata.source_key)!;
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({
      source_key: continuousMetadata.source_key,
      sequence: continuousMetadata.sequence,
      source_hash: continuousMetadata.source_hash,
      track_id: 8,
      observation_frames: observations.map((item) => item.frame),
      gaps: [],
      first_observation: observations[0],
      last_observation: observations.at(-1),
      previous_observation: null,
      next_observation: observations[1],
      observations,
    }) });
  });

  await page.goto("/");
  await page.getByLabel("Source", { exact: true }).selectOption(continuousMetadata.source_key);
  await page.getByLabel("Exact track ID").fill("8");
  await page.getByRole("button", { name: "Find track" }).click();
  await expect(page.getByText("Searching this sequence.")).toBeVisible();
  await page.getByLabel("Source", { exact: true }).selectOption(metadata.source_key);
  releaseEvidence();

  await expect(page.getByText("Track ID search unavailable for this source.")).toBeVisible();
  await expect(page.getByRole("region", { name: "Focus review for track 8" })).toHaveCount(0);
  await expect(page.locator(".frame-stage")).toHaveAttribute("data-selection-mode", "explore");
  expect(errors).toEqual([]);
});

test("real MOT20-01 continuous track stays stable in Focus", async ({ page }) => {
  test.skip(process.env.MOT20_REAL_E2E !== "1", "Requires the configured local MOT20-01 viewer API");
  const errors = collectBrowserErrors(page);

  await page.goto("/");
  await page.getByLabel("Source", { exact: true }).selectOption("mot20-01-gt");
  await page.getByLabel("Exact track ID").fill("1");
  await page.getByRole("button", { name: "Find track" }).click();

  await expect(page.getByRole("region", { name: "Focus review for track 1" })).toBeVisible();
  await expect(page.getByLabel("Frame number")).toHaveValue("1");
  await expect(page.locator('[data-layer="overlay"]')).toHaveAttribute("data-overlay-commands", "1");
  expect(await overlayAlphaSum(page)).toBeGreaterThan(0);
  await page.getByRole("button", { name: "Next observation" }).click();
  await expect(page.getByLabel("Frame number")).toHaveValue("2");
  await expect(page.getByText("Observed on exact frame 2, source row 2.")).toBeVisible();
  expect(await overlayAlphaSum(page)).toBeGreaterThan(0);
  await assertNoHorizontalOverflow(page);
  expect(errors).toEqual([]);
});

test("real dense MOT20-01 Context remains subordinate at three and one competitors", async ({ page }, testInfo) => {
  test.skip(process.env.MOT20_REAL_E2E !== "1", "Requires the configured local MOT20-01 viewer API");
  const errors = collectBrowserErrors(page);

  await page.goto("/");
  await page.getByLabel("Source", { exact: true }).selectOption("mot20-01-gt");
  await page.getByLabel("Frame number").fill("330");
  await page.getByLabel("Exact track ID").fill("1");
  await page.getByRole("button", { name: "Find track" }).click();
  await expect(page.getByRole("region", { name: "Focus review for track 1" })).toBeVisible();
  await page.getByRole("radio", { name: "Context" }).check();

  const overlay = page.locator('[data-layer="overlay"]');
  await expect(overlay).toHaveAttribute("data-context-commands", "3");
  for (const count of [3, 1]) {
    await page.getByRole("spinbutton", { name: "Context tracks" }).fill(String(count));
    await expect(overlay).toHaveAttribute("data-context-commands", String(count));
    const plan = await overlay.evaluate((canvas) => ({
      contextInk: Number(canvas.getAttribute("data-context-ink-area")),
      focalArea: Number(canvas.getAttribute("data-focal-area")),
      labelIntersections: Number(canvas.getAttribute("data-label-intersection-count")),
      focalStroke: Number(canvas.getAttribute("data-focal-stroke-width")),
      contextStroke: Number(canvas.getAttribute("data-context-stroke-width")),
    }));
    expect(plan.contextInk).toBeLessThanOrEqual(plan.focalArea * 0.05);
    expect(plan.labelIntersections).toBe(0);
    expect(plan.focalStroke).toBeGreaterThan(plan.contextStroke);
    await page.screenshot({ path: testInfo.outputPath(`${testInfo.project.name}-mot20-01-frame-330-context-${count}.png`), fullPage: true });
  }
  await assertNoHorizontalOverflow(page);
  expect(errors).toEqual([]);
});