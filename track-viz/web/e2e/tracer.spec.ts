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
const activityMetadata = {
  ...continuousMetadata,
  source_key: "synthetic-tracked-activities",
  sequence: "SYNTHETIC-ACTIVITIES",
  source_hash: "e".repeat(64),
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
  [activityMetadata.source_key, [
    trackedObservation(activityMetadata, 1, 3001),
    trackedObservation(activityMetadata, 2, 3002),
    trackedObservation(activityMetadata, 3, 3003),
    trackedObservation(activityMetadata, 4, 3004),
    trackedObservation(activityMetadata, 5, 3005),
  ]],
]);

const RAW_NEXT_DISPLACEMENT_FRAME = 2;
const GROUPED_NEXT_DISPLACEMENT_ANCHOR = 5;

function displacementEvent(fromFrame: number, toFrame: number, normalizedDisplacement: number, sourceRowOffset = 3000) {
  return {
    from_frame: fromFrame,
    to_frame: toFrame,
    from_row_index: sourceRowOffset + fromFrame,
    to_row_index: sourceRowOffset + toFrame,
    frame_delta: toFrame - fromFrame,
    center_displacement_pixels: normalizedDisplacement * 125,
    normalization_box_height: 125,
    normalized_displacement: normalizedDisplacement,
    threshold: 0.02,
  };
}

function scaleChangeEvent(fromFrame: number, toFrame: number, normalizedScaleChange: number, sourceRowOffset = 3000) {
  return {
    from_frame: fromFrame,
    to_frame: toFrame,
    from_row_index: sourceRowOffset + fromFrame,
    to_row_index: sourceRowOffset + toFrame,
    frame_delta: toFrame - fromFrame,
    absolute_height_change_pixels: normalizedScaleChange * 125,
    normalization_box_height: 125,
    normalized_scale_change: normalizedScaleChange,
    threshold: 0.5,
  };
}

function proximityEvent(frame: number, normalizedEdgeProximity: number) {
  return {
    frame,
    focal_row_index: 3000 + frame,
    competitor_track_id: 9,
    competitor_row_index: 9000 + frame,
    edge_distance_pixels: normalizedEdgeProximity * 125,
    focal_box_height: 125,
    normalized_edge_proximity: normalizedEdgeProximity,
    threshold: 0.25,
  };
}

type FixtureResponseOutcome = void | "error";

interface TrackedFixtureOptions {
  beforeEventResponse?(requestIndex: number, url: URL): Promise<FixtureResponseOutcome> | FixtureResponseOutcome;
  beforeContextResponse?(frame: number, requestIndex: number, url: URL): Promise<FixtureResponseOutcome> | FixtureResponseOutcome;
  beforeObservationResponse?(frame: number, requestIndex: number, url: URL): Promise<FixtureResponseOutcome> | FixtureResponseOutcome;
  beforeImageResponse?(frame: number, requestIndex: number, url: URL): Promise<FixtureResponseOutcome> | FixtureResponseOutcome;
}

async function installTrackedFixtureRoutes(
  page: Page,
  requestedFrames: number[],
  options: TrackedFixtureOptions = {},
) {
  const sources = [continuousMetadata, gappedMetadata, activityMetadata, metadata];
  await installSourceSetupRoutes(page, metadata);
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
  let eventRequestIndex = 0;
  await page.route(/\/api\/sequences\/[^/]+\/tracks\/8\/events(?:\?.*)?$/, async (route) => {
    const url = new URL(route.request().url());
    eventRequestIndex += 1;
    if (await options.beforeEventResponse?.(eventRequestIndex, url) === "error") {
      return route.fulfill({ status: 503, body: "synthetic event failure" });
    }
    const sourceKey = url.pathname.split("/")[3];
    const source = sources.find((item) => item.source_key === sourceKey)!;
    const observations = TRACKED_OBSERVATIONS.get(sourceKey)!;
    const meaningful = sourceKey === gappedMetadata.source_key || sourceKey === activityMetadata.source_key;
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
      displacement_events: displacementEnabled
        ? sourceKey === activityMetadata.source_key
          ? [
            displacementEvent(1, RAW_NEXT_DISPLACEMENT_FRAME, 0.03),
            displacementEvent(2, 3, 0.04),
            displacementEvent(4, GROUPED_NEXT_DISPLACEMENT_ANCHOR, 0.06),
          ]
          : [displacementEvent(1, 2, 0.5, 1000)]
        : [],
      scale_change_events: scaleEnabled && sourceKey === activityMetadata.source_key
        ? [scaleChangeEvent(2, 4, 0.6)]
        : [],
      close_interaction_events: interactionEnabled && sourceKey === activityMetadata.source_key
        ? [proximityEvent(1, 0.25), proximityEvent(2, 0.2), proximityEvent(3, 0.15)]
        : [],
      low_confidence_observations: meaningful
        ? sourceKey === activityMetadata.source_key ? [observations[1], observations[3]] : [observations[1]]
        : [],
    }) });
  });
  let contextRequestIndex = 0;
  await page.route(/\/api\/sequences\/[^/]+\/tracks\/8\/context(?:\?.*)?$/, async (route) => {
    const url = new URL(route.request().url());
    const sourceKey = url.pathname.split("/")[3];
    const source = sources.find((item) => item.source_key === sourceKey)!;
    const frame = Number(url.searchParams.get("frame"));
    contextRequestIndex += 1;
    if (await options.beforeContextResponse?.(frame, contextRequestIndex, url) === "error") {
      return route.fulfill({ status: 503, body: "synthetic context failure" });
    }
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
  let observationRequestIndex = 0;
  await page.route(/\/api\/sequences\/[^/]+\/frames\/\d+\/observations(?:\?.*)?$/, async (route) => {
    const url = new URL(route.request().url());
    const sourceKey = url.pathname.split("/")[3];
    const frame = Number(url.pathname.split("/")[5]);
    observationRequestIndex += 1;
    if (await options.beforeObservationResponse?.(frame, observationRequestIndex, url) === "error") {
      return route.fulfill({ status: 503, body: "synthetic observation failure" });
    }
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
  let imageRequestIndex = 0;
  await page.route(/\/api\/sequences\/[^/]+\/frames\/\d+(?:\?.*)?$/, async (route) => {
    const url = new URL(route.request().url());
    const sourceKey = url.pathname.split("/")[3];
    const frame = Number(url.pathname.split("/")[5]);
    const source = sources.find((item) => item.source_key === sourceKey)!;
    expect(url.searchParams.get("source_hash")).toBe(source.source_hash);
    imageRequestIndex += 1;
    if (await options.beforeImageResponse?.(frame, imageRequestIndex, url) === "error") {
      return route.fulfill({ status: 503, body: "synthetic image failure" });
    }
    requestedFrames.push(frame);
    return route.fulfill({ contentType: "image/jpeg", body: Buffer.from(JPEG_BY_FRAME[frame], "base64") });
  });
}

async function installFixtureRoutes(
  page: Page,
  requestedFrames: number[],
  delayedFrame?: { frame: number; delayMs: number },
) {
  await installSourceSetupRoutes(page, metadata, true);
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
  await page.route(/\/api\/sequences\/[^/]+\/frames\/\d+(?:\?.*)?$/, async (route) => {
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
    if (delayedFrame?.frame === frame) {
      await new Promise((resolve) => setTimeout(resolve, delayedFrame.delayMs));
    }
    return route.fulfill({
      contentType: "image/jpeg",
      body: Buffer.from(body, "base64"),
    });
  });
}

async function installSourceSetupRoutes(
  page: Page,
  loadedSource: typeof metadata,
  validatePost = false,
) {
  await page.route("**/api/source-selection", (route) => {
    if (route.request().method() === "POST") {
      if (validatePost) {
        expect(route.request().postDataJSON()).toEqual({
          images: "/srv/mot20/MOT20-08/img1",
          annotations: "/srv/predictions/MOT20-08.txt",
        });
      }
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({ sources: [loadedSource], unavailable: [], diagnostics: [] }),
      });
    }
    return route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        images: "/srv/mot20/MOT20-06/img1",
        annotations: "/srv/predictions/MOT20-06.txt",
      }),
    });
  });
  await page.route("**/api/source-path-suggestions?*", (route) => {
    const requestUrl = new URL(route.request().url());
    const kind = requestUrl.searchParams.get("kind");
    const query = requestUrl.searchParams.get("query") ?? "";
    const path = kind === "images"
      ? "/srv/mot20/MOT20-08/img1"
      : "/srv/predictions/MOT20-08.txt";
    return route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        kind,
        query,
        directory: kind === "images" ? "/srv/mot20/MOT20-08" : "/srv/predictions",
        parent: "/srv",
        entries: [{ path, entry_type: kind === "images" ? "directory" : "file" }],
        suggestions: [path],
      }),
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
  await viewport.scrollIntoViewIfNeeded();
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

async function reviewLayout(page: Page): Promise<Record<string, { x: number; y: number; width: number; height: number }>> {
  const targets = {
    focus: page.getByRole("region", { name: "Focus review for track 8" }),
    viewport: page.getByTestId("frame-viewport"),
    timeline: page.getByLabel("Track timeline"),
    filmstrip: page.getByLabel(/Track filmstrip/),
    sourceStatus: page.locator(".source-status"),
  };
  return Object.fromEntries(await Promise.all(Object.entries(targets).map(async ([name, target]) => {
    const box = await target.evaluate((element) => {
      const rect = element.getBoundingClientRect();
      return { x: rect.x + scrollX, y: rect.y + scrollY, width: rect.width, height: rect.height };
    });
    if (box.width === 0 || box.height === 0) throw new Error(`${name} was not visible while measuring review layout`);
    return [name, box];
  })));
}

function expectStableReviewLayout(
  before: Record<string, { x: number; y: number; width: number; height: number }>,
  current: Record<string, { x: number; y: number; width: number; height: number }>,
): void {
  for (const name of Object.keys(before)) {
    expect(Math.abs(current[name].x - before[name].x), `${name} x position`).toBeLessThanOrEqual(1);
    expect(Math.abs(current[name].y - before[name].y), `${name} y position`).toBeLessThanOrEqual(1);
    expect(Math.abs(current[name].width - before[name].width), `${name} width`).toBeLessThanOrEqual(1);
    expect(Math.abs(current[name].height - before[name].height), `${name} height`).toBeLessThanOrEqual(1);
  }
}

async function openContinuousFocus(page: Page): Promise<void> {
  await page.goto("/");
  await page.getByLabel("Source", { exact: true }).selectOption(continuousMetadata.source_key);
  await page.getByLabel("Exact track ID").fill("8");
  await page.getByRole("button", { name: "Find track" }).click();
  await expect(page.getByRole("region", { name: "Focus review for track 8" })).toBeVisible();
}

async function assertUnrelatedFocusControlOperable(page: Page): Promise<void> {
  const contextCount = page.getByRole("spinbutton", { name: "Number of nearby tracks" });
  await contextCount.fill("3");
  await expect(contextCount).toHaveValue("3");
}

async function assertNoSeriousAccessibilityViolations(page: Page, include?: string): Promise<void> {
  const builder = new AxeBuilder({ page });
  if (include !== undefined) builder.include(include);
  const results = await builder
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  const violations = results.violations.filter(
    (violation) => violation.impact === "serious" || violation.impact === "critical",
  );
  expect(violations).toEqual([]);
}

function cssRgb(value: string): [number, number, number] {
  const channels = value.match(/[\d.]+/g)?.slice(0, 3).map(Number);
  if (channels?.length !== 3) throw new Error(`Expected an RGB color, received ${value}`);
  return channels as [number, number, number];
}

function relativeLuminance(color: [number, number, number]): number {
  const [red, green, blue] = color.map((channel) => {
    const normalized = channel / 255;
    return normalized <= 0.04045
      ? normalized / 12.92
      : ((normalized + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

function contrastRatio(foreground: string, background: string): number {
  const first = relativeLuminance(cssRgb(foreground));
  const second = relativeLuminance(cssRgb(background));
  const lighter = Math.max(first, second);
  const darker = Math.min(first, second);
  return (lighter + 0.05) / (darker + 0.05);
}

test("uses a dark line-free visual foundation without shrinking the viewport", async ({ page }, testInfo) => {
  const requestedFrames: number[] = [];
  await installFixtureRoutes(page, requestedFrames);
  await page.goto("/");
  await page.getByLabel("Source", { exact: true }).selectOption(SOURCE_KEY);
  await expect(page.getByTestId("frame-viewport")).toHaveAttribute("data-image-rect", /.+/);

  const visual = await page.evaluate(() => {
    const style = (selector: string) => getComputedStyle(document.querySelector<HTMLElement>(selector)!);
    const body = getComputedStyle(document.body);
    const control = style("button");
    const panel = style(".track-search");
    return {
      colorScheme: getComputedStyle(document.documentElement).colorScheme,
      bodyBackground: body.backgroundColor,
      bodyBackgroundImage: body.backgroundImage,
      bodyColor: body.color,
      controlBackground: control.backgroundColor,
      controlColor: control.color,
      panelBackground: panel.backgroundColor,
      panelColor: panel.color,
    };
  });

  expect(visual.colorScheme).toContain("dark");
  expect(visual.bodyBackgroundImage).toBe("none");
  expect(relativeLuminance(cssRgb(visual.bodyBackground))).toBeLessThan(0.1);
  expect(relativeLuminance(cssRgb(visual.panelBackground))).toBeLessThan(0.2);
  expect(contrastRatio(visual.bodyColor, visual.bodyBackground)).toBeGreaterThanOrEqual(4.5);
  expect(contrastRatio(visual.controlColor, visual.controlBackground)).toBeGreaterThanOrEqual(4.5);
  expect(contrastRatio(visual.panelColor, visual.panelBackground)).toBeGreaterThanOrEqual(4.5);

  const viewport = await page.getByTestId("frame-viewport").boundingBox();
  expect(viewport).not.toBeNull();
  expect(viewport!.width).toBeGreaterThanOrEqual(testInfo.project.name === "narrow-chromium" ? 300 : 900);
  expect(viewport!.height).toBeGreaterThanOrEqual(260);
  await assertNoHorizontalOverflow(page);
  await assertCanvasGeometryAndPixels(page);
  await assertNoSeriousAccessibilityViolations(page);
});

test("loads image and annotation paths suggested by the server", async ({ page }) => {
  const requestedFrames: number[] = [];
  const errors = collectBrowserErrors(page);
  await installFixtureRoutes(page, requestedFrames);
  await page.goto("/");

  const images = page.getByLabel("Images folder on server");
  const annotations = page.getByLabel("Predictions or ground truth file on server");
  await expect(images).toHaveValue("/srv/mot20/MOT20-06/img1");
  await expect(annotations).toHaveValue("/srv/predictions/MOT20-06.txt");
  await images.fill("/srv/mot20/MOT20-08/img1");
  await expect(
    page.getByRole("dialog", { name: "Server image folder browser" })
      .getByText("/srv/mot20/MOT20-08/img1"),
  ).toBeVisible();
  await annotations.fill("/srv/predictions/MOT20-08.txt");
  await page.getByRole("button", { name: "Load source" }).click();

  await expect(page.getByLabel("Source", { exact: true })).toHaveValue(metadata.source_key);
  await expect(page.getByLabel("Frame number")).toHaveValue("1");
  await assertNoHorizontalOverflow(page);
  expect(errors).toEqual([]);
});

test("playback waits for a delayed frame and resumes after it decodes", async ({ page }) => {
  const requestedFrames: number[] = [];
  await installFixtureRoutes(page, requestedFrames, { frame: 2, delayMs: 1000 });
  await page.goto("/");
  await page.getByLabel("Source", { exact: true }).selectOption(SOURCE_KEY);
  await expect(page.getByRole("status", { name: /loading exact frame/i })).toHaveCount(0);

  await page.getByRole("button", { name: "Start playback" }).click();
  const frameInput = page.getByLabel("Frame number");
  await expect(frameInput).toHaveValue("2");
  await page.waitForTimeout(450);
  expect(await frameInput.inputValue()).toBe("2");
  await expect(frameInput).toHaveValue("3", { timeout: 2000 });
  expect(requestedFrames.filter((frame) => frame === 2)).toHaveLength(1);
});

test("selects a source and renders exact first, middle, and last JPEGs", async ({ page }, testInfo: TestInfo) => {
  const requestedFrames: number[] = [];
  const errors = collectBrowserErrors(page);
  await installFixtureRoutes(page, requestedFrames);

  await page.goto("/");
  await page.getByLabel("Source", { exact: true }).selectOption(SOURCE_KEY);
  await expect(page.getByText(/local test-adapted development material/i)).toHaveCount(0);
  await expect(page.getByText(/not a held-out benchmark result/i)).toHaveCount(0);

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
  await page.setViewportSize({ width: 829, height: 780 });
  await assertNoHorizontalOverflow(page);

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
  const trajectoryToggle = page.getByRole("checkbox", { name: "Show future trajectory" }).locator("..");
  expect((await trajectoryToggle.boundingBox())?.height).toBeLessThanOrEqual(40);
  for (const target of [
    ".track-timeline__rail",
    '[aria-label="Displacement activity controls"]',
    '[aria-label="Scale change activity controls"]',
    '[aria-label="Proximity activity controls"]',
    '[aria-label="Low-confidence observation controls"]',
    ".trajectory-toggle",
  ]) await assertNoSeriousAccessibilityViolations(page, target);

  await page.getByLabel("Source", { exact: true }).selectOption(activityMetadata.source_key);
  await page.getByLabel("Exact track ID").fill("8");
  await page.getByRole("button", { name: "Find track" }).click();
  await expect(page.getByRole("region", { name: "Focus review for track 8" })).toBeVisible();
  await page.getByRole("checkbox", { name: "Abrupt displacement" }).check();
  await page.getByRole("checkbox", { name: "Scale change" }).check();
  await page.getByRole("checkbox", { name: "Close interaction" }).check();
  await expect(page.getByRole("button", { name: "Next Scale change activity" })).toBeEnabled();
  for (const target of [
    ".track-timeline__rail",
    '[aria-label="Displacement activity controls"]',
    '[aria-label="Scale change activity controls"]',
    '[aria-label="Proximity activity controls"]',
    '[aria-label="Low-confidence observation controls"]',
    ".trajectory-toggle",
  ]) await assertNoSeriousAccessibilityViolations(page, target);
  expect(errors).toEqual([]);
});

test("delayed Context playback keeps the Focus review geometry fixed", async ({ page }) => {
  const requestedFrames: number[] = [];
  let releaseContext!: () => void;
  const contextGate = new Promise<void>((resolve) => { releaseContext = resolve; });
  let contextRequestStarted!: () => void;
  const contextStarted = new Promise<void>((resolve) => { contextRequestStarted = resolve; });
  await installTrackedFixtureRoutes(page, requestedFrames, {
    beforeContextResponse: async (frame) => {
      if (frame === 2) {
        contextRequestStarted();
        await contextGate;
      }
    },
  });
  await openContinuousFocus(page);
  await page.getByRole("spinbutton", { name: "Number of nearby tracks" }).fill("3");
  await expect(page.getByText("Loading context evidence")).toHaveCount(0);

  const before = await reviewLayout(page);
  await page.getByRole("button", { name: "Start playback" }).click();
  await contextStarted;
  await expect(page.getByLabel("Frame number")).toHaveValue("2");
  await expect(page.getByText("Loading context evidence")).toBeVisible();
  expectStableReviewLayout(before, await reviewLayout(page));
  await assertUnrelatedFocusControlOperable(page);

  releaseContext();
  await expect(page.getByText("Loading context evidence")).toHaveCount(0);
  expectStableReviewLayout(before, await reviewLayout(page));
});

test("delayed current image playback keeps the Focus review geometry fixed", async ({ page }) => {
  const requestedFrames: number[] = [];
  let releaseImage!: () => void;
  const imageGate = new Promise<void>((resolve) => { releaseImage = resolve; });
  await installTrackedFixtureRoutes(page, requestedFrames, {
    beforeImageResponse: async (frame) => {
      if (frame === 2) await imageGate;
    },
  });
  await openContinuousFocus(page);

  const before = await reviewLayout(page);
  await page.getByRole("button", { name: "Start playback" }).click();
  await expect(page.getByLabel("Frame number")).toHaveValue("2");
  await expect(page.getByText("Loading exact frame 2")).toBeVisible();
  expectStableReviewLayout(before, await reviewLayout(page));
  await assertUnrelatedFocusControlOperable(page);
  await page.getByRole("button", { name: "Pause playback" }).click();

  releaseImage();
  await expect(page.getByText("Loading exact frame 2")).toHaveCount(0);
  expectStableReviewLayout(before, await reviewLayout(page));
});

test("failed current image playback keeps the Focus review geometry fixed and transport operable", async ({ page }) => {
  const requestedFrames: number[] = [];
  let releaseImage!: () => void;
  const imageGate = new Promise<void>((resolve) => { releaseImage = resolve; });
  await installTrackedFixtureRoutes(page, requestedFrames, {
    beforeImageResponse: async (frame) => {
      if (frame !== 2) return;
      await imageGate;
      return "error";
    },
  });
  await openContinuousFocus(page);

  const before = await reviewLayout(page);
  await page.getByRole("button", { name: "Start playback" }).click();
  await expect(page.getByLabel("Frame number")).toHaveValue("2");
  await expect(page.getByText("Loading exact frame 2")).toBeVisible();
  expectStableReviewLayout(before, await reviewLayout(page));
  await assertUnrelatedFocusControlOperable(page);
  await page.getByRole("button", { name: "Pause playback" }).click();

  releaseImage();
  await expect(page.getByRole("alert")).toContainText("Exact frame 2 could not be loaded.");
  expectStableReviewLayout(before, await reviewLayout(page));
  await assertUnrelatedFocusControlOperable(page);
  await page.getByRole("button", { name: "Next frame" }).click();
  await expect(page.getByLabel("Frame number")).toHaveValue("3");
});

test("delayed frame observations during playback keep the Focus review geometry fixed", async ({ page }) => {
  const requestedFrames: number[] = [];
  let releaseObservations!: () => void;
  const observationGate = new Promise<void>((resolve) => { releaseObservations = resolve; });
  await installTrackedFixtureRoutes(page, requestedFrames, {
    beforeObservationResponse: async (frame) => {
      if (frame === 2) await observationGate;
    },
  });
  await openContinuousFocus(page);

  const before = await reviewLayout(page);
  await page.getByRole("button", { name: "Start playback" }).click();
  await expect(page.getByLabel("Frame number")).toHaveValue("2");
  await expect(page.getByText("Loading observations")).toBeVisible();
  expectStableReviewLayout(before, await reviewLayout(page));
  await assertUnrelatedFocusControlOperable(page);
  await page.getByRole("button", { name: "Pause playback" }).click();

  releaseObservations();
  await expect(page.getByText("Loading observations")).toHaveCount(0);
  expectStableReviewLayout(before, await reviewLayout(page));
});

test("failed frame observations keep the Focus review geometry fixed and transport operable", async ({ page }) => {
  const requestedFrames: number[] = [];
  let releaseObservations!: () => void;
  const observationGate = new Promise<void>((resolve) => { releaseObservations = resolve; });
  await installTrackedFixtureRoutes(page, requestedFrames, {
    beforeObservationResponse: async (frame) => {
      if (frame !== 2) return;
      await observationGate;
      return "error";
    },
  });
  await openContinuousFocus(page);

  const before = await reviewLayout(page);
  await page.getByRole("button", { name: "Start playback" }).click();
  await expect(page.getByLabel("Frame number")).toHaveValue("2");
  await expect(page.getByText("Loading observations")).toBeVisible();
  expectStableReviewLayout(before, await reviewLayout(page));
  await assertUnrelatedFocusControlOperable(page);
  await page.getByRole("button", { name: "Pause playback" }).click();

  releaseObservations();
  await expect(page.getByRole("alert")).toContainText("Frame observations could not be loaded.");
  expectStableReviewLayout(before, await reviewLayout(page));
  await assertUnrelatedFocusControlOperable(page);
  await page.getByRole("button", { name: "Next frame" }).click();
  await expect(page.getByLabel("Frame number")).toHaveValue("3");
});

test("failed Context playback keeps the Focus review geometry fixed", async ({ page }) => {
  const requestedFrames: number[] = [];
  let releaseContext!: () => void;
  const contextGate = new Promise<void>((resolve) => { releaseContext = resolve; });
  await installTrackedFixtureRoutes(page, requestedFrames, {
    beforeContextResponse: async (frame) => {
      if (frame !== 2) return;
      await contextGate;
      return "error";
    },
  });
  await openContinuousFocus(page);
  await page.getByRole("spinbutton", { name: "Number of nearby tracks" }).fill("3");
  await expect(page.getByText("Loading context evidence")).toHaveCount(0);

  const before = await reviewLayout(page);
  await page.getByRole("button", { name: "Start playback" }).click();
  await expect(page.getByLabel("Frame number")).toHaveValue("2");
  await expect(page.getByText("Loading context evidence")).toBeVisible();
  expectStableReviewLayout(before, await reviewLayout(page));
  await assertUnrelatedFocusControlOperable(page);
  await page.getByRole("button", { name: "Pause playback" }).click();

  releaseContext();
  await expect(page.getByRole("alert")).toContainText("Context evidence could not be loaded.");
  expectStableReviewLayout(before, await reviewLayout(page));
  await assertUnrelatedFocusControlOperable(page);
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

test("Focus refreshes only delayed event data and keeps the newest settings", async ({ page }) => {
  const requestedFrames: number[] = [];
  let releaseSecondEvent!: () => void;
  const secondEventGate = new Promise<void>((resolve) => { releaseSecondEvent = resolve; });
  let signalSecondEvent!: () => void;
  const secondEventStarted = new Promise<void>((resolve) => { signalSecondEvent = resolve; });
  const eventQueries: URL[] = [];
  let evidenceRequests = 0;
  let filmstripRequests = 0;
  page.on("request", (request) => {
    const pathname = new URL(request.url()).pathname;
    if (/\/tracks\/8$/.test(pathname)) evidenceRequests += 1;
    if (/\/tracks\/8\/filmstrip$/.test(pathname)) filmstripRequests += 1;
  });
  await installTrackedFixtureRoutes(page, requestedFrames, {
    beforeEventResponse: async (requestIndex, url) => {
      eventQueries.push(url);
      if (requestIndex === 3) {
        signalSecondEvent();
        await secondEventGate;
      }
    },
  });
  await page.goto("/");
  await page.getByLabel("Source", { exact: true }).selectOption(continuousMetadata.source_key);
  await page.getByLabel("Exact track ID").fill("8");
  await page.getByRole("button", { name: "Find track" }).click();
  const review = page.getByRole("region", { name: "Focus review for track 8" });
  await expect(review).toBeVisible();
  const initialEvidenceRequests = evidenceRequests;
  const initialFilmstripRequests = filmstripRequests;
  await page.getByRole("checkbox", { name: "Abrupt displacement" }).check();
  await expect(page.getByLabel("Displacement activity controls")).toContainText("1 activity / 1 raw match");
  const layoutBeforeEventRefresh = await reviewLayout(page);
  const threshold = page.getByRole("spinbutton", { name: "Displacement threshold in box heights" });
  await threshold.fill("");
  await threshold.type("0.02");
  await expect(threshold).toHaveValue("0.02");
  await page.waitForTimeout(150);
  expect(eventQueries.at(-1)?.searchParams.get("displacement_threshold")).not.toBe("0.02");
  await secondEventStarted;
  await expect(review).toBeVisible();
  await expect(page.getByLabel("Track timeline")).toBeVisible();
  await expect(page.getByLabel("Track filmstrip, 3 samples")).toBeVisible();
  await expect(page.getByRole("status")).toContainText("Updating review events");
  expectStableReviewLayout(layoutBeforeEventRefresh, await reviewLayout(page));
  await page.getByRole("button", { name: "Next observation" }).click();
  await expect(page.getByLabel("Frame number")).toHaveValue("2");
  expectStableReviewLayout(layoutBeforeEventRefresh, await reviewLayout(page));

  releaseSecondEvent();
  await expect(page.locator(".event-status")).toHaveText("");
  expectStableReviewLayout(layoutBeforeEventRefresh, await reviewLayout(page));

  await page.getByRole("checkbox", { name: "Scale change" }).check();
  await page.getByRole("checkbox", { name: "Close interaction" }).check();
  await expect.poll(() => eventQueries.at(-1)?.searchParams.get("displacement_threshold")).toBe("0.02");
  await expect(page.getByLabel("Track timeline")).toContainText("Scale change >= 0.5");
  await expect(page.getByLabel("Track timeline")).toContainText("Close interaction <= 0.25");
  expect(evidenceRequests).toBe(initialEvidenceRequests);
  expect(filmstripRequests).toBe(initialFilmstripRequests);

  await page.waitForTimeout(100);
  await expect(page.getByLabel("Track timeline")).toContainText("Scale change >= 0.5");
  await expect(page.getByRole("button", { name: "Next enabled event" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Next Displacement activity" })).toBeDisabled();
});

test("narrow failed event refresh keeps the timeline and filmstrip geometry fixed", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "narrow-chromium", "This regression targets narrow event-status wrapping.");
  const requestedFrames: number[] = [];
  await installTrackedFixtureRoutes(page, requestedFrames, {
    beforeEventResponse: (requestIndex) => requestIndex === 2 ? "error" : undefined,
  });
  await openContinuousFocus(page);

  const before = await reviewLayout(page);
  await page.getByRole("checkbox", { name: "Abrupt displacement" }).check();
  await expect(page.getByRole("status")).toContainText("Event refresh failed; showing the last successful settings.");
  await expect(page.getByRole("button", { name: "Retry event refresh" })).toBeVisible();
  expectStableReviewLayout(before, await reviewLayout(page));
});

test("Focus navigates deterministic family activities without combining raw event frames", async ({ page }) => {
  const requestedFrames: number[] = [];
  const errors = collectBrowserErrors(page);
  await installTrackedFixtureRoutes(page, requestedFrames);
  await page.goto("/");
  await page.getByLabel("Source", { exact: true }).selectOption(activityMetadata.source_key);
  await page.getByLabel("Exact track ID").fill("8");
  await page.getByRole("button", { name: "Find track" }).click();
  await expect(page.getByRole("region", { name: "Focus review for track 8" })).toBeVisible();

  expect(RAW_NEXT_DISPLACEMENT_FRAME).toBe(2);
  expect(GROUPED_NEXT_DISPLACEMENT_ANCHOR).toBe(5);
  expect(RAW_NEXT_DISPLACEMENT_FRAME).not.toBe(GROUPED_NEXT_DISPLACEMENT_ANCHOR);
  await expect(page.getByRole("button", { name: "Next Scale change activity" })).toBeDisabled();
  await expect(page.getByLabel("Scale change activity controls")).toContainText("Scale change is off.");
  await page.getByRole("checkbox", { name: "Scale change" }).check();
  await expect(page.getByLabel("Scale change activity controls")).toContainText("1 activity / 1 raw match.");
  await page.getByRole("button", { name: "Next Scale change activity" }).click();
  await expect(page.getByLabel("Frame number")).toHaveValue("4");
  await page.getByLabel("Frame number").fill("1");

  await page.getByRole("checkbox", { name: "Abrupt displacement" }).check();
  await expect(page.getByLabel("Displacement activity controls")).toContainText("2 activities / 3 raw matches");
  await page.getByRole("button", { name: "Next Displacement activity" }).click();
  await expect(page.getByLabel("Frame number")).toHaveValue(String(GROUPED_NEXT_DISPLACEMENT_ANCHOR));

  await page.getByRole("checkbox", { name: "Close interaction" }).check();
  await expect(page.getByLabel("Proximity activity controls")).toContainText("1 activity / 3 raw matches");
  await page.getByLabel("Frame number").fill("4");
  await page.getByRole("button", { name: "Previous Proximity activity" }).click();
  await expect(page.getByLabel("Frame number")).toHaveValue("3");
  await expect(page.getByLabel("Current event activities")).toContainText("Proximity with competitor 9");
  await expect(page.getByRole("button", { name: "Next enabled event" })).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("Focus exercises each event-family combination and keeps the complete history review stable", async ({ page }, testInfo) => {
  const requestedFrames: number[] = [];
  const errors = collectBrowserErrors(page);
  const combinations = [
    ["Abrupt displacement"],
    ["Scale change"],
    ["Close interaction"],
    ["Abrupt displacement", "Scale change"],
    ["Abrupt displacement", "Close interaction"],
    ["Scale change", "Close interaction"],
    ["Abrupt displacement", "Scale change", "Close interaction"],
  ];
  await installTrackedFixtureRoutes(page, requestedFrames);

  for (const enabled of combinations) {
    await page.goto("/");
    await page.getByLabel("Source", { exact: true }).selectOption(activityMetadata.source_key);
    await page.getByLabel("Exact track ID").fill("8");
    await page.getByRole("button", { name: "Find track" }).click();
    await expect(page.getByRole("region", { name: "Focus review for track 8" })).toBeVisible();
    for (const label of enabled) await page.getByRole("checkbox", { name: label }).check();
    for (const label of ["Abrupt displacement", "Scale change", "Close interaction"]) {
      await expect(page.getByRole("checkbox", { name: label })).toHaveJSProperty("checked", enabled.includes(label));
    }
    for (const [label, controls] of [
      ["Abrupt displacement", "Displacement activity controls"],
      ["Scale change", "Scale change activity controls"],
      ["Close interaction", "Proximity activity controls"],
    ] as const) {
      await expect(page.getByLabel(controls)).toContainText(enabled.includes(label) ? /\d+ activit/ : "is off.");
    }
  }

  await page.getByRole("checkbox", { name: "Show future trajectory" }).check();
  await page.getByLabel("Frame number").fill("3");
  await expect(page.locator('[data-layer="overlay"]')).toHaveAttribute("data-overlay-commands", "3");
  await page.getByRole("slider", { name: "Sequence timeline, current frame 3" }).focus();
  await page.keyboard.press("Home");
  await expect(page.getByLabel("Frame number")).toHaveValue("1");
  await page.keyboard.press("End");
  await expect(page.getByLabel("Frame number")).toHaveValue("5");
  await page.screenshot({ path: testInfo.outputPath(`focus-review-${testInfo.project.name}.png`), fullPage: true });
  expect(errors).toEqual([]);
});

test("reduced motion keeps the static focal box and trajectory", async ({ page }) => {
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
  await expect(page.locator('[data-layer="overlay"]')).toHaveAttribute("data-overlay-commands", "2");
  expect(await overlayAlphaSum(page)).toBeGreaterThan(0);
  expect(errors).toEqual([]);
});

test("sequence-wide rail seeks one-based frames and complete trajectories retain gap breaks", async ({ page }) => {
  const requestedFrames: number[] = [];
  const errors = collectBrowserErrors(page);
  await installTrackedFixtureRoutes(page, requestedFrames);
  await page.goto("/");
  await page.getByLabel("Source", { exact: true }).selectOption(gappedMetadata.source_key);
  await page.getByLabel("Exact track ID").fill("8");
  await page.getByRole("button", { name: "Find track" }).click();

  const rail = page.locator(".track-timeline__rail");
  await expect(rail).toHaveAttribute("aria-valuemin", "1");
  await expect(rail).toHaveAttribute("aria-valuemax", String(gappedMetadata.frame_count));
  await expect(page.locator(".track-timeline__run")).toHaveCount(3);
  await expect(page.locator(".track-timeline__run--singleton")).toHaveCount(3);
  await expect(rail).toHaveAccessibleDescription(/Observed runs: 1; 3; 5\. Missing ranges: 2-2; 4-4\./);
  await page.getByRole("checkbox", { name: "Show future trajectory" }).check();
  await expect(page.locator('[data-layer="overlay"]')).toHaveAttribute("data-overlay-commands", "4");
  await rail.scrollIntoViewIfNeeded();
  const railBox = await rail.boundingBox();
  if (railBox === null) throw new Error("Timeline rail was not visible for pointer seeking");
  await page.mouse.click(railBox.x + railBox.width * 0.25, railBox.y + railBox.height / 2);
  await expect(page.getByLabel("Frame number")).toHaveValue("2");
  await rail.press("ArrowRight");
  await expect(page.getByLabel("Frame number")).toHaveValue("3");
  await rail.press("Shift+ArrowRight");
  await expect(page.getByLabel("Frame number")).toHaveValue(String(gappedMetadata.frame_count));
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

  const count = page.getByRole("spinbutton", { name: "Number of nearby tracks" });
  await expect(count).toHaveValue("0");
  await count.focus();
  await page.keyboard.press("Control+A");
  await page.keyboard.type("3");
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

  await count.focus();
  await page.keyboard.press("Control+A");
  await page.keyboard.type("1");
  await expect(overlay).toHaveAttribute("data-context-commands", "1");
  await page.getByRole("checkbox", { name: "Abrupt displacement" }).focus();
  await page.keyboard.press("Space");
  await expect(page.getByRole("checkbox", { name: "Abrupt displacement" })).toBeChecked();
  await expect(page.locator(".track-timeline__rail")).toHaveAccessibleDescription(/Displacement activity frames 1-2, anchor 2, severity 0.5, 1 raw match/);

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
  await expect(page.locator('[data-layer="overlay"]')).toHaveAttribute("data-overlay-commands", "1");
  expect(await overlayAlphaSum(page)).toBeGreaterThan(0);
  await assertNoHorizontalOverflow(page);
  await page.getByRole("button", { name: "Next observation" }).click();
  await expect(page.getByLabel("Frame number")).toHaveValue("3");
  await page.getByRole("button", { name: "Next gap" }).click();
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
  await page.route(/\/api\/sequences\/synthetic-tracked-continuous\/tracks\?track_id=8(?:&.*)?$/, async (route) => {
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
  const overlay = page.locator('[data-layer="overlay"]');
  for (const count of [3, 1]) {
    await page.getByRole("spinbutton", { name: "Number of nearby tracks" }).fill(String(count));
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
