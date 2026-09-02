import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchTimelineEvents, fetchTrackContext, fetchTrackFilmstrip, fetchTrackSearch, type EventSettings, type SourceMetadata } from "./api";

const SOURCE = {
  source_key: "mot20-01/gt",
  source_hash: "hash value",
} as SourceMetadata;

describe("track API", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("searches an exact sequence-local track ID under the source hash", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        source_key: SOURCE.source_key,
        sequence: "MOT20-01",
        source_hash: SOURCE.source_hash,
        track_id: 8,
        observation_frames: [3],
        gaps: [],
        first_observation: { row_index: 2 },
        last_observation: { row_index: 2 },
        previous_observation: null,
        next_observation: null,
        observations: [{ row_index: 2 }],
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchTrackSearch(SOURCE, 8);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/sequences/mot20-01%2Fgt/tracks?track_id=8&source_hash=hash+value",
      { signal: undefined },
    );
    expect(result.track_id).toBe(8);
  });

  it("rejects a filmstrip response above the 64-sample API cap", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          source_key: SOURCE.source_key,
          sequence: "MOT20-01",
          source_hash: SOURCE.source_hash,
          track_id: 8,
          current_row_index: 4,
          total_observations: 100,
          sampled_count: 65,
          samples: Array.from({ length: 65 }, () => ({ is_current: false, observation: {} })),
        }),
      }),
    );

    await expect(fetchTrackFilmstrip(SOURCE, 8, 4)).rejects.toThrow(
      "Filmstrip response exceeded 64 samples",
    );
  });

  it("requests frame-local context and caps the count at eight", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        source_key: SOURCE.source_key,
        sequence: "MOT20-01",
        source_hash: SOURCE.source_hash,
        track_id: 8,
        geometry_basis: "raw_xywh",
        window: { center_frame: 12, start_frame: 9, end_frame: 15, radius: 3 },
        requested_count: 8,
        hard_cap: 8,
        total_competitors: 0,
        competitors: [],
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchTrackContext(SOURCE, 8, 12, 99);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/sequences/mot20-01%2Fgt/tracks/8/context?frame=12&count=8&source_hash=hash+value",
      { signal: undefined },
    );
    expect(result.hard_cap).toBe(8);
  });

  it("sends explicit optional event settings", async () => {
    const settings: EventSettings = {
      displacement_enabled: true,
      displacement_threshold: 0.75,
      displacement_operator: "greater_than_or_equal",
      scale_change_enabled: false,
      scale_change_threshold: 0.5,
      scale_change_operator: "greater_than_or_equal",
      close_interaction_enabled: true,
      close_interaction_threshold: 0.2,
      close_interaction_operator: "less_than_or_equal",
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ settings }) });
    vi.stubGlobal("fetch", fetchMock);

    await fetchTimelineEvents(SOURCE, 8, settings);

    const requestUrl = String(fetchMock.mock.calls[0][0]);
    expect(requestUrl).toContain("enable_displacement=true");
    expect(requestUrl).toContain("displacement_threshold=0.75");
    expect(requestUrl).toContain("enable_scale_change=false");
    expect(requestUrl).toContain("enable_close_interaction=true");
    expect(requestUrl).toContain("close_interaction_threshold=0.2");
  });
});