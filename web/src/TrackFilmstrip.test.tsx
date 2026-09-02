import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { FilmstripResponse, SourceMetadata } from "./api";
import { TrackFilmstrip } from "./TrackFilmstrip";

describe("TrackFilmstrip", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("shows an explicit missing crop state", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 404 }));
    const observation = {
      source_key: "tracked", sequence: "MOT20-01", frame: 3, row_index: 3,
      row_hash: "row-3", source_hash: "hash-a", raw_track_id: 8, usable_track_id: 8,
      raw_geometry: { x: 1, y: 1, width: 2, height: 4 },
      display_geometry: { x1: 1, y1: 1, x2: 3, y2: 5 }, score: 0.8,
      ground_truth: null, opaque_result_fields: null, score_semantics: "tracker_score" as const,
      ground_truth_semantics: "not_defined" as const,
    };
    const filmstrip = {
      source_key: "tracked", sequence: "MOT20-01", source_hash: "hash-a", track_id: 8,
      current_row_index: 3, total_observations: 1, sampled_count: 1,
      samples: [{ is_current: true, observation }],
    } satisfies FilmstripResponse;
    render(<TrackFilmstrip filmstrip={filmstrip} onSeek={vi.fn()} source={{ source_key: "tracked", source_hash: "hash-a" } as SourceMetadata} />);

    expect(await screen.findByText("Crop unavailable")).toBeVisible();
  });

  it("rejects a crop response from a stale source identity", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        source_key: "stale-source",
        sequence: "MOT20-01",
        source_hash: "stale-hash",
        frame: 3,
        row_index: 3,
        row_hash: "row-3",
        media_type: "image/jpeg",
        image_base64: "crop",
      }),
    }));
    const observation = {
      source_key: "tracked", sequence: "MOT20-01", frame: 3, row_index: 3,
      row_hash: "row-3", source_hash: "hash-a", raw_track_id: 8, usable_track_id: 8,
      raw_geometry: { x: 1, y: 1, width: 2, height: 4 },
      display_geometry: { x1: 1, y1: 1, x2: 3, y2: 5 }, score: 0.8,
      ground_truth: null, opaque_result_fields: null, score_semantics: "tracker_score" as const,
      ground_truth_semantics: "not_defined" as const,
    };
    const filmstrip = {
      source_key: "tracked", sequence: "MOT20-01", source_hash: "hash-a", track_id: 8,
      current_row_index: 3, total_observations: 1, sampled_count: 1,
      samples: [{ is_current: true, observation }],
    } satisfies FilmstripResponse;

    render(<TrackFilmstrip filmstrip={filmstrip} onSeek={vi.fn()} source={{ source_key: "tracked", source_hash: "hash-a" } as SourceMetadata} />);

    expect(await screen.findByText("Crop unavailable")).toBeVisible();
  });
});