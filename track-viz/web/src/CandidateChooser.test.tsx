import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Observation, SourceMetadata } from "./api";
import { CandidateChooser } from "./CandidateChooser";

const source = {
  source_key: "tracked",
  source_hash: "hash-a",
} as SourceMetadata;

const observation = {
  source_key: source.source_key,
  sequence: "MOT20-01",
  frame: 3,
  row_index: 3,
  row_hash: "row-3",
  source_hash: source.source_hash,
  raw_track_id: -1,
  usable_track_id: null,
  raw_geometry: { x: 1, y: 1, width: 2, height: 4 },
  display_geometry: { x1: 1, y1: 1, x2: 3, y2: 5 },
  score: 0.8,
  ground_truth: null,
  opaque_result_fields: null,
  score_semantics: "tracker_score",
  ground_truth_semantics: "not_defined",
} satisfies Observation;

describe("CandidateChooser", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("rejects a current crop response from another source", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        source_key: "stale-source",
        sequence: observation.sequence,
        source_hash: source.source_hash,
        frame: observation.frame,
        row_index: observation.row_index,
        row_hash: observation.row_hash,
        media_type: "image/jpeg",
        image_base64: "crop",
      }),
    }));

    render(
      <CandidateChooser
        activeRowIndex={observation.row_index}
        candidates={[observation]}
        onActivate={vi.fn()}
        onConfirm={vi.fn()}
        onFocus={vi.fn()}
        source={source}
      />,
    );

    expect(await screen.findByText("Crop unavailable")).toBeVisible();
  });

  it("exposes active-descendant listbox semantics and keyboard activation", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        source_key: source.source_key,
        sequence: observation.sequence,
        source_hash: source.source_hash,
        frame: observation.frame,
        row_index: observation.row_index,
        row_hash: observation.row_hash,
        media_type: "image/jpeg",
        image_base64: "crop",
      }),
    }));
    const second = { ...observation, row_index: 4, row_hash: "row-4" };
    const onActivate = vi.fn();
    const onConfirm = vi.fn();
    const onFocus = vi.fn();
    render(
      <CandidateChooser
        activeRowIndex={observation.row_index}
        candidates={[observation, second]}
        onActivate={onActivate}
        onConfirm={onConfirm}
        onFocus={onFocus}
        source={source}
      />,
    );

    const listbox = screen.getByRole("listbox", { name: "Observation candidates" });
    expect(listbox).toHaveAttribute("tabindex", "0");
    expect(listbox).toHaveAttribute("aria-activedescendant", "candidate-3");
    expect(screen.getByText("2 candidates pinned.")).toHaveAttribute("aria-live", "polite");

    listbox.focus();
    await userEvent.keyboard("{ArrowDown}{Enter}");
    expect(onActivate).toHaveBeenCalledWith(4);
    expect(onConfirm).toHaveBeenCalledOnce();
    expect(onFocus).toHaveBeenCalledWith(observation);
  });
});