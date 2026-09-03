import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { FilmstripResponse, SourceMetadata, TimelineEventsResponse, TrackEvidenceResponse } from "./api";
import { FocusReview } from "./FocusReview";

const observation = (frame: number, rowIndex: number) => ({
  source_key: "tracked",
  sequence: "MOT20-01",
  frame,
  row_index: rowIndex,
  row_hash: `row-${rowIndex}`,
  source_hash: "hash-a",
  raw_track_id: 8,
  usable_track_id: 8,
  raw_geometry: { x: 1, y: 1, width: 2, height: 4 },
  display_geometry: { x1: 1, y1: 1, x2: 3, y2: 5 },
  score: 0.8,
  ground_truth: null,
  opaque_result_fields: null,
  score_semantics: "tracker_score" as const,
  ground_truth_semantics: "not_defined" as const,
});

const observations = [observation(1, 1), observation(3, 3), observation(6, 6)];
const evidence = {
  source_key: "tracked",
  sequence: "MOT20-01",
  source_hash: "hash-a",
  track_id: 8,
  observation_frames: [1, 3, 6],
  gaps: [{ start_frame: 2, end_frame: 2, length: 1 }, { start_frame: 4, end_frame: 5, length: 2 }],
  first_observation: observations[0],
  last_observation: observations[2],
  previous_observation: observations[0],
  next_observation: observations[2],
  observations,
} satisfies TrackEvidenceResponse;
const filmstrip = {
  source_key: "tracked",
  sequence: "MOT20-01",
  source_hash: "hash-a",
  track_id: 8,
  current_row_index: 3,
  total_observations: 3,
  sampled_count: 0,
  samples: [],
} satisfies FilmstripResponse;
const events = {
  source_key: "tracked",
  sequence: "MOT20-01",
  source_hash: "hash-a",
  track_id: 8,
  geometry_basis: "raw_xywh",
  settings: {
    displacement_enabled: false,
    displacement_threshold: 0.5,
    displacement_operator: "greater_than_or_equal",
    scale_change_enabled: false,
    scale_change_threshold: 0.5,
    scale_change_operator: "greater_than_or_equal",
    close_interaction_enabled: false,
    close_interaction_threshold: 0.25,
    close_interaction_operator: "less_than_or_equal",
  },
  confidence: {
    status: "absent",
    meaningful: false,
    score_semantics: "not_defined",
    threshold: 0.5,
    threshold_operator: "less_than_or_equal",
    diagnostic: null,
  },
  displacement_events: [],
  scale_change_events: [],
  close_interaction_events: [],
  low_confidence_observations: [],
} satisfies TimelineEventsResponse;
const source = { frame_count: 10 } as SourceMetadata;

describe("FocusReview", () => {
  it("shows exact previous/next evidence at a gap and seeks timeline markers exactly", async () => {
    const onSeek = vi.fn();
    render(<FocusReview contextCount={3} evidence={evidence} events={events} filmstrip={filmstrip} focusStatus="ready" focusTarget={{ trackId: 8, confirmedRowIndex: 3 }} frame={4} mode="focus" onContextCountChange={vi.fn()} onEventSettingsChange={vi.fn()} onExit={vi.fn()} onModeChange={vi.fn()} onSeek={onSeek} source={source} />);

    expect(screen.getByText("Gap at frame 4. Previous observation 3; next observation 6.")).toBeVisible();
    expect(screen.getByLabelText("Track timeline")).toHaveTextContent("Low confidence unavailable: absent.");
    await userEvent.click(screen.getByRole("button", { name: "Gap 4-5, seek frame 4" }));
    expect(onSeek).toHaveBeenCalledWith(4);
  });

  it("shows objective and enabled markers with only meaningful confidence", async () => {
    const onSeek = vi.fn();
    const meaningfulEvents = {
      ...events,
      settings: {
        ...events.settings,
        scale_change_enabled: true,
        scale_change_threshold: 0.75,
      },
      confidence: {
        ...events.confidence,
        status: "meaningful" as const,
        meaningful: true,
        threshold: 0.6,
      },
      displacement_events: [{ from_frame: 1, to_frame: 2, threshold: 0.5 }],
      scale_change_events: [{ from_frame: 2, to_frame: 3, threshold: 0.75 }],
      low_confidence_observations: [observations[2]],
    } satisfies TimelineEventsResponse;
    render(<FocusReview contextCount={3} evidence={evidence} events={meaningfulEvents} filmstrip={filmstrip} focusStatus="ready" focusTarget={{ trackId: 8, confirmedRowIndex: 3 }} frame={4} mode="focus" onContextCountChange={vi.fn()} onEventSettingsChange={vi.fn()} onExit={vi.fn()} onModeChange={vi.fn()} onSeek={onSeek} source={source} />);

    expect(screen.getByRole("button", { name: "First observation, seek frame 1" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Last observation, seek frame 6" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "Enabled event at frame 2, seek frame 2" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Enabled event at frame 3, seek frame 3" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Enabled event at frame 6, seek frame 6" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Gap 4-5, seek frame 4" })).toHaveClass("track-timeline__marker--gap");
    expect(screen.getByRole("button", { name: "Enabled event at frame 3, seek frame 3" })).toHaveClass("track-timeline__marker--heuristic");
    expect(screen.getByLabelText("Track timeline")).toHaveTextContent("Low confidence at or below 0.6");
    expect(screen.getByLabelText("Track timeline")).toHaveTextContent("Scale change >= 0.75");
    expect(screen.getByLabelText("Track timeline")).not.toHaveTextContent("Displacement >=");

    await userEvent.click(screen.getByRole("button", { name: "Previous enabled event" }));
    expect(onSeek).toHaveBeenCalledWith(3);
  });

  it("exposes bounded context and opt-in event controls with visible thresholds", async () => {
    const onModeChange = vi.fn();
    const onContextCountChange = vi.fn();
    const onEventSettingsChange = vi.fn();
    render(<FocusReview contextCount={3} evidence={evidence} events={events} filmstrip={filmstrip} focusStatus="ready" focusTarget={{ trackId: 8, confirmedRowIndex: 3 }} frame={3} mode="focus" onContextCountChange={onContextCountChange} onEventSettingsChange={onEventSettingsChange} onExit={vi.fn()} onModeChange={onModeChange} onSeek={vi.fn()} source={source} />);

    expect(screen.getByRole("radio", { name: "Focus" })).toBeChecked();
    await userEvent.click(screen.getByRole("radio", { name: "Context" }));
    expect(onModeChange).toHaveBeenCalledWith("context");

    const count = screen.getByRole("spinbutton", { name: "Context tracks" });
    expect(count).toHaveAttribute("min", "0");
    expect(count).toHaveAttribute("max", "8");
    expect(count).toHaveValue(3);
    await userEvent.clear(count);
    await userEvent.type(count, "9");
    expect(onContextCountChange).toHaveBeenLastCalledWith(8);

    const displacement = screen.getByRole("checkbox", { name: "Abrupt displacement" });
    const displacementThreshold = screen.getByRole("spinbutton", { name: "Displacement threshold in box heights" });
    expect(displacement).not.toBeChecked();
    expect(displacementThreshold).toHaveValue(0.5);
    expect(displacementThreshold).toBeDisabled();
    await userEvent.click(displacement);
    expect(onEventSettingsChange).toHaveBeenCalledWith({
      ...events.settings,
      displacement_enabled: true,
    });
  });
});