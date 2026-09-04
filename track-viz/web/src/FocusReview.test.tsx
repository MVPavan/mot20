import { render, screen, waitFor } from "@testing-library/react";
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
const eventProps = {
  eventError: "",
  eventSettings: events.settings,
  eventStatus: "idle" as const,
  onRetryEvents: vi.fn(),
  onTrajectoryModeChange: vi.fn(),
  trajectoryMode: "past" as const,
};

describe("FocusReview", () => {
  it("reports a failed track-evidence request even when no evidence was retained", () => {
    render(<FocusReview {...eventProps} contextCount={3} evidence={null} events={null} filmstrip={null} focusStatus="error" focusTarget={{ trackId: 8, confirmedRowIndex: 3 }} frame={3} onContextCountChange={vi.fn()} onEventSettingsChange={vi.fn()} onExit={vi.fn()} onSeek={vi.fn()} source={source} />);

    expect(screen.getByRole("alert")).toHaveTextContent("Track evidence could not be loaded.");
    expect(screen.queryByText("Loading track evidence")).not.toBeInTheDocument();
  });

  it("shows exact previous/next evidence at a gap and seeks the sequence-wide timeline exactly", async () => {
    const onSeek = vi.fn();
    const { container } = render(<FocusReview {...eventProps} contextCount={3} evidence={evidence} events={events} filmstrip={filmstrip} focusStatus="ready" focusTarget={{ trackId: 8, confirmedRowIndex: 3 }} frame={4} onContextCountChange={vi.fn()} onEventSettingsChange={vi.fn()} onExit={vi.fn()} onSeek={onSeek} source={source} />);

    expect(screen.getByText("Gap at frame 4. Previous observation 3; next observation 6.")).toBeVisible();
    expect(screen.getByLabelText("Track timeline")).toHaveTextContent("Low confidence unavailable: absent.");
    const rail = screen.getByRole("slider", { name: "Sequence timeline, current frame 4" });
    expect(rail).toHaveAttribute("aria-valuemin", "1");
    expect(rail).toHaveAttribute("aria-valuemax", "10");
    expect(rail).toHaveAttribute("aria-valuenow", "4");
    expect(rail).toHaveAccessibleDescription(/Observed runs: 1; 3; 6\. Missing ranges: 2-2; 4-5\./);
    await userEvent.click(rail);
    await userEvent.keyboard("{ArrowRight}");
    expect(onSeek).toHaveBeenCalledWith(5);
    expect(container.querySelector(".track-timeline__run--singleton")).toBeVisible();
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
      displacement_events: [{ from_frame: 1, to_frame: 2, from_row_index: 1, to_row_index: 2, frame_delta: 1, center_displacement_pixels: 2, normalization_box_height: 4, normalized_displacement: 0.5, threshold: 0.5 }],
      scale_change_events: [{ from_frame: 2, to_frame: 3, from_row_index: 2, to_row_index: 3, frame_delta: 1, absolute_height_change_pixels: 3, normalization_box_height: 4, normalized_scale_change: 0.75, threshold: 0.75 }],
      low_confidence_observations: [observations[2]],
    } satisfies TimelineEventsResponse;
    const { container } = render(<FocusReview {...eventProps} eventSettings={meaningfulEvents.settings} contextCount={3} evidence={evidence} events={meaningfulEvents} filmstrip={filmstrip} focusStatus="ready" focusTarget={{ trackId: 8, confirmedRowIndex: 3 }} frame={4} onContextCountChange={vi.fn()} onEventSettingsChange={vi.fn()} onExit={vi.fn()} onSeek={onSeek} source={source} />);

    expect(container.querySelectorAll(".track-timeline__endpoint")).toHaveLength(2);
    expect(container.querySelector(".track-timeline__glyph--activity")).toBeVisible();
    expect(container.querySelector(".track-timeline__glyph--low-confidence")).toBeVisible();
    expect(container.querySelectorAll(".track-timeline__gap")).toHaveLength(2);
    expect(screen.getByRole("slider")).toHaveAccessibleDescription(/Scale change activity frames 2-3/);
    expect(screen.getByLabelText("Track timeline")).toHaveTextContent("Low confidence at or below 0.6");
    expect(screen.getByLabelText("Track timeline")).toHaveTextContent("Scale change >= 0.75");
    expect(screen.getByLabelText("Track timeline")).not.toHaveTextContent("Displacement >=");

    await userEvent.click(screen.getByRole("button", { name: "Previous Scale change activity" }));
    expect(onSeek).toHaveBeenCalledWith(3);
  });

  it("keeps family navigation separate, explains unavailable controls, and exposes current activities", async () => {
    const onSeek = vi.fn();
    const groupedEvents = {
      ...events,
      settings: {
        ...events.settings,
        displacement_enabled: true,
        displacement_threshold: 0,
        close_interaction_enabled: true,
        close_interaction_threshold: 0,
      },
      confidence: { ...events.confidence, status: "meaningful" as const, meaningful: true, threshold: 0.5 },
      displacement_events: [
        { from_frame: 1, to_frame: 2, from_row_index: 1, to_row_index: 2, frame_delta: 1, center_displacement_pixels: 0.12, normalization_box_height: 4, normalized_displacement: 0.03, threshold: 0.02 },
        { from_frame: 2, to_frame: 3, from_row_index: 2, to_row_index: 3, frame_delta: 1, center_displacement_pixels: 0.16, normalization_box_height: 4, normalized_displacement: 0.04, threshold: 0.02 },
        { from_frame: 6, to_frame: 7, from_row_index: 6, to_row_index: 7, frame_delta: 1, center_displacement_pixels: 0.2, normalization_box_height: 4, normalized_displacement: 0.05, threshold: 0.02 },
      ],
      close_interaction_events: [
        { frame: 1, focal_row_index: 1, competitor_track_id: 9, competitor_row_index: 91, edge_distance_pixels: 1, focal_box_height: 4, normalized_edge_proximity: 0.25, threshold: 0.25 },
        { frame: 2, focal_row_index: 2, competitor_track_id: 9, competitor_row_index: 92, edge_distance_pixels: 0.8, focal_box_height: 4, normalized_edge_proximity: 0.2, threshold: 0.25 },
        { frame: 3, focal_row_index: 3, competitor_track_id: 9, competitor_row_index: 93, edge_distance_pixels: 0.6, focal_box_height: 4, normalized_edge_proximity: 0.15, threshold: 0.25 },
      ],
      low_confidence_observations: [observations[0], observations[2]],
    } satisfies TimelineEventsResponse;
    render(<FocusReview {...eventProps} eventSettings={groupedEvents.settings} contextCount={3} evidence={evidence} events={groupedEvents} filmstrip={filmstrip} focusStatus="ready" focusTarget={{ trackId: 8, confirmedRowIndex: 3 }} frame={2} onContextCountChange={vi.fn()} onEventSettingsChange={vi.fn()} onExit={vi.fn()} onSeek={onSeek} source={source} />);

    expect(screen.getByLabelText("Displacement activity controls")).toHaveTextContent("2 activities / 3 raw matches");
    expect(screen.getByLabelText("Proximity activity controls")).toHaveTextContent("1 activity / 3 raw matches");
    expect(screen.getByRole("button", { name: "Next Displacement activity" })).toBeEnabled();
    await userEvent.click(screen.getByRole("button", { name: "Next Displacement activity" }));
    expect(onSeek).toHaveBeenCalledWith(7);
    expect(screen.getByRole("button", { name: "Next Scale change activity" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Next Scale change activity" })).toHaveAccessibleDescription("Scale change is off.");
    expect(screen.getByLabelText("Scale change activity controls")).toHaveTextContent("Scale change is off.");
    expect(screen.getByRole("button", { name: "Previous Displacement activity" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Previous Displacement activity" })).toHaveAccessibleDescription("No previous displacement activity.");
    expect(screen.getByRole("button", { name: "Next Proximity activity" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Next Proximity activity" })).toHaveAccessibleDescription("No next proximity activity.");
    expect(screen.getByLabelText("Displacement activity controls")).toHaveTextContent("A zero displacement threshold matches every valid transition because the backend uses >=.");
    expect(screen.getByLabelText("Proximity activity controls")).toHaveTextContent("A zero proximity threshold can match touching or overlapping boxes because edge distance can be zero.");
    expect(screen.getByLabelText("Current event activities")).toHaveTextContent("Displacement: frames 1-3; anchor 3; severity 0.04; 2 raw matches.");
    expect(screen.getByLabelText("Current event activities")).toHaveTextContent("Proximity with competitor 9: frames 1-3; anchor 3; severity 0.15; 3 raw matches.");
    await userEvent.click(screen.getByRole("button", { name: "Next low-confidence observation" }));
    expect(onSeek).toHaveBeenLastCalledWith(6);
  });

  it("keeps an enabled default scale control visibly and accessibly disabled when no raw match exists", () => {
    const onSeek = vi.fn();
    const noScaleMatch = {
      ...events,
      settings: { ...events.settings, scale_change_enabled: true },
    } satisfies TimelineEventsResponse;
    render(<FocusReview {...eventProps} eventSettings={noScaleMatch.settings} contextCount={3} evidence={evidence} events={noScaleMatch} filmstrip={filmstrip} focusStatus="ready" focusTarget={{ trackId: 8, confirmedRowIndex: 3 }} frame={3} onContextCountChange={vi.fn()} onEventSettingsChange={vi.fn()} onExit={vi.fn()} onSeek={onSeek} source={source} />);

    const previous = screen.getByRole("button", { name: "Previous Scale change activity" });
    const next = screen.getByRole("button", { name: "Next Scale change activity" });
    expect(screen.getByLabelText("Scale change activity controls")).toHaveTextContent("0 activities / 0 raw matches.");
    expect(screen.getByLabelText("Scale change activity controls")).toHaveTextContent("No matching scale change activity.");
    expect(previous).toBeDisabled();
    expect(next).toBeDisabled();
    expect(previous).toHaveAccessibleDescription("No matching scale change activity.");
    expect(next).toHaveAccessibleDescription("No matching scale change activity.");
  });

  it("uses nearby-track count and a compact future-trajectory toggle", async () => {
    const onContextCountChange = vi.fn();
    const onEventSettingsChange = vi.fn();
    const onTrajectoryModeChange = vi.fn();
    render(<FocusReview {...eventProps} contextCount={0} evidence={evidence} events={events} filmstrip={filmstrip} focusStatus="ready" focusTarget={{ trackId: 8, confirmedRowIndex: 3 }} frame={3} onContextCountChange={onContextCountChange} onEventSettingsChange={onEventSettingsChange} onExit={vi.fn()} onSeek={vi.fn()} onTrajectoryModeChange={onTrajectoryModeChange} source={source} />);

    expect(screen.queryByRole("radio", { name: "Focus" })).not.toBeInTheDocument();
    expect(screen.queryByRole("radio", { name: "Context" })).not.toBeInTheDocument();
    const futureTrajectory = screen.getByRole("checkbox", { name: "Show future trajectory" });
    expect(futureTrajectory).not.toBeChecked();
    await userEvent.click(futureTrajectory);
    expect(onTrajectoryModeChange).toHaveBeenCalledWith("complete");

    const count = screen.getByRole("spinbutton", { name: "Number of nearby tracks" });
    expect(count).toHaveAttribute("min", "0");
    expect(count).toHaveAttribute("max", "8");
    expect(count).toHaveValue(0);
    await userEvent.clear(count);
    await userEvent.type(count, "9");
    expect(onContextCountChange).toHaveBeenLastCalledWith(8);

    const displacement = screen.getByRole("checkbox", { name: "Abrupt displacement" });
    const displacementThreshold = screen.getByRole("spinbutton", { name: "Displacement threshold in box heights" });
    expect(displacement).not.toBeChecked();
    expect(displacementThreshold).toHaveValue("0.5");
    expect(displacementThreshold).toBeDisabled();
    await userEvent.click(displacement);
    const update = onEventSettingsChange.mock.calls[0][0] as (settings: typeof events.settings) => typeof events.settings;
    expect(update(events.settings)).toEqual({
      ...events.settings,
      displacement_enabled: true,
    });
  });

  it("keeps a decimal threshold draft intact until its debounce commits", async () => {
    const onEventSettingsChange = vi.fn();
    const enabledEvents = {
      ...events,
      settings: { ...events.settings, displacement_enabled: true },
    } satisfies TimelineEventsResponse;
    const user = userEvent.setup();
    render(<FocusReview {...eventProps} eventSettings={enabledEvents.settings} contextCount={3} evidence={evidence} events={enabledEvents} filmstrip={filmstrip} focusStatus="ready" focusTarget={{ trackId: 8, confirmedRowIndex: 3 }} frame={3} onContextCountChange={vi.fn()} onEventSettingsChange={onEventSettingsChange} onExit={vi.fn()} onSeek={vi.fn()} source={source} />);

    const threshold = screen.getByRole("spinbutton", { name: "Displacement threshold in box heights" });
    await user.clear(threshold);
    await user.type(threshold, "0.02");

    expect(threshold).toHaveValue("0.02");
    expect(onEventSettingsChange).not.toHaveBeenCalled();
    await waitFor(() => expect(onEventSettingsChange).toHaveBeenCalledTimes(1));
    const update = onEventSettingsChange.mock.calls[0][0] as (settings: typeof enabledEvents.settings) => typeof enabledEvents.settings;
    expect(update(enabledEvents.settings)).toEqual({
      ...enabledEvents.settings,
      displacement_threshold: 0.02,
    });
  });

  it("does not recommit a debounced threshold when its unchanged draft then blurs", async () => {
    const onEventSettingsChange = vi.fn();
    const enabledEvents = {
      ...events,
      settings: { ...events.settings, displacement_enabled: true },
    } satisfies TimelineEventsResponse;
    const user = userEvent.setup();
    render(<FocusReview {...eventProps} eventSettings={enabledEvents.settings} contextCount={3} evidence={evidence} events={enabledEvents} filmstrip={filmstrip} focusStatus="ready" focusTarget={{ trackId: 8, confirmedRowIndex: 3 }} frame={3} onContextCountChange={vi.fn()} onEventSettingsChange={onEventSettingsChange} onExit={vi.fn()} onSeek={vi.fn()} source={source} />);

    const threshold = screen.getByRole("spinbutton", { name: "Displacement threshold in box heights" });
    await user.clear(threshold);
    await user.type(threshold, "0.02");
    await waitFor(() => expect(onEventSettingsChange).toHaveBeenCalledTimes(1));
    await user.tab();

    expect(onEventSettingsChange).toHaveBeenCalledTimes(1);
  });

  it("retains review evidence while event data updates and exposes refresh failures in its status slot", async () => {
    const onRetryEvents = vi.fn();
    const props = { ...eventProps, contextCount: 3, evidence, events, filmstrip, focusStatus: "ready" as const, focusTarget: { trackId: 8, confirmedRowIndex: 3 }, frame: 3, onContextCountChange: vi.fn(), onEventSettingsChange: vi.fn(), onExit: vi.fn(), onSeek: vi.fn(), source };
    const { rerender } = render(<FocusReview {...props} eventStatus="updating" onRetryEvents={onRetryEvents} />);

    expect(screen.getByRole("heading", { name: "Track 8" })).toBeVisible();
    expect(screen.getByLabelText("Track timeline")).toBeVisible();
    expect(screen.getByRole("status")).toHaveTextContent("Updating review events");
    rerender(<FocusReview {...props} eventError="Event refresh failed; showing the last successful settings." eventStatus="error" onRetryEvents={onRetryEvents} />);
    expect(screen.getByRole("status")).toHaveTextContent("Event refresh failed");
    await userEvent.click(screen.getByRole("button", { name: "Retry event refresh" }));
    expect(onRetryEvents).toHaveBeenCalledOnce();
  });

  it("reverts invalid drafts on blur and announces why no refresh was sent", async () => {
    const onEventSettingsChange = vi.fn();
    const enabledEvents = {
      ...events,
      settings: { ...events.settings, displacement_enabled: true },
    } satisfies TimelineEventsResponse;
    render(<FocusReview {...eventProps} eventSettings={enabledEvents.settings} contextCount={3} evidence={evidence} events={enabledEvents} filmstrip={filmstrip} focusStatus="ready" focusTarget={{ trackId: 8, confirmedRowIndex: 3 }} frame={3} onContextCountChange={vi.fn()} onEventSettingsChange={onEventSettingsChange} onExit={vi.fn()} onSeek={vi.fn()} source={source} />);

    const threshold = screen.getByRole("spinbutton", { name: "Displacement threshold in box heights" });
    await userEvent.clear(threshold);
    await userEvent.tab();

    expect(threshold).toHaveValue("0.5");
    expect(screen.getByRole("status")).toHaveTextContent("Displacement: Enter a threshold from 0 through 100.");
    expect(onEventSettingsChange).not.toHaveBeenCalled();
  });

  it("uses functional settings updates so rapid checkbox changes retain every family", async () => {
    const onEventSettingsChange = vi.fn();
    render(<FocusReview {...eventProps} contextCount={3} evidence={evidence} events={events} filmstrip={filmstrip} focusStatus="ready" focusTarget={{ trackId: 8, confirmedRowIndex: 3 }} frame={3} onContextCountChange={vi.fn()} onEventSettingsChange={onEventSettingsChange} onExit={vi.fn()} onSeek={vi.fn()} source={source} />);

    await userEvent.click(screen.getByRole("checkbox", { name: "Abrupt displacement" }));
    await userEvent.click(screen.getByRole("checkbox", { name: "Scale change" }));

    const enableDisplacement = onEventSettingsChange.mock.calls[0][0] as (settings: typeof events.settings) => typeof events.settings;
    const enableScale = onEventSettingsChange.mock.calls[1][0] as (settings: typeof events.settings) => typeof events.settings;
    expect(enableScale(enableDisplacement(events.settings))).toMatchObject({
      displacement_enabled: true,
      scale_change_enabled: true,
    });
  });
});
