import { describe, expect, it } from "vitest";

import {
  activityNavigationTarget,
  deriveEventActivities,
  lowConfidenceNavigationTarget,
} from "./eventEpisodes";

describe("event episodes", () => {
  it("merges only contiguous displacement transitions and anchors at the greatest severity", () => {
    const activities = deriveEventActivities({
      displacement_events: [
        { from_frame: 1, to_frame: 2, frame_delta: 1, normalized_displacement: 0.03 },
        { from_frame: 2, to_frame: 3, frame_delta: 1, normalized_displacement: 0.08 },
        { from_frame: 3, to_frame: 5, frame_delta: 2, normalized_displacement: 0.4 },
        { from_frame: 5, to_frame: 6, frame_delta: 1, normalized_displacement: 0.2 },
      ],
      scale_change_events: [],
      close_interaction_events: [],
    }).displacement;

    expect(activities).toEqual([
      expect.objectContaining({ startFrame: 1, endFrame: 3, anchorFrame: 3, severity: 0.08, rawMatchCount: 2 }),
      expect.objectContaining({ startFrame: 3, endFrame: 5, anchorFrame: 5, severity: 0.4, rawMatchCount: 1 }),
      expect.objectContaining({ startFrame: 5, endFrame: 6, anchorFrame: 6, severity: 0.2, rawMatchCount: 1 }),
    ]);
  });

  it("breaks equal displacement and scale severity ties at the earliest anchor frame", () => {
    const activities = deriveEventActivities({
      displacement_events: [
        { from_frame: 1, to_frame: 2, frame_delta: 1, normalized_displacement: 0.1 },
        { from_frame: 2, to_frame: 3, frame_delta: 1, normalized_displacement: 0.1 },
      ],
      scale_change_events: [
        { from_frame: 5, to_frame: 6, frame_delta: 1, normalized_scale_change: 0.2 },
        { from_frame: 6, to_frame: 7, frame_delta: 1, normalized_scale_change: 0.2 },
      ],
      close_interaction_events: [],
    });

    expect(activities.displacement[0]).toMatchObject({ anchorFrame: 2, severity: 0.1 });
    expect(activities.scaleChange[0]).toMatchObject({ anchorFrame: 6, severity: 0.2 });
  });

  it("groups continuous proximity per competitor and breaks equal anchors deterministically", () => {
    const activities = deriveEventActivities({
      displacement_events: [],
      scale_change_events: [],
      close_interaction_events: [
        { frame: 2, competitor_track_id: 9, normalized_edge_proximity: 0.2 },
        { frame: 3, competitor_track_id: 9, normalized_edge_proximity: 0.1 },
        { frame: 4, competitor_track_id: 9, normalized_edge_proximity: 0.1 },
        { frame: 3, competitor_track_id: 4, normalized_edge_proximity: 0.1 },
      ],
    }).proximity;

    expect(activities).toEqual([
      expect.objectContaining({ competitorTrackId: 4, startFrame: 3, endFrame: 3, anchorFrame: 3, severity: 0.1, rawMatchCount: 1 }),
      expect.objectContaining({ competitorTrackId: 9, startFrame: 2, endFrame: 4, anchorFrame: 3, severity: 0.1, rawMatchCount: 3 }),
    ]);
  });

  it("skips an activity containing the current frame and keeps low confidence per-frame", () => {
    const activities = deriveEventActivities({
      displacement_events: [
        { from_frame: 1, to_frame: 2, frame_delta: 1, normalized_displacement: 0.03 },
        { from_frame: 2, to_frame: 3, frame_delta: 1, normalized_displacement: 0.04 },
        { from_frame: 6, to_frame: 7, frame_delta: 1, normalized_displacement: 0.05 },
      ],
      scale_change_events: [],
      close_interaction_events: [],
    }).displacement;

    expect(activityNavigationTarget(activities, 2, -1)).toBeNull();
    expect(activityNavigationTarget(activities, 2, 1)).toBe(7);
    expect(lowConfidenceNavigationTarget([2, 2, 4], 2, 1)).toBe(4);
    expect(lowConfidenceNavigationTarget([2, 2, 4], 3, -1)).toBe(2);
  });
});
