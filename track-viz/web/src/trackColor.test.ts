import { describe, expect, it } from "vitest";

import { trackColor } from "./trackColor";

describe("track color", () => {
  it("matches the shared backend golden vectors", () => {
    expect(trackColor("MOT20-01", 1)).toBe("#3ae65a");
    expect(trackColor("MOT20-01", 8)).toBe("#e6c43a");
    expect(trackColor("MOT20-06", 8)).toBe("#e63a93");
  });
});