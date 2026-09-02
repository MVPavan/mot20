import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { SourceMetadata } from "./api";
import { TrackSearch } from "./TrackSearch";

const SOURCE = {
  source_key: "mot20-01",
  source_hash: "hash-a",
  capability: { track_features: true },
} as SourceMetadata;

describe("TrackSearch", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("shows capability unavailable without offering a search request", () => {
    render(<TrackSearch onSelect={vi.fn()} source={{ ...SOURCE, capability: { ...SOURCE.capability, track_features: false } }} />);
    expect(screen.getByText("Track ID search unavailable for this source.")).toBeVisible();
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("reports an exact sequence-local no-result state", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 404 }));
    render(<TrackSearch onSelect={vi.fn()} source={SOURCE} />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Exact track ID"), "80");
    await user.click(screen.getByRole("button", { name: "Find track" }));
    expect(await screen.findByText("Track 80 is not present in this sequence.")).toBeVisible();
  });

  it("aborts and ignores an older exact-ID search response", async () => {
    let resolveFirst!: (response: Response) => void;
    let firstSignal: AbortSignal | undefined;
    const firstResponse = new Promise<Response>((resolve) => { resolveFirst = resolve; });
    vi.stubGlobal("fetch", vi.fn().mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const trackId = Number(new URL(String(input), "http://viewer.test").searchParams.get("track_id"));
      if (trackId === 8) {
        firstSignal = init?.signal ?? undefined;
        return firstResponse;
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({
          source_key: SOURCE.source_key,
          source_hash: SOURCE.source_hash,
          track_id: trackId,
        }),
      });
    }));
    const onSelect = vi.fn();
    render(<TrackSearch onSelect={onSelect} source={SOURCE} />);
    const user = userEvent.setup();

    await user.type(screen.getByLabelText("Exact track ID"), "8");
    await user.keyboard("{Enter}");
    await user.clear(screen.getByLabelText("Exact track ID"));
    await user.type(screen.getByLabelText("Exact track ID"), "9{Enter}");

    await waitFor(() => expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ track_id: 9 })));
    expect(firstSignal?.aborted).toBe(true);
    await act(async () => {
      resolveFirst({
        ok: true,
        json: async () => ({
          source_key: SOURCE.source_key,
          source_hash: SOURCE.source_hash,
          track_id: 8,
        }),
      } as Response);
      await firstResponse;
    });

    expect(onSelect).toHaveBeenCalledTimes(1);
  });
});