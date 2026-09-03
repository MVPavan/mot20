import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type { SourceMetadata } from "./api";

const TEST_SOURCE: SourceMetadata = {
  source_key: "mot20-06-joco-v1",
  sequence: "MOT20-06",
  frame_numbering: "one_based",
  frame_count: 5,
  width: 1920,
  height: 1080,
  frame_rate: 25,
  source_hash: "abc123def456",
  adapter: "mot_result_10",
  source_class: "tracker_result",
  policy_classification: "local_test_adapted_development_material",
  source_row_count: 12,
  observation_count: 12,
  capability: {
    id_status: "sentinel_only",
    track_features: false,
    usable_track_ids: [],
    diagnostics: [],
  },
  provenance: {
    producer: "joco_v1",
    detector: "YOLOX-X",
    checkpoint: null,
    tracker: null,
    post_processing: null,
    adaptation_iterations: 3,
    notes: null,
  },
  diagnostics: [],
};

describe("App", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "createImageBitmap",
      vi.fn().mockResolvedValue({ width: 320, height: 180, close: vi.fn() }),
    );
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url === "/api/source-selection" && init?.method === "POST") {
          return {
            ok: true,
            json: async () => ({ sources: [TEST_SOURCE], unavailable: [], diagnostics: [] }),
          };
        }
        if (url === "/api/source-selection") {
          return {
            ok: true,
            json: async () => ({
              images: "/srv/mot20/MOT20-06/img1",
              annotations: "/srv/predictions/MOT20-06.txt",
            }),
          };
        }
        if (url.startsWith("/api/source-path-suggestions")) {
          return { ok: true, json: async () => ({ suggestions: [] }) };
        }
        if (url === "/api/sequences") {
          return {
            ok: true,
            json: async () => ({ sources: [TEST_SOURCE], unavailable: [], diagnostics: [] }),
          };
        }
        if (url.includes("/observations?")) {
          const frame = Number(url.match(/frames\/(\d+)/)?.[1]);
          return {
            ok: true,
            json: async () => ({
              source_key: TEST_SOURCE.source_key,
              sequence: TEST_SOURCE.sequence,
              frame,
              frame_numbering: "one_based",
              source_hash: TEST_SOURCE.source_hash,
              observations: [],
            }),
          };
        }
        if (url.includes("/frames/")) {
          return { ok: true, blob: async () => new Blob(["jpeg"]) };
        }
        throw new Error(`Unexpected request: ${url}`);
      }),
    );
  });

  it("loads a source from editable server paths", async () => {
    const user = userEvent.setup();
    render(<App />);

    const images = await screen.findByLabelText("Images folder on server");
    const annotations = screen.getByLabelText("Predictions or ground truth file on server");
    expect(images).toHaveValue("/srv/mot20/MOT20-06/img1");
    expect(annotations).toHaveValue("/srv/predictions/MOT20-06.txt");

    await user.clear(images);
    await user.type(images, "/data/MOT20-08/img1");
    await user.clear(annotations);
    await user.type(annotations, "/data/predictions/MOT20-08.txt");
    await user.click(screen.getByRole("button", { name: "Load source" }));

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        "/api/source-selection",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({
            images: "/data/MOT20-08/img1",
            annotations: "/data/predictions/MOT20-08.txt",
          }),
        }),
      );
    });
    expect(await screen.findByLabelText("Frame number")).toHaveValue(1);
  });

  it("browses full server filesystem paths and navigates to their parent", async () => {
    const originalFetch = vi.mocked(fetch).getMockImplementation();
    vi.mocked(fetch).mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.includes("/api/source-path-suggestions")) {
        const query = new URL(url, "http://viewer.local").searchParams.get("query");
        if (query === "/") {
          return {
            ok: true,
            json: async () => ({
              kind: "images",
              query,
              directory: "/",
              parent: null,
              entries: [{ path: "/data", entry_type: "directory" }],
              suggestions: ["/data"],
            }),
          } as Response;
        }
        return {
          ok: true,
          json: async () => ({
            kind: "images",
            query,
            directory: "/data/mot20/MOT20-06",
            parent: "/data/mot20",
            entries: [
              {
                path: "/data/mot20/MOT20-06/a-very-long-server-directory-name/img1",
                entry_type: "directory",
              },
            ],
            suggestions: [],
          }),
        } as Response;
      }
      return originalFetch!(input, init);
    });
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Browse server images" }));
    await user.click(await screen.findByText("/data"));
    expect(
      await screen.findByText("/data/mot20/MOT20-06/a-very-long-server-directory-name/img1"),
    ).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Go to parent directory" }));

    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining("query=%2Fdata%2Fmot20"),
      expect.anything(),
    );
  });

  it("keeps the active source when server paths are rejected", async () => {
    const successfulFetch = vi.mocked(fetch).getMockImplementation();
    vi.mocked(fetch).mockImplementation(async (input, init) => {
      if (String(input) === "/api/source-selection" && init?.method === "POST") {
        return {
          ok: false,
          status: 400,
          json: async () => ({ error: { message: "images path is not a directory" } }),
        } as Response;
      }
      return successfulFetch!(input, init);
    });
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(await screen.findByLabelText("Source"), TEST_SOURCE.source_key);

    await user.click(screen.getByRole("button", { name: "Load source" }));

    expect(await screen.findByText("images path is not a directory")).toBeVisible();
    expect(screen.getByLabelText("Source")).toHaveValue(TEST_SOURCE.source_key);
    expect(screen.getByLabelText("Frame number")).toHaveValue(1);
  });

  it("selects a source and seeks exact one-based frames with persistent source status", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(await screen.findByLabelText("Source"), TEST_SOURCE.source_key);

    expect(screen.getByLabelText("Frame number")).toHaveValue(1);
    expect(screen.getByText(/local test-adapted development material/i)).toBeVisible();
    expect(screen.getByText(/not a held-out benchmark result/i)).toBeVisible();
    expect(screen.getByText(TEST_SOURCE.source_hash)).toBeVisible();
    expect(screen.getByText("Track tools unavailable")).toBeVisible();
    expect(screen.getByText("joco_v1")).toBeVisible();

    const frameInput = screen.getByLabelText("Frame number");
    await user.clear(frameInput);
    await user.type(frameInput, "3");

    await waitFor(() => {
      expect(screen.getByTestId("frame-viewport")).toHaveAttribute(
        "data-frame-url",
        "/api/sequences/mot20-06-joco-v1/frames/3?source_hash=abc123def456",
      );
    });
  });

  it("shows an explicit valid empty state when no sources are available", async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ sources: [], unavailable: [], diagnostics: [] }),
    } as Response);

    render(<App />);

    expect(await screen.findByRole("heading", { name: "No local sources available" })).toBeVisible();
  });

  it("reports source metadata transport errors", async () => {
    vi.mocked(fetch).mockResolvedValue({ ok: false, status: 503 } as Response);

    render(<App />);

    expect(await screen.findByRole("alert")).toHaveTextContent("Source metadata request failed (503)");
  });

  it("rejects zero-based frame input without requesting an ambiguous image", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(await screen.findByLabelText("Source"), TEST_SOURCE.source_key);

    const frameInput = screen.getByLabelText("Frame number");
    await user.clear(frameInput);
    await user.type(frameInput, "0");

    expect(screen.getByRole("alert")).toHaveTextContent("exact one-based frame from 1 through 5");
    expect(screen.queryByTestId("frame-viewport")).not.toBeInTheDocument();
  });

  it("reports an exact JPEG transport failure", async () => {
    const successfulFetch = vi.mocked(fetch).getMockImplementation();
    vi.mocked(fetch).mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.includes("/frames/") && !url.includes("/observations")) {
        return { ok: false, status: 404 } as Response;
      }
      return successfulFetch!(input, init);
    });
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(await screen.findByLabelText("Source"), TEST_SOURCE.source_key);

    expect(await screen.findByText("Exact frame 1 could not be loaded.")).toBeVisible();
  });

  it("steps exact frames by 1 and 10 without crossing sequence bounds", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(await screen.findByLabelText("Source"), TEST_SOURCE.source_key);

    await user.click(screen.getByRole("button", { name: "Next frame" }));
    expect(screen.getByLabelText("Frame number")).toHaveValue(2);
    await user.click(screen.getByRole("button", { name: "Next 10 frames" }));
    expect(screen.getByLabelText("Frame number")).toHaveValue(5);
    await user.click(screen.getByRole("button", { name: "Previous 10 frames" }));
    expect(screen.getByLabelText("Frame number")).toHaveValue(1);
  });

  it("does not advance playback beyond a frame that has not rendered", async () => {
    let resolveFrameTwo!: (response: Response) => void;
    const frameTwo = new Promise<Response>((resolve) => { resolveFrameTwo = resolve; });
    const originalFetch = vi.mocked(fetch).getMockImplementation();
    vi.mocked(fetch).mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.includes("/frames/2?") && !url.includes("/observations")) return frameTwo;
      return originalFetch!(input, init);
    });
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(await screen.findByLabelText("Source"), TEST_SOURCE.source_key);
    await waitFor(() => {
      expect(screen.queryByText("Loading exact frame 1")).not.toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: "Start playback" }));
    await waitFor(() => expect(screen.getByLabelText("Frame number")).toHaveValue(2));
    await act(async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 450));
    });

    expect(screen.getByLabelText("Frame number")).toHaveValue(2);
    resolveFrameTwo({ ok: true, blob: async () => new Blob(["jpeg"]) } as Response);
  });

  it("ignores an aborted stale observation response after a rapid seek", async () => {
    let resolveFirst!: (response: Response) => void;
    const firstResponse = new Promise<Response>((resolve) => { resolveFirst = resolve; });
    vi.mocked(fetch).mockImplementation(async (input) => {
      const url = String(input);
      if (url === "/api/sequences") {
        return { ok: true, json: async () => ({ sources: [TEST_SOURCE], unavailable: [], diagnostics: [] }) } as Response;
      }
      if (url.includes("frames/1/observations")) return firstResponse;
      if (url.includes("frames/2/observations")) {
        return { ok: true, json: async () => ({ source_key: TEST_SOURCE.source_key, sequence: TEST_SOURCE.sequence, frame: 2, frame_numbering: "one_based", source_hash: TEST_SOURCE.source_hash, observations: [] }) } as Response;
      }
      if (url.includes("/frames/")) return { ok: true, blob: async () => new Blob(["jpeg"]) } as Response;
      throw new Error(`Unexpected request: ${url}`);
    });
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(await screen.findByLabelText("Source"), TEST_SOURCE.source_key);
    await user.click(screen.getByRole("button", { name: "Next frame" }));
    expect(await screen.findByText("No observations on frame 2")).toBeVisible();

    await act(async () => {
      resolveFirst({ ok: true, json: async () => ({ source_key: TEST_SOURCE.source_key, sequence: TEST_SOURCE.sequence, frame: 1, frame_numbering: "one_based", source_hash: TEST_SOURCE.source_hash, observations: [{ row_index: 99 }] }) } as Response);
      await firstResponse;
    });

    expect(screen.getByText("No observations on frame 2")).toBeVisible();
  });
});