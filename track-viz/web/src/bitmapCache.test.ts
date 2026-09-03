import { describe, expect, it, vi } from "vitest";

import {
  BitmapFrameLoader,
  BitmapLru,
  DEFAULT_BITMAP_CACHE_SIZE,
  decodeFrame,
} from "./bitmapCache";

interface FakeBitmap {
  id: number;
  close(): void;
}

function fakeBitmap(id: number): FakeBitmap {
  return { id, close: vi.fn() };
}

describe("BitmapLru", () => {
  it("defaults to 150 and accepts configuration only from 100 through 200", () => {
    expect(new BitmapLru().capacity).toBe(DEFAULT_BITMAP_CACHE_SIZE);
    expect(() => new BitmapLru(99)).toThrow(RangeError);
    expect(() => new BitmapLru(201)).toThrow(RangeError);
    expect(new BitmapLru(100).capacity).toBe(100);
    expect(new BitmapLru(200).capacity).toBe(200);
  });

  it("closes the least-recently-used bitmap on eviction and every bitmap on clear", () => {
    const cache = new BitmapLru<FakeBitmap>(100);
    const bitmaps = Array.from({ length: 101 }, (_, index) => fakeBitmap(index));
    bitmaps.slice(0, 100).forEach((bitmap) => cache.set(bitmap.id, bitmap));
    cache.get(0);
    cache.set(100, bitmaps[100]);

    expect(bitmaps[1].close).toHaveBeenCalledOnce();
    expect(bitmaps[0].close).not.toHaveBeenCalled();
    expect(cache.closedCount).toBe(1);

    cache.clear();
    expect(bitmaps[0].close).toHaveBeenCalledOnce();
    expect(bitmaps[100].close).toHaveBeenCalledOnce();
    expect(cache.size).toBe(0);
    expect(cache.closedCount).toBe(101);
  });
});

describe("decodeFrame", () => {
  it("closes a decoded bitmap when the request becomes stale", async () => {
    const bitmap = fakeBitmap(1);
    const controller = new AbortController();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, blob: async () => new Blob() }));
    vi.stubGlobal(
      "createImageBitmap",
      vi.fn().mockImplementation(async () => {
        controller.abort();
        return bitmap;
      }),
    );

    await expect(decodeFrame("/frame.jpg", controller.signal)).rejects.toMatchObject({
      name: "AbortError",
    });
    expect(bitmap.close).toHaveBeenCalledOnce();
  });
});

describe("BitmapFrameLoader", () => {
  it("shares an in-flight frame decode between prefetch and current-frame loading", async () => {
    let resolveResponse!: (response: Response) => void;
    const response = new Promise<Response>((resolve) => { resolveResponse = resolve; });
    const bitmap = { width: 8, height: 6, close: vi.fn() } as unknown as ImageBitmap;
    vi.stubGlobal("fetch", vi.fn().mockReturnValue(response));
    vi.stubGlobal("createImageBitmap", vi.fn().mockResolvedValue(bitmap));
    const cache = new BitmapLru();
    const loader = new BitmapFrameLoader(cache);

    const prefetched = loader.load(7, "/frame-7.jpg");
    const current = loader.load(7, "/frame-7.jpg");
    expect(fetch).toHaveBeenCalledOnce();
    expect(current).toBe(prefetched);
    resolveResponse({ ok: true, blob: async () => new Blob(["jpeg"]) } as Response);

    await expect(current).resolves.toBe(bitmap);
    expect(cache.get(7)).toBe(bitmap);
  });
});