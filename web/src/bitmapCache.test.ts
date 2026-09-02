import { describe, expect, it, vi } from "vitest";

import {
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