import { describe, expect, it } from "vitest";

import { createViewportTransform } from "./viewport";

describe("createViewportTransform", () => {
  it("round-trips image and CSS-screen coordinates through letterboxing", () => {
    const transform = createViewportTransform({
      imageWidth: 1920,
      imageHeight: 1080,
      cssWidth: 800,
      cssHeight: 600,
      devicePixelRatio: 2,
    });

    const screenPoint = transform.imageToScreen({ x: 960, y: 540 });
    expect(screenPoint).toEqual({ x: 400, y: 300 });
    expect(transform.screenToImage(screenPoint)).toEqual({ x: 960, y: 540 });
  });

  it("preserves image aspect ratio and exposes the CSS letterbox", () => {
    const transform = createViewportTransform({
      imageWidth: 1600,
      imageHeight: 900,
      cssWidth: 500,
      cssHeight: 500,
      devicePixelRatio: 1,
    });

    expect(transform.imageRectCss).toEqual({
      x: 0,
      y: 109.375,
      width: 500,
      height: 281.25,
    });
    expect(transform.imageRectCss.width / transform.imageRectCss.height).toBeCloseTo(1600 / 900);
  });

  it("scales the backing store and draw rectangle for DPR without changing CSS geometry", () => {
    const transform = createViewportTransform({
      imageWidth: 100,
      imageHeight: 50,
      cssWidth: 301,
      cssHeight: 201,
      devicePixelRatio: 2,
    });

    expect(transform.canvasSize).toEqual({ width: 602, height: 402 });
    expect(transform.imageRectCss).toEqual({ x: 0, y: 25.25, width: 301, height: 150.5 });
    expect(transform.imageRectPixels).toEqual({ x: 0, y: 50.5, width: 602, height: 301 });
  });
});