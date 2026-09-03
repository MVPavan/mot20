export interface Point {
  x: number;
  y: number;
}

export interface Rectangle {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ViewportInput {
  imageWidth: number;
  imageHeight: number;
  cssWidth: number;
  cssHeight: number;
  devicePixelRatio: number;
}

export interface ViewportTransform {
  canvasSize: { width: number; height: number };
  imageRectCss: Rectangle;
  imageRectPixels: Rectangle;
  imageToScreen(point: Point): Point;
  screenToImage(point: Point): Point;
}

export function createViewportTransform(input: ViewportInput): ViewportTransform {
  const values = Object.values(input);
  if (values.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new RangeError("viewport dimensions and devicePixelRatio must be positive and finite");
  }

  const scale = Math.min(input.cssWidth / input.imageWidth, input.cssHeight / input.imageHeight);
  const imageRectCss = {
    x: (input.cssWidth - input.imageWidth * scale) / 2,
    y: (input.cssHeight - input.imageHeight * scale) / 2,
    width: input.imageWidth * scale,
    height: input.imageHeight * scale,
  };
  const canvasSize = {
    width: Math.round(input.cssWidth * input.devicePixelRatio),
    height: Math.round(input.cssHeight * input.devicePixelRatio),
  };
  const imageRectPixels = {
    x: imageRectCss.x * input.devicePixelRatio,
    y: imageRectCss.y * input.devicePixelRatio,
    width: imageRectCss.width * input.devicePixelRatio,
    height: imageRectCss.height * input.devicePixelRatio,
  };

  return {
    canvasSize,
    imageRectCss,
    imageRectPixels,
    imageToScreen: ({ x, y }) => ({
      x: imageRectCss.x + x * scale,
      y: imageRectCss.y + y * scale,
    }),
    screenToImage: ({ x, y }) => ({
      x: (x - imageRectCss.x) / scale,
      y: (y - imageRectCss.y) / scale,
    }),
  };
}