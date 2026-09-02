export const MIN_BITMAP_CACHE_SIZE = 100;
export const MAX_BITMAP_CACHE_SIZE = 200;
export const DEFAULT_BITMAP_CACHE_SIZE = 150;

export interface CloseableDrawable {
  close?: () => void;
}

export type DecodedFrame = (ImageBitmap | HTMLImageElement) & CloseableDrawable;

export class BitmapLru<T extends CloseableDrawable = DecodedFrame> {
  readonly capacity: number;
  readonly #entries = new Map<number, T>();
  #closedCount = 0;

  constructor(capacity = DEFAULT_BITMAP_CACHE_SIZE) {
    if (
      !Number.isInteger(capacity) ||
      capacity < MIN_BITMAP_CACHE_SIZE ||
      capacity > MAX_BITMAP_CACHE_SIZE
    ) {
      throw new RangeError(
        `bitmap cache capacity must be an integer from ${MIN_BITMAP_CACHE_SIZE} through ${MAX_BITMAP_CACHE_SIZE}`,
      );
    }
    this.capacity = capacity;
  }

  get size(): number {
    return this.#entries.size;
  }

  get closedCount(): number {
    return this.#closedCount;
  }

  #close(bitmap: T): void {
    bitmap.close?.();
    this.#closedCount += 1;
  }

  get(frame: number): T | undefined {
    const bitmap = this.#entries.get(frame);
    if (bitmap !== undefined) {
      this.#entries.delete(frame);
      this.#entries.set(frame, bitmap);
    }
    return bitmap;
  }

  set(frame: number, bitmap: T): void {
    const replaced = this.#entries.get(frame);
    if (replaced !== undefined && replaced !== bitmap) {
      this.#close(replaced);
    }
    this.#entries.delete(frame);
    this.#entries.set(frame, bitmap);
    while (this.#entries.size > this.capacity) {
      const oldestFrame = this.#entries.keys().next().value as number | undefined;
      if (oldestFrame === undefined) {
        break;
      }
      const oldest = this.#entries.get(oldestFrame);
      if (oldest !== undefined) {
        this.#close(oldest);
      }
      this.#entries.delete(oldestFrame);
    }
  }

  clear(): void {
    this.#entries.forEach((bitmap) => this.#close(bitmap));
    this.#entries.clear();
  }
}

function abortError(): DOMException {
  return new DOMException("Frame request was superseded", "AbortError");
}

async function decodeHtmlImage(blob: Blob, signal: AbortSignal): Promise<HTMLImageElement> {
  const objectUrl = URL.createObjectURL(blob);
  try {
    return await new Promise<HTMLImageElement>((resolve, reject) => {
      const image = new Image();
      const abort = () => reject(abortError());
      signal.addEventListener("abort", abort, { once: true });
      image.onload = () => {
        signal.removeEventListener("abort", abort);
        resolve(image);
      };
      image.onerror = () => {
        signal.removeEventListener("abort", abort);
        reject(new Error("Frame image could not be decoded"));
      };
      image.src = objectUrl;
    });
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

export async function decodeFrame(url: string, signal: AbortSignal): Promise<DecodedFrame> {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new Error(`Exact frame request failed (${response.status})`);
  }
  const blob = await response.blob();
  if (signal.aborted) {
    throw abortError();
  }
  if (typeof createImageBitmap === "function") {
    const bitmap = await createImageBitmap(blob);
    if (signal.aborted) {
      bitmap.close();
      throw abortError();
    }
    return bitmap;
  }
  return decodeHtmlImage(blob, signal);
}