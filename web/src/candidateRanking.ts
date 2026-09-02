import type { Point } from "./viewport";

export interface HitRegion {
  rowIndex: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface RankedCandidate {
  region: HitRegion;
  contains: boolean;
  area: number;
  normalizedPointerGeometry: number;
  edgeDistance: number;
}

function metrics(region: HitRegion, pointer: Point): RankedCandidate {
  const width = region.x2 - region.x1;
  const height = region.y2 - region.y1;
  const normalizedX = (pointer.x - region.x1) / width;
  const normalizedY = (pointer.y - region.y1) / height;

  return {
    region,
    contains:
      pointer.x >= region.x1 &&
      pointer.x <= region.x2 &&
      pointer.y >= region.y1 &&
      pointer.y <= region.y2,
    area: width * height,
    normalizedPointerGeometry: Math.hypot(normalizedX - 0.5, normalizedY - 0.5),
    edgeDistance: Math.min(
      Math.abs(pointer.x - region.x1),
      Math.abs(pointer.x - region.x2),
      Math.abs(pointer.y - region.y1),
      Math.abs(pointer.y - region.y2),
    ),
  };
}

export function rankCandidates(regions: readonly HitRegion[], pointer: Point): RankedCandidate[] {
  return regions.map((region) => metrics(region, pointer)).sort((left, right) => {
    return (
      Number(right.contains) - Number(left.contains) ||
      left.area - right.area ||
      left.normalizedPointerGeometry - right.normalizedPointerGeometry ||
      left.edgeDistance - right.edgeDistance ||
      left.region.rowIndex - right.region.rowIndex
    );
  });
}

export function containingCandidates(
  regions: readonly HitRegion[],
  pointer: Point,
): RankedCandidate[] {
  return rankCandidates(regions, pointer).filter((candidate) => candidate.contains);
}