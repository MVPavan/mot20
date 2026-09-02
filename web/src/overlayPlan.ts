import type { HitRegion } from "./candidateRanking";
import type { SelectionState } from "./selectionState";
import type { Point } from "./viewport";

export interface BoxCommand {
  type: "box";
  rowIndex: number;
  region: HitRegion;
  emphasis: "audit" | "candidate" | "active" | "confirmed";
  number?: number;
}

export interface MagnifierCommand {
  type: "magnifier";
  rowIndex: number;
  pointerImage: Point;
  region: HitRegion;
}

export type OverlayCommand = BoxCommand | MagnifierCommand;

export interface OverlayPlan {
  commands: OverlayCommand[];
  strokeCount: number;
}

interface OverlayPlanInput {
  selection: SelectionState;
  regions: readonly HitRegion[];
  pointerImage: Point | null;
  revealAll: boolean;
}

export function buildOverlayPlan(input: OverlayPlanInput): OverlayPlan {
  const byRowIndex = new Map(input.regions.map((region) => [region.rowIndex, region]));
  const commands: OverlayCommand[] = [];

  if (input.selection.mode === "pinned") {
    input.selection.rowIndexes.forEach((rowIndex, index) => {
      const region = byRowIndex.get(rowIndex);
      if (region !== undefined) {
        commands.push({
          type: "box",
          rowIndex,
          region,
          emphasis: rowIndex === input.selection.activeRowIndex ? "active" : "candidate",
          number: index + 1,
        });
      }
    });
  } else if (input.selection.mode === "confirmed") {
    const region = byRowIndex.get(input.selection.activeRowIndex);
    if (region !== undefined) {
      commands.push({
        type: "box",
        rowIndex: input.selection.activeRowIndex,
        region,
        emphasis: "confirmed",
      });
    }
  } else {
    if (input.revealAll) {
      input.regions.forEach((region) => {
        commands.push({
          type: "box",
          rowIndex: region.rowIndex,
          region,
          emphasis: region.rowIndex === input.selection.activeRowIndex ? "active" : "audit",
        });
      });
    } else if (input.selection.activeRowIndex !== null) {
      const active = byRowIndex.get(input.selection.activeRowIndex);
      if (active !== undefined) {
        commands.push({
          type: "box",
          rowIndex: active.rowIndex,
          region: active,
          emphasis: "active",
        });
      }
    }

    if (input.selection.activeRowIndex !== null && input.pointerImage !== null) {
      const active = byRowIndex.get(input.selection.activeRowIndex);
      if (active !== undefined) {
        commands.push({
          type: "magnifier",
          rowIndex: active.rowIndex,
          pointerImage: input.pointerImage,
          region: active,
        });
      }
    }
  }

  return { commands, strokeCount: commands.length };
}