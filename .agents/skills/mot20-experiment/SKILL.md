---
name: mot20-experiment
description: Plan, run, validate, or report a MOT20 detection, ReID, association, or tracking experiment with reproducible provenance and protected artifacts.
---

# MOT20 Experiment

Use this workflow for experiment design, execution, comparison, or reporting. Do not start expensive or long-running work unless the user explicitly authorized execution.

## Before Execution

- Define the question, baseline, changed variable, success metric, and non-goals.
- Record the code revision, dataset sequences/split, detector/checkpoint, ReID model/checkpoint, tracker and association configuration, seed, device, and environment.
- Identify output paths and confirm they do not overwrite prior artifacts.
- Validate configuration and input paths with the cheapest available preflight.

## Verification

- Check exported result structure before metric evaluation.
- Record the exact command, exit status, output artifact, and evaluator version/configuration.
- Compare against the intended baseline using the same dataset and evaluation settings.
- Distinguish measured results from hypotheses and interpretation.

## Tracking

- Beads tracks the experiment work item, dependencies, and blockers.
- Durable run provenance and metrics belong in an experiment registry or report under `docs/` once that structure exists.
- Large outputs, predictions, embeddings, weights, and datasets remain git-ignored.
- Never fabricate missing metrics or treat a partial run as complete.
