# Project Brief

Last updated: 2026-08-27

## What This Repository Is

`mot-20` is a new research and engineering repository for the MOT20 Multiple Object Tracking Challenge. The intended pipeline includes detection, ReID representation, association, track lifecycle management, evaluation, and experiment analysis.

## Current State

- The repository contains a self-contained MOT20 track viewer under `track-viz/`, CVAT provisioning tools, the agent harness, and Beads setup.
- The viewer establishes its own Python 3.12 package and React/Vite toolchain; it does not establish the ML framework for other workstreams.
- Local MOT20 datasets are present but remain ignored, protected records.
- Beads is the durable work tracker, using issue prefix `mot`.

## Intended Workstreams

1. Dataset acquisition, validation, and conversion.
2. Detector training or integration.
3. ReID embedding extraction and evaluation.
4. Motion and appearance association.
5. Track birth, update, occlusion, and termination logic.
6. MOT-format export and challenge evaluation.
7. Reproducible experiment tracking and comparison.

## Non-Negotiable Constraints

- Raw datasets, model weights, embeddings, predictions, and run artifacts stay out of Git.
- Do not overwrite experimental artifacts without approval.
- Do not report metrics without reproducible provenance.
- Preserve MOT sequence/frame/identity contracts through all transformations.
- Do not invent build, test, or evaluation commands before the implementation provides them.
