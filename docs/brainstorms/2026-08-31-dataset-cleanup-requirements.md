# Dataset Cleanup Requirements

Date: 2026-08-31

## Goal

Keep the local `datasets/` directory understandable and remove the large redundant test extraction while preserving official MOT20 data, unique detections, tracks, embeddings, and their source archives.

## Approved changes

- Document the retained dataset artifacts in `datasets/README.md`.
- Move all top-level ZIP and PKL source files into `datasets/zip files/`.
- Remove `datasets/test/`, after verifying that its images and public detections are identical to `datasets/MOT20/test/` and its YOLOX-X MOT20 detections are identical to those in `datasets/MOT20_TEST_DET/`.
- Remove the empty `datasets/mot20_embeddings/` directory.
- Keep `datasets/MOT20/`, `datasets/MOT20_TEST_DET/`, `datasets/val_half/`, and `datasets/test_tracks/` unchanged.

The retained `MOT20_TEST_DET/` layout intentionally repeats small `seqinfo.ini` and public-detection files from `MOT20/test/` so that the delivered detector bundle remains structurally intact.

## Non-goals

- Do not rename sequences, detector variants, or result files.
- Do not deserialize or modify the PKL artifact.
- Do not alter images, annotations, detections, tracks, or archives.
- Do not remove source archives.
