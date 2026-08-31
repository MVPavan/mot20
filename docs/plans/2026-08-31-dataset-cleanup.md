# Dataset Cleanup Plan

## Goal and scope

Apply the cleanup approved in `docs/brainstorms/2026-08-31-dataset-cleanup-requirements.md` without changing unique MOT20 experiment data.

## Paths

- Create: `datasets/README.md`
- Create: `datasets/zip files/`
- Move: top-level `datasets/*.zip` and `datasets/*.pkl` into `datasets/zip files/`
- Remove: `datasets/test/`
- Remove: `datasets/mot20_embeddings/`
- Preserve: `datasets/MOT20/`, `datasets/MOT20_TEST_DET/`, `datasets/val_half/`, and `datasets/test_tracks/`

## Invariants and risks

- Preserve every official sequence, frame number, annotation, detector output, and track result.
- Move source files without overwriting an existing target.
- Delete `datasets/test/` only because every contained artifact was verified to exist byte-for-byte in a retained location.
- The PKL remains opaque and is never deserialized during cleanup.

## Execution

1. Create the README and destination directory.
2. Move the seven explicitly identified ZIP/PKL files.
3. Confirm the retained copies of duplicate data still exist.
4. Remove the approved duplicate and empty directories.
5. Verify the final tree, archive integrity, retained sequence coverage, and repository status.

