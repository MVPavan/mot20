# Byte65 MOT20 CVAT review project

## Goal

Create the isolated CVAT project `byte65` for the two
sequences actually supplied by the corresponding Deepak label bundle:
`MOT20-06` and `MOT20-08`.

## Contract

- Input images remain the read-only `datasets/MOT20/test/<sequence>/img1/`
  images mounted into CVAT as shared files.
- Task ingestion requests CVAT image quality `100`, so the annotation UI uses
  full-resolution, maximum-quality image chunks. CVAT also retains the
  byte-identical original JPEGs for these source files.
- Input labels are normalized YOLO files under
  `datasets/MOT20_TEST_DET_DEEPAK/byte65/labels/`.
- Each input label becomes a task-local, zero-based CVAT rectangle with source
  `semi-auto`. Class `0` is the single person label; optional detector
  confidence is not imported because the project schema has no confidence
  attribute, and neither a score nor a track ID is invented.
- YOLO conversion retains the supplied geometry. It clamps only floating-point
  boundary overflow below `0.01` pixel (present in this bundle) to the image
  edge and rejects any material out-of-bounds box.
- Twelve contiguous range tasks cover every source frame exactly once. Calanit,
  Haim, Tamir, and Deepak receive 100 frames each; every other reviewer
  receives 202 frames. Pavan has one task at each sequence boundary.
- This experiment is not a detector merge, a tracking result, or a quality
  evaluation. MOT20 test has no public ground truth.

## Success criteria

- The project has exactly one `pedestrian` rectangle label.
- There are exactly twelve tasks, each with one job, the expected source
  images, and the configured assignee.
- The imported CVAT shape count exactly equals the source YOLO row count per
  sequence, and source frame `N` maps to CVAT frame `N - 1`.
