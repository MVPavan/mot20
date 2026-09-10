# Byte65 NMS 68-Sequence Dataset Audit

## Scope

This record covers the local, test-adapted dataset named `byte65nms_68seq`.
It was materialized from the Byte65 post-modification YOLO archive and is not
part of the clean MOT20/CrowdHuman training baseline.

## Source And Output

| Item | Value |
| --- | --- |
| Source archive | `datasets/zip-files/byte65-modified-images-yolo.zip` |
| Source SHA-256 | `628aacc1fad07a5bd4bffb5a0f9ea2872d2530c8b13a343d2bda2c634ff7a3b3` |
| Label authority | `post_modification_annotations/` only |
| Generated dataset | `datasets/MOT20_TEST_DET_DEEPAK/byte65nms_68seq/` |
| Classification | local, test-adapted overlay source; not leaderboard-comparable |
| Test sequences | `MOT20-06`, `MOT20-08` |

The output contains portable YOLO image/label trees, `data.yaml`, `lists/`,
`annotations.coco.json`, and `audit.json`. It is ignored local data and was
created only after confirming the destination did not exist.

## Audit Results

| Check | Result |
| --- | --- |
| Archive images | 21 |
| Post-modification YOLO label files | 21 |
| CVAT post-modification records | 21 |
| Image/label name mismatches | 0 |
| CVAT/YOLO per-image annotation-count mismatches | 0 |
| Annotations | 2,588, all YOLO class `0` / pedestrian |
| Generated YOLO label files | 21 |
| Generated COCO annotations | 2,588, all category `1` / pedestrian |
| Invalid output boxes | 0 |
| Clipped source boxes | 7 |

The seven clipped boxes crossed an image boundary by floating-point rounding or
source-coordinate extent. The generated YOLO and COCO boxes were clipped to
native image bounds, with every adjustment recorded in `audit.json`. No source
row was silently dropped.

The coverage checks establish that the archive's exported labels are complete
with respect to its corresponding CVAT post-modification records. On 2026-09-04,
the user confirmed that the 21 images received an exhaustive human semantic
audit. This is a user-confirmed audit assertion, not an independent review
performed by this repository automation.

The labels are approved for a separately named `local_test_adapted` RF-DETR
baseline containing test-derived frames from `MOT20-06` and `MOT20-08`. The
clean MOT20/CrowdHuman baseline remains separate. No results using this source
are held-out or leaderboard-comparable.

## Assembled Baseline

The immutable RF-DETR root
`datasets/finetuning/rfdetr-mot20-crowdhuman-byte65-test-adapted-2026-09-04`
was assembled with a `train/byte65` linked source root. Its audit records the
user-confirmed audit authority, `local_test_adapted` classification, 23,859
train images, 1,154,031 train annotations, no cross-split duplicate bytes, and
no MOT20 temporal overlap. The MOT20 `val_half` split remains the unchanged
4,463-image, 676,519-annotation detector-selection set.

## Validation

The post-materialization structural check confirmed 21 images, 21 labels,
2,588 annotations in both representations, valid category mappings, and
in-bounds finite COCO and normalized YOLO geometry.

See `docs/MOTPolicy.md` for mandatory reporting language for any run that uses
this test-derived data.