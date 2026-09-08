# I5 CrowdHuman/MOT20 mix ablation assembly

## Goal and scope

Build two immutable local-test-adapted RF-DETR COCO ablation roots: Arm B repeats
each MOT20 `train_half` item four times alongside the existing CrowdHuman and
Byte65 composition; Arm C uses MOT20 `train_half` and the same Byte65 overlay
only. Both retain `val_half` unchanged as valid.

Origin: `docs/tracker-improvements.md` I5. This plan does not authorize
training, checkpoint selection, or a competition build.

## Changes

- Add an isolated COCO-manifest repeat/merge helper in
  `finetuning/src/mot20/detection/coco_conversion.py`; existing conversion and
  merge behavior remains unchanged.
- Add a train-manifest audit invariant in
  `finetuning/src/mot20/detection/dataset_audit.py`: duplicate train file names
  are rejected except for an explicit, exact MOT20 repeat-factor declaration.
- Add `finetuning/scripts/assemble_i5_mix_ablation.py`, matching the existing
  Byte65 baseline assembler's immutable-root and provenance pattern.
- Cover repeat IDs/provenance, ratios, and accidental-repeat rejection in the
  mirrored detection tests.

## Risks and invariants

- The 2026-09-04 Arm-A root is read-only input and must not be overwritten.
- Output image and annotation IDs are sequential and unique; source-manifest
  IDs remain in provenance fields for every repeated copy.
- Arm B permits only four copies of each declared MOT20 source identity, with
  repeat indices 1 through 4. This cannot accept an extra/missing/substituted
  copy or a repeat from any other source.
- Each root writes canonical manifests, source manifests, checksums, and audit
  evidence; `classification` remains `local_test_adapted` and held-out
  comparability remains false.

## Verification

1. Test-first: add and run a focused repeat/ratio test, observing its failure
   before adding the helper.
2. Run the touched detection test modules and `make -C finetuning compile`.
3. Assemble both new roots and inspect fresh audit/count evidence plus Git
   status. RF-DETR loader compatibility is established from the installed
   dataset implementation with a focused duplicate-file-name fixture if the
   environment provides it.
