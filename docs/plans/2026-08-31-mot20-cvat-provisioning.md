# MOT20 CVAT provisioning plan

## Goal and scope

Create a repo-local CVAT workspace that can run an isolated CVAT deployment,
validate the local MOT20 test image contract, generate deterministic task
assignments, create/reuse tasks safely, assign each resulting job to a reviewer,
and later import MOT-format annotations. The source images under `datasets/`
remain read-only.

This plan is based on the local CVAT deployment at
`/data/pavan/tycoai/project_helpers/data_miner/manual_reviewer_cvat` and its
idempotent REST task importer, not on its incomplete stub scripts. The prior
port-8081 stack was stopped with its volumes preserved; the MOT20 stack uses its
own Compose project and port 8082.

## Decisions and assumptions

- The four test sequences are `MOT20-04`, `MOT20-06`, `MOT20-07`, and
  `MOT20-08`; their `seqinfo.ini` frame counts match the image files.
- MOT20 has a single object category, so the initial project schema contains
  one rectangle label: `pedestrian`. Extend the committed schema before the
  first live project creation if the supplied annotations need more labels or
  attributes.
- One planned task is a contiguous inclusive frame range from one sequence.
  Its CVAT segment size equals its image count, which creates exactly one job
  and makes a job assignment unambiguous.
- Reviewer identities and passwords are not yet known. They belong in ignored
  local configuration and environment variables; no task is created until the
  reviewer plan has concrete usernames.
- Existing project/task names are never overwritten. A matching existing task
  is reused only after its project, frame names, size, and job assignee match;
  any mismatch is a hard error.

## Implementation

1. Add `cvat/` with a CVAT 2.18.0 Compose deployment, a local env template,
   and lifecycle commands. The mounted test root is read-only and the Compose
   project name/port/volumes do not overlap the existing review stack.
2. Add pure Python contract helpers and tests first. They prove `seqinfo.ini`
   consistency, six-digit frame name preservation, deterministic balanced task
   planning, full frame coverage, and MOT annotation conversion with task-local
   zero-based frames.
3. Add scripts to generate the ignored reviewer assignment plan, bootstrap
   reviewers only when explicitly invoked, seed/reuse tasks through CVAT REST,
   and import supplied MOT text annotations only after exact task validation.
4. Document the operator sequence: copy local env/config templates, start the
   stack, create an admin, add reviewer passwords, generate/review the plan,
   dry-run, seed, verify, and import annotations.

## Tracer bullet and verification

The tracer bullet is an offline dry-run that reads all 4,479 actual test images
and creates a deterministic assignment plan without contacting CVAT or mutating
the dataset. Fresh verification will run unit tests, the dry-run, shell syntax
checks, Docker Compose rendering, and the isolated stack health check when the
required local admin configuration is supplied. Live project creation is blocked
only on the deliberately absent credentials and reviewer list.

## Risks and invariants

- CVAT stores task frames as zero-based indexes; the importer maps each MOT
  one-based frame through the task range and refuses frames outside it.
- Images are referenced via the read-only shared-folder mount so the original
  pixels are neither copied nor altered during task creation.
- Annotation imports are explicit, after task validation, and never run as part
  of task seeding.
- CVAT database/media volumes are persistent. `stop` does not remove volumes;
  deleting them requires an explicit manual operation outside the automation.
