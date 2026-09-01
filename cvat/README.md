# MOT20 CVAT workspace

This directory is the isolated, repeatable CVAT workflow for manual review of
the four local MOT20 test sequences. It uses the separate `mot20_cvat` Compose
project on port 8082. The prior port-8081 CVAT stack was stopped with its
volumes preserved.

The MOT20 image root is mounted read-only at
`/home/django/share/mot20-test`. Tasks reference those source files through
CVAT's shared-folder ingestion, so image names remain `000001.jpg` through the
official sequence length and pixels are not duplicated or changed.

All task upload requests set CVAT's `image_quality` to `100`. This preserves
the source resolution in the annotation view at CVAT's maximum JPEG quality;
the separate CVAT original-image chunks retain the source JPEG bytes for the
MOT20 images.

## One-time local setup

Install the small REST dependency in the Python environment you will use:

```bash
python3 -m pip install -r cvat/requirements.txt
cp cvat/config/stack.env.example cvat/.env
chmod 600 cvat/.env
```

Edit `cvat/.env`: choose an admin password and, if this repository is checked
out elsewhere, set `CVAT_SHARE_PATH` to that checkout's `datasets/MOT20/test`
directory. Keep it local; it is ignored.

Start CVAT and create the configured administrator:

```bash
bash cvat/scripts/manage_cvat.sh start
bash cvat/scripts/manage_cvat.sh create-admin
```

Open `http://127.0.0.1:8082`. `stop` preserves all volumes. The scripts never
run `docker compose down --volumes`.

The local health threshold is explicitly set to 98% in `.env`, rather than
CVAT's default 90%, because this host currently keeps less than 10% free space
for other research artifacts. It is not a storage fix: keep at least 2% free
and free capacity before uploads or exports grow substantially.

## Reviewer and task provisioning

After the reviewer names are decided, create `cvat/config/reviewers.json` from
`reviewers.example.json`. Give each reviewer a distinct password environment
variable and export those variables locally. Create users once:

```bash
python3 cvat/scripts/bootstrap_users.py --reviewers cvat/config/reviewers.json
```

Generate a reviewable, ignored plan. It assigns contiguous frame ranges by
least current frame load; the default 500-frame cap produces one CVAT job per
task and balances the 4,479 total frames across the named reviewers.

```bash
python3 cvat/scripts/plan_tasks.py \
  --reviewers alice,bob,carol \
  --output cvat/config/assignments.json
```

Review `assignments.json` before proceeding. It must cover every source frame
exactly once. Then run the non-mutating preflight and the idempotent live seed:

```bash
python3 cvat/scripts/seed_tasks.py --dry-run
python3 cvat/scripts/seed_tasks.py
```

The live run creates `MOT20 Test Manual Annotation` from
`config/project.json`, finds each configured CVAT user, uploads each range by
read-only shared paths, waits for ingestion, and assigns its only job. A rerun
reuses a task only if its project, frame names, size, and assignee still match;
otherwise it stops without replacing anything.

## Supplied MOT annotations

The committed label schema is deliberately minimal: one `pedestrian` rectangle
label, matching MOT20. Change `config/project.json` before creating the live
project if your annotation format needs a different schema.

When annotations arrive, place either `MOT20-04/gt/gt.txt` style files or
`MOT20-04.txt` style files under one local annotations root. The importer reads
MOT rows (`frame,id,left,top,width,height,...`), retains track IDs within each
task, converts one-based MOT frames to CVAT task-local zero-based frames, and
refuses mismatched image identities.

```bash
python3 cvat/scripts/import_mot_annotations.py \
  --annotations-root /path/to/supplied-annotations --dry-run
python3 cvat/scripts/import_mot_annotations.py \
  --annotations-root /path/to/supplied-annotations
```

Imports refuse to replace non-empty CVAT annotations. Before an intentional
replacement, run `bash cvat/scripts/manage_cvat.sh backup`, inspect the backup,
and then add `--replace-existing` explicitly.

## Checks

```bash
python3 -m unittest discover -s cvat/tests -v
python3 cvat/scripts/plan_tasks.py --reviewers smoke_a,smoke_b --output /tmp/mot20-cvat-assignments.json
python3 cvat/scripts/seed_tasks.py --plan /tmp/mot20-cvat-assignments.json --reviewers cvat/config/reviewers.example.json --dry-run
docker compose --env-file cvat/config/stack.env.example -f cvat/docker-compose.yml config
```

The final live check is `bash cvat/scripts/manage_cvat.sh status`, then the
first real `seed_tasks.py` output: it lists project and task IDs plus assignees.
