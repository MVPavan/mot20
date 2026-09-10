#!/usr/bin/env bash
# Does arm E2's much better detector actually track better? val_half, det_thresh 0.50.
#
# Arm E2 has gained ~0.10 mAP@50:95 over the arm E checkpoint that produced the
# tracking results so far. That is a detection gain measured on data the model
# trained on. This probe asks the only question that matters downstream: does it
# convert into HOTA?
#
# READ BEFORE QUOTING ANYTHING FROM THIS RUN.
#
# val_half is the second half of MOT20 train, and arm E2 trained on all of it
# (4,463/4,463 frames). Every number here is TRAINING-SET FIT, as for arm E e19.
#
# THE TWO SIDES ARE NOT CONTAMINATED TO THE SAME DEGREE, AND THAT IS THE WHOLE
# CAVEAT. Arm E e19 saw these frames for 20 epochs; arm E2 e69 has seen them for
# 70. So the delta does NOT isolate "a better detector" -- the weight difference
# it isolates is itself mostly 50 extra epochs of fitting these exact images.
# Read the result as an upper bound with no generalization content. It does not
# license expecting the same gain on MOT20 test, and the absolute HOTA may not
# share a table with any held-out row.
#
# The tell is in the artifact counts, not just the metric: arm E2 e69 emits
# ~14% FEWER surviving detections than e19 and still scores higher on DetA. A
# detector that got genuinely better would normally not need to become that much
# more confident on the very frames it trained on.
#
# CHECKPOINT: epoch 69, not 68.
#
# The request was epoch 68. Its EMA weights no longer exist: BestModelCallback
# overwrites checkpoint_best_ema.pth in place, and epoch 69 replaced it at
# 09:46:58 before it could be copied. checkpoint_interval = 10 means no periodic
# checkpoint covers 68 either (29/39/49/59/69). Epoch 69 is strictly better
# (EMA mAP@50:95 0.8590 vs 0.8582; regular 0.856342 vs 0.854846), so the probe
# uses it rather than the nearest saved but weaker alternative.
#
# The weights are read from a SNAPSHOT taken outside the run directory. Reading
# the live file would race the trainer, which rewrites it on every improvement
# and is still improving every epoch.
#
# det_thresh 0.50 is the sweep optimum for arm E e19 (HOTA 80.78 vs 80.48 at the
# 0.40 default). Whether 0.50 is still optimal for a differently calibrated
# detector is NOT established by this run; it fixes the knob so the detector is
# the only thing that changes.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
container=nvpt-dm
split=val_half
reid=osnet-ain-msdc
base_tracker=btpp-default
det_thresh=0.50
source_variant=rfdetr2xl-arme2-e69-t005
detector=rfdetr2xl-arme2-e69-t010-nms070
checkpoint=finetuning/artifacts/arme2-snapshots/arme2-best-ema-e69.pth
config=finetuning/configs/rfdetr_2xl_i5-arm-e2-continue-100e-6gpu.toml
dataset_root=datasets/finetuning/rfdetr-mot20-crowdhuman-byte65-test-adapted-2026-09-04
# GPU 7 is idle; 0-5 are training arm E2 and must not be touched.
gpu=${PROBE_GPU:-7}

cd "$repo_root"

run() {
    printf '\n### %s\n' "$*"
    docker exec -w "$repo_root" -e CUDA_VISIBLE_DEVICES="$gpu" "$container" sh -lc "$*"
}
have() { [[ -f "artifacts/tracking/$1/$2/$split/manifest.json" ]]; }

for required in "$checkpoint" "$config" "$dataset_root"; do
    [[ -e "$required" ]] || { printf 'Refusing to start: missing %s\n' "$required" >&2; exit 1; }
done
printf 'preflight ok | checkpoint sha256 %s\n' "$(sha256sum "$checkpoint" | cut -d' ' -f1)"

# Step 1: detections at the export floor, captured from the library's own
# validation loop so inference geometry matches training exactly.
if ! have detections "$source_variant"; then
    run ".venv/bin/python tracking/scripts/export_detections.py \
         --variant $source_variant --split $split --config $config \
         --dataset-root $dataset_root --checkpoint $checkpoint --threshold 0.05"
else
    printf 'detections present: %s\n' "$source_variant"
fi

# Step 2: the same score filter and NMS the arm E tracking variant used, so the
# two detectors differ only in weights.
if ! have detections "$detector"; then
    run ".venv-tracking/bin/python tracking/scripts/derive_filtered_variant.py \
         --source $source_variant --target $detector --split $split \
         --min-score 0.1 --nms-iou 0.7"
else
    printf 'filtered detections present: %s\n' "$detector"
fi

# Step 3: embeddings, same ReID as the arm E rows.
if ! have embeddings "${detector}__${reid}"; then
    run ".venv-tracking/bin/python tracking/scripts/export_embeddings.py \
         --detector $detector --reid $reid --split $split"
else
    printf 'embeddings present: %s__%s\n' "$detector" "$reid"
fi

# Step 4: tracking at det_thresh 0.50 plus evaluation. sweep_association.py
# resolves the tracker slug from the point, runs the tracker and evaluates, and
# skips cleanly if the metrics manifest already exists.
run ".venv-tracking/bin/python tracking/scripts/sweep_association.py \
     --detector $detector --reid $reid --base-tracker $base_tracker \
     --slug-prefix dt --split $split --point det_thresh=$det_thresh"

docker exec -w "$repo_root" "$container" sh -lc \
    "chown -R ${HOST_UID:-2006}:${HOST_GID:-2006} '$repo_root/artifacts/tracking'" || true

printf '\n=== arm E2 e69 val_half probe complete (TRAINING-SET FIT, not held out) ===\n'
