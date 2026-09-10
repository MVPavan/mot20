#!/usr/bin/env bash
# Produce MOT20 *test* tracker files for an RF-DETR arm with BoostTrack++
# postprocessing, for both ReID models.
#
# Defaults reproduce the arm E e19 configuration that was actually measured on
# val_half: arm E + osnet-ain-msdc + btpp-default + DTI + GBI scored HOTA 81.031
# (raw 80.476). Override the environment variables below to run another arm.
#
# NOTHING HERE CAN BE SCORED LOCALLY. MOT20 test ships no ground truth. The only
# feedback is a MOTChallenge submission. Every number this script prints is a
# count or a checksum, never a metric, and no step in it evaluates anything.
# That is also why the tracker is driven directly rather than through
# sweep_association.py, which would try to run TrackEval on a split with no GT.
#
# WHY TWO ReID MODELS, AND WHY THAT IS NOT A CONTRADICTION
#
# osnet-ain-msdc has never seen MOT20, which is why the experiment report builds
# its headline comparison on it. fastreid-sbs-s50-mot20 is MOT20-trained, so on
# val_half it has already seen those identities and its measured +0.70 HOTA is
# inflated. On TEST that objection does not apply: MOT20 test identities are not
# in MOT20 train, so FastReID there is a domain-matched model rather than a
# leaking one. It is still an extrapolation -- FastReID has never been scored
# with either arm -- and it cannot be checked without submitting.
#
# ON THE ARM E2 CONFIGURATION, IF YOU RUN IT
#
# Arm E2 e69 scores HOTA 87.78 on val_half against arm E e19's 80.78 at the same
# det_thresh. Do NOT expect that margin here. val_half is inside arm E2's
# training data and it has fitted those exact frames for 70 epochs against arm
# E's 20, so most of that gap is memorisation of the evaluation images. MOT20
# test is genuinely unseen (bar 21 overlapping images out of 4,479), so the
# honest prior is that the two arms land much closer together than +7 HOTA.
#
# The score filter is 0.10 AND greedy NMS at IoU 0.70. Both matter: the score
# filter alone leaves 71,364 extra rows on train, which NMS removes. RF-DETR is
# set prediction and exports without NMS by design; on MOT20 density that
# produced ~33 duplicate pairs per frame, so NMS is not optional here.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
container=nvpt-dm
python=.venv-tracking/bin/python
split=test

# --- configuration, overridable from the environment -------------------------
source_variant=${SOURCE_VARIANT:-rfdetr2xl-arme-e19-t005}
detector=${DETECTOR:-rfdetr2xl-arme-e19-t010-nms070}
# Only needed when the source detections do not exist yet.
checkpoint=${CHECKPOINT:-}
config=${CONFIG:-}
dataset_root=${DATASET_ROOT:-datasets/finetuning/rfdetr-mot20-test-2026-09-08}
# Tracker slug plus the settings that produce it. Leave TRACKER at btpp-default
# for library defaults; for a swept point pass the slug sweep_association.py
# resolves for that point, together with the matching --set override.
tracker=${TRACKER:-btpp-default}
tracker_overrides=${TRACKER_OVERRIDES:-}
submission_root=${SUBMISSION_ROOT:-artifacts/submissions/mot20-test-arme-e19}
reids=(${REIDS:-osnet-ain-msdc fastreid-sbs-s50-mot20})
gpu=${TEST_GPU:-6}

cd "$repo_root"

run() {
    printf '\n### %s\n' "$*"
    docker exec -w "$repo_root" -e CUDA_VISIBLE_DEVICES="$gpu" "$container" sh -lc "$*"
}
have() { [[ -f "artifacts/tracking/$1/$2/$split/manifest.json" ]]; }

# A level commits by writing its manifest last, so files without a manifest mean
# an earlier attempt died part-way. Say what to remove rather than letting the
# retry fail at the first existing file.
refuse_partial() {
    local dir="artifacts/tracking/$1/$2/$split" pattern=$3
    if [[ ! -f "$dir/manifest.json" ]] && compgen -G "$dir/$pattern" >/dev/null; then
        printf 'Refusing to start: %s holds output but no manifest, so an earlier\n' "$dir" >&2
        printf 'run failed part-way. Inspect and remove it, then re-run.\n' >&2
        exit 1
    fi
}

# Step 0: detections at the export floor. Captured from the library's own
# validation loop so inference geometry matches training exactly.
if ! have detections "$source_variant"; then
    if [[ -z "$checkpoint" || -z "$config" ]]; then
        printf 'Refusing to start: no %s detections for %s, and no CHECKPOINT/CONFIG\n' \
            "$split" "$source_variant" >&2
        printf 'given to produce them.\n' >&2
        exit 1
    fi
    [[ -e "$checkpoint" ]] || { printf 'Refusing to start: missing %s\n' "$checkpoint" >&2; exit 1; }
    printf 'checkpoint sha256 %s\n' "$(sha256sum "$checkpoint" | cut -d' ' -f1)"
    run ".venv/bin/python tracking/scripts/export_detections.py \
         --variant $source_variant --split $split --config $config \
         --dataset-root $dataset_root --checkpoint $checkpoint --threshold 0.05"
else
    printf 'source detections present: %s\n' "$source_variant"
fi

# Step 1: score filter + NMS. Detector-side, so it is shared by both ReID arms.
refuse_partial detections "$detector" '*/det.txt'
if ! have detections "$detector"; then
    run "$python tracking/scripts/derive_filtered_variant.py \
         --source $source_variant --target $detector --split $split \
         --min-score 0.1 --nms-iou 0.7"
else
    printf 'filtered detections present: %s\n' "$detector"
fi

for reid in "${reids[@]}"; do
    combination="${detector}__${reid}__${tracker}"
    printf '\n=== ReID arm: %s ===\n' "$reid"

    # Step 2: embeddings. `--reid` only names the output directory; the model is
    # selected by `--test-dataset`, so the two must be set together or the run
    # stores one model's vectors under the other's slug. export_embeddings.py
    # refuses when they disagree, which is what caught this the first time.
    reid_model_flag=""
    [[ "$reid" == fastreid-* ]] && reid_model_flag="--test-dataset"
    refuse_partial embeddings "${detector}__${reid}" '*.npz'
    if ! have embeddings "${detector}__${reid}"; then
        run "$python tracking/scripts/export_embeddings.py \
             --detector $detector --reid $reid --split $split $reid_model_flag"
    else
        printf 'embeddings present: %s__%s\n' "$detector" "$reid"
    fi

    # Step 3: tracks. ECC warps cache under <sequence>-test, separate from the
    # val_half cache, because test frame indices name different images.
    refuse_partial tracks "$combination" '*.txt'
    if ! have tracks "$combination"; then
        run "$python tracking/scripts/run_boosttrack.py \
             --detector $detector --reid $reid --tracker $tracker \
             --split $split $tracker_overrides"
    else
        printf 'tracks present: %s\n' "$combination"
    fi

    # Step 4: the two postprocessing rungs, scored separately on val_half and
    # therefore produced separately here. dti-gbi is the exported configuration.
    for rung in dti dti-gbi; do
        target="${detector}__${reid}__${tracker}-${rung}"
        refuse_partial tracks "$target" '*.txt'
        verify=""
        [[ "$rung" == "dti" ]] && verify="--verify-against-upstream"
        if ! have tracks "$target"; then
            run "$python tracking/scripts/postprocess_tracks.py \
                 --combination $combination --rung $rung --split $split $verify"
        else
            printf 'rung tracks present: %s\n' "$target"
        fi
    done

    # Step 5: MOTChallenge 10-column submission files for the final rung.
    destination="$submission_root/$reid"
    if [[ -d "$destination" ]]; then
        printf 'submission directory already exists, leaving it alone: %s\n' "$destination"
    else
        run "$python tracking/scripts/export_mot20_format.py --kind tracks --split $split \
             --detector $detector --reid $reid --tracker ${tracker}-dti-gbi \
             --destination $destination"
    fi
done

docker exec -w "$repo_root" "$container" sh -lc \
    "chown -R ${HOST_UID:-2006}:${HOST_GID:-2006} '$repo_root/artifacts'" || true

printf '\n=== test submission files built under %s ===\n' "$submission_root"
printf 'These are UNSCORED. MOT20 test has no public ground truth.\n'
