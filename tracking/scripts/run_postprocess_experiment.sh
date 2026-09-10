#!/usr/bin/env bash
# Experiment mot-n2n.6: how much does BoostTrack++'s postprocessing buy us?
#
# Upstream scores its `_post_gbi` folder; we have only ever scored raw tracks.
# This produces and evaluates three rungs per detector so the two passes are not
# conflated:
#
#   btpp-default            raw, already measured for yoloxx20-official
#   btpp-default-dti        + linear gap fill        (adds boxes, moves none)
#   btpp-default-dti-gbi    + gradient-boosting smooth (REPLACES every box)
#
# CONTAMINATION, READ BEFORE QUOTING ANY NUMBER FROM THIS RUN. Both detectors
# trained on the full MOT20 train split, which contains every val_half frame:
# arm E verified at 4463/4463 overlap, and the official ByteTrack MOT20 YOLOX-X
# likewise. Every figure here is TRAINING-SET FIT. The pair is internally
# matched -- both sides are contaminated identically -- so the postprocessing
# DELTA is meaningful, but no absolute value may share a table with a held-out
# val_half figure such as arm D's.
#
# Nothing here runs a detector, a ReID model, or a tracker over images except
# the one arm E baseline, which is needed because arm E has no val_half tracks
# yet. The rungs themselves are pure text transforms and cost no GPU.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
container=nvpt-dm
python=.venv-tracking/bin/python
split=val_half
reid=osnet-ain-msdc
base_tracker=btpp-default
log_dir="artifacts/tracking"
detectors=(
    "rfdetr2xl-arme-e19-t010-nms070"
    "yoloxx20-official"
)

cd "$repo_root"

run() {
    printf '\n### %s\n' "$*"
    docker exec -w "$repo_root" "$container" sh -lc "$*"
}

# Skip a stage whose manifest already exists, so this script is re-runnable and
# never trips the artifact store's refuse-to-overwrite rule.
have_manifest() { [[ -f "artifacts/tracking/$1/$2/$split/manifest.json" ]]; }

# A level is committed by writing its manifest last, so a run that died
# mid-sequence leaves files with no manifest. Detect that here and say what to
# remove, rather than letting the retry die at the first existing file.
refuse_partial() {
    local dir="artifacts/tracking/$1/$2/$split"
    if [[ ! -f "$dir/manifest.json" ]] && compgen -G "$dir/*.txt" >/dev/null; then
        printf 'Refusing to start: %s holds track files but no manifest, so an\n' "$dir" >&2
        printf 'earlier run failed part-way. Inspect and remove them, then re-run.\n' >&2
        ls -1 "$dir"/*.txt >&2
        exit 1
    fi
}

for detector in "${detectors[@]}"; do
    combination="${detector}__${reid}__${base_tracker}"

    if ! have_manifest detections "$detector"; then
        printf 'Refusing to start: no %s detections for %s\n' "$split" "$detector" >&2
        exit 1
    fi
    if ! have_manifest embeddings "${detector}__${reid}"; then
        printf 'Refusing to start: no %s embeddings for %s__%s\n' "$split" "$detector" "$reid" >&2
        exit 1
    fi

    refuse_partial tracks "$combination"
    for rung in dti dti-gbi; do
        refuse_partial tracks "${detector}__${reid}__${base_tracker}-${rung}"
    done

    # Baseline. Arm E needs it produced; yoloxx20-official already has it.
    if ! have_manifest tracks "$combination"; then
        run "$python tracking/scripts/run_boosttrack.py \
             --detector $detector --reid $reid --tracker $base_tracker --split $split"
    else
        printf 'baseline tracks present: %s\n' "$combination"
    fi
    if ! have_manifest metrics "$combination"; then
        run "$python tracking/scripts/evaluate_tracks.py \
             --combination $combination --split $split"
    else
        printf 'baseline metrics present: %s\n' "$combination"
    fi

    for rung in dti dti-gbi; do
        target="${detector}__${reid}__${base_tracker}-${rung}"
        # The dti rung is cross-checked against the vendored utils.dti, which is
        # what licenses reimplementing it here rather than shelling out to a
        # function that rewrites directories and destroys the score column.
        verify=""
        [[ "$rung" == "dti" ]] && verify="--verify-against-upstream"
        if ! have_manifest tracks "$target"; then
            run "$python tracking/scripts/postprocess_tracks.py \
                 --combination $combination --rung $rung --split $split $verify"
        else
            printf 'rung tracks present: %s\n' "$target"
        fi
        if ! have_manifest metrics "$target"; then
            run "$python tracking/scripts/evaluate_tracks.py \
                 --combination $target --split $split"
        else
            printf 'rung metrics present: %s\n' "$target"
        fi
    done
done

docker exec -w "$repo_root" "$container" sh -lc \
    "chown -R ${HOST_UID:-2006}:${HOST_GID:-2006} '$repo_root/artifacts/tracking'" || true

printf '\n=== postprocess experiment complete; summarise with build_results_reference.py ===\n'
printf 'log: %s\n' "$log_dir"
