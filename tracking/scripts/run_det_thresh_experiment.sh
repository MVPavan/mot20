#!/usr/bin/env bash
# Experiment mot-n2n.7: is det_thresh = 0.4 a YOLOX-shaped default?
#
# BoostTrack's MOT20 det_thresh of 0.4 was tuned for YOLOX-X score
# distributions. RF-DETR is DETR-based set prediction with no NMS and a
# different score distribution; Roboflow's own examples use 0.5. Our own data
# already hints at this: on rfdetr2xl-e5-t005, det_thresh 0.4/0.5/0.6 gave HOTA
# 68.08/68.35/68.02, so 0.5 already beat the default by +0.27. That sweep was
# never repeated on a best-configuration variant.
#
# det_thresh is applied to BOOSTED scores, after DLO and DUO have already
# rewritten them -- see docs/tracker-parameters.md Stage 6 and cross-stage
# coupling 3. It is not the detector's own threshold, and the detection export
# floor is untouched by this sweep.
#
# BOTH detectors are swept. The association sweep taught us that a tracker
# setting can look detector-specific and turn out to be simply a better MOT20
# value; a claim about RF-DETR here is not believable until the same grid has
# run on YOLOX.
#
# CONTAMINATION: both detectors trained on the full MOT20 train split, which
# contains every val_half frame (arm E verified at 4463/4463). Every figure is
# TRAINING-SET FIT. The pair is internally matched, so the SHAPE of each curve
# and the location of its optimum are the results; the absolute heights are not
# comparable with held-out rows such as arm D's.
#
# PLUMBING GATE, TWO-SIDED. It runs FIRST and BLOCKS the rest: the association
# sweep already produced one setting whose 0.3 and 0.7 runs were byte-identical
# because the value never reached the code path, and eighteen tracking jobs is
# an expensive way to rediscover that.
#
# One side is not enough. det_thresh=0.40 restates the MOT20 default, so a run
# that IGNORES the override entirely still reproduces btpp-default byte-for-byte
# and would pass a same-side-only gate -- exactly the failure being guarded
# against. So the gate is:
#
#   0.40 (the default)   MUST reproduce btpp-default byte-for-byte  -> plumbing
#                        carries the value without corrupting it
#   0.70 (the sentinel)  MUST differ from btpp-default              -> the value
#                        actually reaches the threshold comparison
#
# The sentinel is a real grid point, so nothing is thrown away: the sweep below
# finds its metrics already present and skips it.
#
# Both gate points must be produced BY THIS RUN. The sweep skips any point whose
# metrics manifest exists, so a gate that accepted a pre-existing artifact would
# be comparing output from whatever code was checked out when that artifact was
# written -- a correct 0.70 run from last week passes Gate B even if the tracker
# has since regressed to ignore det_thresh entirely, which is the exact failure
# the gate exists to catch. If a gate artifact is already present the script
# refuses and names it, rather than silently vouching for stale evidence.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
container=nvpt-dm
python=.venv-tracking/bin/python
split=val_half
reid=osnet-ain-msdc
base_tracker=btpp-default
gate_threshold=0.40
sentinel_threshold=0.70
other_thresholds=(0.30 0.35 0.45 0.50 0.55 0.60 0.65 0.70)
detectors=(
    "rfdetr2xl-arme-e19-t010-nms070"
    "yoloxx20-official"
)

cd "$repo_root"

in_container() { docker exec -w "$repo_root" "$container" sh -lc "$*"; }
manifest() { printf 'artifacts/tracking/%s/%s/%s/manifest.json' "$1" "$2" "$split"; }

# Preflight: every input the sweep needs, before any GPU-free but hour-long work.
for detector in "${detectors[@]}"; do
    for level in "detections/$detector" "embeddings/${detector}__${reid}" \
                 "tracks/${detector}__${reid}__${base_tracker}"; do
        path="artifacts/tracking/${level}/${split}/manifest.json"
        if [[ ! -f "$path" ]]; then
            printf 'Refusing to start: missing %s\n' "$path" >&2
            printf 'The btpp-default baseline is required as the gate reference.\n' >&2
            exit 1
        fi
    done
done
printf 'preflight ok: detections, embeddings and baseline tracks present for both detectors\n'

sweep_point() {
    local detector=$1 threshold=$2
    in_container "$python tracking/scripts/sweep_association.py \
        --detector $detector --reid $reid --base-tracker $base_tracker \
        --slug-prefix dt --split $split --point det_thresh=$threshold"
}

# Resolves the derived tracker slug without a pipeline: under `set -o pipefail`
# a `... | head -1` can return non-zero via SIGPIPE, and because the caller
# assigns through a command substitution, `set -e` would abort the run before
# the empty-slug check could report anything useful. Matches the sweep's
# "dry run <slug>:" line, or its "skip <slug>:" line when the point already has
# metrics from an earlier attempt.
point_slug() {
    local detector=$1 threshold=$2 out
    out=$(in_container "$python tracking/scripts/sweep_association.py \
        --detector $detector --reid $reid --base-tracker $base_tracker \
        --slug-prefix dt --split $split --point det_thresh=$threshold --dry-run") || return 1
    if [[ "$out" =~ (dry\ run|skip)\ ([^:[:space:]]+): ]]; then
        printf '%s' "${BASH_REMATCH[2]}"
    fi
}

# Lists a track directory's sequence filenames, sorted. Used to compare the two
# runs as SETS before comparing contents: iterating the reference glob alone
# would silently ignore a candidate that produced extra files, or one that wrote
# a differently named sequence, and both mean the runs are not comparable.
sequence_names() {
    local dir=$1
    find "$dir" -maxdepth 1 -type f -name '*.txt' -printf '%f\n' | LC_ALL=C sort
}

# Echoes "identical" or "differ" for two completed track directories, and exits
# the script if their filename sets do not match or the reference is empty.
compare_runs() {
    local reference=$1 candidate=$2 label=$3
    local reference_names candidate_names name differing=0 compared=0 expected

    if [[ ! -d "$candidate" ]]; then
        printf 'Refusing to continue: %s produced no track directory (%s)\n' \
            "$label" "$candidate" >&2
        exit 1
    fi
    reference_names=$(sequence_names "$reference")
    candidate_names=$(sequence_names "$candidate")
    # An unmatched glob would otherwise leave the loop comparing a literal
    # pattern, and "no files compared" must never read as "gate passed".
    expected=$(printf '%s' "$reference_names" | grep -c . || true)
    if (( expected == 0 )); then
        printf 'Refusing to continue: no reference track files in %s\n' "$reference" >&2
        exit 1
    fi
    if [[ "$reference_names" != "$candidate_names" ]]; then
        printf 'Refusing to continue: %s wrote a different set of sequences.\n' "$label" >&2
        printf '  reference (%s):\n%s\n' "$reference" "$reference_names" >&2
        printf '  candidate (%s):\n%s\n' "$candidate" "$candidate_names" >&2
        exit 1
    fi
    while IFS= read -r name; do
        [[ -n "$name" ]] || continue
        compared=$((compared + 1))
        cmp -s "$reference/$name" "$candidate/$name" || differing=$((differing + 1))
    done <<< "$reference_names"
    if (( compared != expected )); then
        printf 'Refusing to continue: compared %d of %d sequences\n' "$compared" "$expected" >&2
        exit 1
    fi
    if (( differing == 0 )); then
        printf 'identical'
    else
        printf 'differ'
    fi
}

# A gate must test the code running now. The sweep skips a point whose metrics
# already exist, so an existing gate artifact would be waved through untested.
require_fresh() {
    local detector=$1 slug=$2 threshold=$3 dir
    for level in tracks metrics; do
        dir="artifacts/tracking/${level}/${detector}__${reid}__${slug}/$split"
        if [[ -e "$dir/manifest.json" ]]; then
            printf 'Refusing to continue: gate point det_thresh=%s already has %s at\n' \
                "$threshold" "$level" >&2
            printf '  %s\n' "$dir" >&2
            printf 'The sweep would skip it, so the gate would vouch for output produced by\n' >&2
            printf 'whatever code wrote that artifact rather than by the code running now.\n' >&2
            printf 'Remove both levels for this slug and re-run so the gate is genuinely fresh.\n' >&2
            exit 1
        fi
    done
}

for detector in "${detectors[@]}"; do
    reference="artifacts/tracking/tracks/${detector}__${reid}__${base_tracker}/$split"

    printf '\n### plumbing gate A (same-side): %s at det_thresh=%s\n' "$detector" "$gate_threshold"
    gate_slug=$(point_slug "$detector" "$gate_threshold")
    if [[ -z "$gate_slug" ]]; then
        printf 'Refusing to continue: could not resolve the gate tracker slug\n' >&2
        exit 1
    fi
    require_fresh "$detector" "$gate_slug" "$gate_threshold"
    sweep_point "$detector" "$gate_threshold"
    verdict=$(compare_runs "$reference" \
        "artifacts/tracking/tracks/${detector}__${reid}__${gate_slug}/$split" \
        "det_thresh=$gate_threshold")
    if [[ "$verdict" != "identical" ]]; then
        printf '\nGATE A FAILED: det_thresh=%s does not reproduce %s byte-for-byte.\n' \
            "$gate_threshold" "$base_tracker" >&2
        printf 'The override is corrupting a value that should be a no-op, or an input\n' >&2
        printf 'changed under the baseline. Aborting before the remaining points run.\n' >&2
        exit 1
    fi
    printf 'gate A passed: det_thresh=%s reproduces %s byte-for-byte (%s)\n' \
        "$gate_threshold" "$base_tracker" "$gate_slug"

    printf '\n### plumbing gate B (sentinel): %s at det_thresh=%s\n' "$detector" "$sentinel_threshold"
    sentinel_slug=$(point_slug "$detector" "$sentinel_threshold")
    if [[ -z "$sentinel_slug" ]]; then
        printf 'Refusing to continue: could not resolve the sentinel tracker slug\n' >&2
        exit 1
    fi
    require_fresh "$detector" "$sentinel_slug" "$sentinel_threshold"
    sweep_point "$detector" "$sentinel_threshold"
    verdict=$(compare_runs "$reference" \
        "artifacts/tracking/tracks/${detector}__${reid}__${sentinel_slug}/$split" \
        "det_thresh=$sentinel_threshold")
    if [[ "$verdict" != "differ" ]]; then
        printf '\nGATE B FAILED: det_thresh=%s produced tracks identical to the %s\n' \
            "$sentinel_threshold" "$base_tracker" >&2
        printf 'default of %s.\n\n' "$gate_threshold" >&2
        printf 'The likely cause is that the setting never reaches the threshold comparison,\n' >&2
        printf 'which would make the whole sweep a flat curve that says nothing. Confirm\n' >&2
        printf 'before assuming it, though: the other explanation is that every boosted score\n' >&2
        printf 'in [%s, %s) belongs to a detection whose removal changes no surviving track.\n' \
            "$gate_threshold" "$sentinel_threshold" >&2
        printf 'Check the boosted-score histogram to tell them apart. Aborting either way.\n' >&2
        exit 1
    fi
    printf 'gate B passed: det_thresh=%s changes the tracks (%s)\n' \
        "$sentinel_threshold" "$sentinel_slug"
done

for detector in "${detectors[@]}"; do
    printf '\n### det_thresh sweep: %s\n' "$detector"
    # Each value is shell-quoted with %q before entering the container's `sh -lc`
    # string, so the inner shell cannot re-split or expand it. `sh -lc` is kept
    # rather than docker's exec form because the tracking venv relies on the
    # login profile for its CUDA library path.
    points_string=""
    for threshold in "${other_thresholds[@]}"; do
        points_string+=$(printf ' --point %q' "det_thresh=${threshold}")
    done
    in_container "$python tracking/scripts/sweep_association.py \
        --detector $detector --reid $reid --base-tracker $base_tracker \
        --slug-prefix dt --split $split$points_string"
done

in_container "chown -R ${HOST_UID:-2006}:${HOST_GID:-2006} '$repo_root/artifacts/tracking'" || true

printf '\n=== det_thresh sweep complete, gate passed for both detectors ===\n'
