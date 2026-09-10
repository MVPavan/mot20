#!/usr/bin/env bash
# Unattended arm chain: wait for arm B to finish, then run arm D.
#
# Runs INSIDE the nvpt-dm container, detached from any host process, so it
# survives the VS Code session that started it. All output goes to files rather
# than to a pipe: the arm-A launcher teed from the host into a root-owned run
# directory and lost that run's console log permanently, and a host-side pipe is
# a needless dependency for an overnight job.
#
# Deliberately NOT `set -e`. A failing stage must be recorded in the status file
# and the chain must continue or stop explicitly, never die silently.

set -uo pipefail

REPO=/media/data_2/opensource/mot-20
cd "$REPO" || exit 1

# The container runs as root; artifacts are chowned back so they are writable
# from the host session afterwards.
HOST_UID=2006
HOST_GID=2006

STATUS="$REPO/finetuning/artifacts/overnight-status.md"
LOG="$REPO/finetuning/artifacts/overnight-supervisor.log"

ARMB_RUN=finetuning/artifacts/rfdetr-2xl-i5-armb-2026-09-07-r1
# Micro-batch 4 is the primary attempt, not micro-batch 8. Arm B occupies 21 GB
# of each 24 GB card at the 1333px cap; raising the cap to 1600 multiplies the
# token count by about 1.44, so micro-batch 8 would OOM with near certainty and
# only waste a startup. Every arm-D config keeps grad_accum_steps x batch_size
# x devices = 64, matching arm A's effective batch, so whichever one runs stays
# comparable. rfdetr forwards grad_accum_steps to Lightning's
# accumulate_grad_batches, so the accumulation is real.
ARMD_RUN=finetuning/artifacts/rfdetr-2xl-i5-armd-2026-09-07-r1
ARMD_RUN_FALLBACK=finetuning/artifacts/rfdetr-2xl-i5-armd-2026-09-07-r2
ARMD_CONFIG=finetuning/configs/rfdetr_2xl_i5-arm-d-maxsize1600-bs4.toml
ARMD_CONFIG_FALLBACK=finetuning/configs/rfdetr_2xl_i5-arm-d-maxsize1600-bs2.toml
ARMD_DATASET=datasets/finetuning/rfdetr-mot20-crowdhuman-byte65-test-adapted-2026-09-04

export LD_LIBRARY_PATH=/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=finetuning/src

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*" >>"$LOG"
    chown "$HOST_UID:$HOST_GID" "$LOG" 2>/dev/null || true
}

note() {
    printf '%s\n' "$*" >>"$STATUS"
    chown "$HOST_UID:$HOST_GID" "$STATUS" 2>/dev/null || true
}

refresh_results_doc() {
    .venv-tracking/bin/python tracking/scripts/build_results_reference.py \
        --output docs/results-reference.md >>"$LOG" 2>&1
    chown "$HOST_UID:$HOST_GID" docs/results-reference.md 2>/dev/null || true
}

last_epoch() {
    # metrics.csv logs a row per step; the epoch column is the first field.
    awk -F, 'NR > 1 && $1 ~ /^[0-9]+$/ { epoch = $1 } END { print epoch + 0 }' "$1/metrics.csv" 2>/dev/null || echo -1
}

peak_map() {
    # Best of the regular and EMA validation mAP@50:95 seen so far.
    .venv-tracking/bin/python - "$1" <<'PY' 2>/dev/null || echo "unavailable"
import csv, sys
best = 0.0
with open(sys.argv[1] + "/metrics.csv", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        for key in ("val/mAP_50_95", "val/ema_mAP_50_95"):
            if row.get(key):
                best = max(best, float(row[key]))
print(f"{best:.4f}" if best else "no eval yet")
PY
}

arm_pids() {
    pgrep -f "train_rfdetr_2xl\.py.*$1" 2>/dev/null | grep -v "^$$\$" || true
}

wait_for_arm() {
    # Poll rather than wait(): the target is not a child of this supervisor.
    local pattern=$1 cap_hours=$2
    local deadline=$(( $(date +%s) + cap_hours * 3600 ))
    while [ -n "$(arm_pids "$pattern")" ]; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
            return 1
        fi
        sleep 60
    done
    return 0
}

run_arm() {
    local config=$1 run_dir=$2
    log "preparing $run_dir with $config"
    .venv/bin/python finetuning/scripts/train_rfdetr_2xl.py \
        --config "$config" --dataset-root "$ARMD_DATASET" \
        --run-dir "$run_dir" --prepare-run >>"$LOG" 2>&1
    if [ $? -ne 0 ]; then
        log "prepare-run FAILED for $run_dir"
        return 1
    fi
    log "launching torchrun for $run_dir"
    .venv/bin/torchrun --standalone --nproc_per_node=8 \
        finetuning/scripts/train_rfdetr_2xl.py \
        --config "$config" --dataset-root "$ARMD_DATASET" \
        --run-dir "$run_dir" >"$run_dir/console.log" 2>&1
    local rc=$?
    chown "$HOST_UID:$HOST_GID" "$run_dir/console.log" 2>/dev/null || true
    log "torchrun for $run_dir exited with $rc"
    return $rc
}

# ---------------------------------------------------------------- stage 1: arm B
note ""
note "## Supervisor started $(date -Is)"
note ""
log "supervisor started, pid $$"

if wait_for_arm "armb-2026-09-07-r1" 16; then
    armb_epoch=$(last_epoch "$ARMB_RUN")
    armb_peak=$(peak_map "$ARMB_RUN")
    if [ -f "$ARMB_RUN/launcher-result.json" ]; then
        note "- **Arm B: COMPLETE** at $(date -Is). Last epoch $armb_epoch, peak mAP@50:95 $armb_peak."
    else
        note "- **Arm B: EXITED WITHOUT COMPLETION MARKER** at $(date -Is). Last epoch $armb_epoch, peak mAP@50:95 $armb_peak."
        note "  No \`launcher-result.json\`, so it did not finish \`train()\` cleanly. Check \`$ARMB_RUN/console.log\`."
        note "  Proceeding to arm D anyway: its checkpoints are independent of arm B's."
    fi
else
    note "- **Arm B: STILL RUNNING after the 16h cap.** Arm D was NOT started, to avoid two jobs contending for the same 8 GPUs."
    log "16h cap reached with arm B alive; stopping"
    refresh_results_doc
    exit 1
fi
refresh_results_doc

# ---------------------------------------------------------------- stage 2: arm D
note "- Arm D starting $(date -Is): \`max_size = 1600\`, arm-A mix, 30 epochs, micro-batch 4 x"
note "  \`grad_accum_steps = 2\` (effective batch 64, same as arm A). Run dir \`$ARMD_RUN\`."
armd_started=$(date +%s)
run_arm "$ARMD_CONFIG" "$ARMD_RUN"
armd_rc=$?
armd_elapsed=$(( $(date +%s) - armd_started ))

if [ $armd_rc -ne 0 ] \
    && [ $armd_elapsed -lt 3600 ] \
    && grep -qiE "out of memory|CUDA out of memory" "$ARMD_RUN/console.log" 2>/dev/null; then
    note "- Arm D at micro-batch 4 hit **CUDA OOM** after ${armd_elapsed}s. Retrying at micro-batch 2 with"
    note "  \`grad_accum_steps = 4\`, which keeps the effective batch at 64 so the arm stays comparable to arm A."
    note "  The failed attempt is preserved at \`$ARMD_RUN\`; the retry writes to \`$ARMD_RUN_FALLBACK\`."
    log "arm D OOM at micro-batch 4; falling back to micro-batch 2"
    refresh_results_doc
    armd_started=$(date +%s)
    run_arm "$ARMD_CONFIG_FALLBACK" "$ARMD_RUN_FALLBACK"
    armd_rc=$?
    armd_elapsed=$(( $(date +%s) - armd_started ))
    ARMD_RUN=$ARMD_RUN_FALLBACK
fi

armd_epoch=$(last_epoch "$ARMD_RUN")
armd_peak=$(peak_map "$ARMD_RUN")
if [ $armd_rc -eq 0 ] && [ -f "$ARMD_RUN/launcher-result.json" ]; then
    note "- **Arm D: COMPLETE** at $(date -Is) after ${armd_elapsed}s. Last epoch $armd_epoch, peak mAP@50:95 $armd_peak."
else
    note "- **Arm D: FAILED OR INCOMPLETE** (exit $armd_rc) at $(date -Is) after ${armd_elapsed}s."
    note "  Last epoch $armd_epoch, peak mAP@50:95 $armd_peak. See \`$ARMD_RUN/console.log\`."
fi

refresh_results_doc
note "- Supervisor finished $(date -Is). Regenerated \`docs/results-reference.md\`."
log "supervisor finished"
