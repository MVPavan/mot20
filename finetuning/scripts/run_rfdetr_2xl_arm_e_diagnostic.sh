#!/usr/bin/env bash
# Launch arm E: I4's recipe, validating on data it trains on, to watch the curve.
#
# THE VALIDATION SPLIT IS CONTAMINATED ON PURPOSE. `valid` is MOT20 `val_half`,
# and all 4,463 of those frames are also in `train` (verified 100% overlap).
# Every `val/mAP_*` from this run is a TRAINING-SET FIT measurement. It is not
# comparable with arms A-D and must never share a table with them.
#
# Purpose is I7. Arms A-D showed held-out mAP peaking early and declining, with
# no explanation. This measures the training-set side of the same curve, which
# forks the question: if training-set fit also declines, the cause cannot be
# overfitting, because a model cannot overfit while getting worse on its own
# training data. See the config header.
#
# Seven GPUs, matching I4 exactly, so effective batch stays 56 and arm E's epoch
# N is comparable to I4's epoch N. Device 7 stays free for concurrent inference.
#
# Every epoch is checkpointed (about 2.0 GB each, ~40 GB for 20 epochs). There
# is no valid "best" on a contaminated split, so nothing is selected and every
# epoch is kept for later analysis. Each .ckpt carries both the regular and EMA
# weight branches.
#
# Stopped early by the operator once 6 consecutive epochs fail to beat the
# running best EMA mAP@50:95. There is no automatic early stopping.
#
# The console log is written by `tee` INSIDE the container. Teeing from the host
# into a container-created root-owned run directory fails silently and loses the
# log permanently, which is how the arm-A console log was lost.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
run_dir="finetuning/artifacts/rfdetr-2xl-arme-diagnostic-2026-09-08-r1"
config="finetuning/configs/rfdetr_2xl_i5-arm-e-contaminated-val-diagnostic.toml"
dataset_root="datasets/finetuning/rfdetr-i5-arme-2026-09-08"
gpus="0,1,2,3,4,5,6"
nproc=7

cd "$repo_root"
if [[ -e "$run_dir" ]]; then
    printf 'Refusing to overwrite existing run directory: %s\n' "$run_dir" >&2
    exit 1
fi

# 20 epochs x ~2.0 GB is about 40 GB of checkpoints. Refuse to start if the
# filesystem cannot take it, rather than dying at epoch 15 with a full disk.
avail_gb=$(df -BG --output=avail "$repo_root" | tail -1 | tr -dc '0-9')
if (( avail_gb < 80 )); then
    printf 'Refusing to start: %sG free, need 80G headroom for 20 per-epoch checkpoints.\n' "$avail_gb" >&2
    exit 1
fi
printf 'Disk headroom check passed: %sG free.\n' "$avail_gb"

common_env=$(cat <<EOF
cd "$repo_root"
export LD_LIBRARY_PATH=/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export CUDA_VISIBLE_DEVICES=$gpus
export PYTHONPATH=finetuning/src
EOF
)

docker exec nvpt-dm sh -lc "$common_env
.venv/bin/python finetuning/scripts/train_rfdetr_2xl.py --config \"$config\" --dataset-root \"$dataset_root\" --run-dir \"$run_dir\" --prepare-run"

docker exec nvpt-dm sh -lc "$common_env
.venv/bin/torchrun --standalone --nproc_per_node=$nproc finetuning/scripts/train_rfdetr_2xl.py --config \"$config\" --dataset-root \"$dataset_root\" --run-dir \"$run_dir\" 2>&1 | tee \"$run_dir/console.log\""

# The container writes as root; hand the artifacts back to the host user so the
# tracking environment and git can read them.
docker exec nvpt-dm sh -lc "chown -R ${HOST_UID:-2006}:${HOST_GID:-2006} \"$repo_root/$run_dir\"" || true
