#!/usr/bin/env bash
# Launch arm E2: a true resume of arm E from epoch 20, to 100, on six GPUs.
#
# THE VALIDATION SPLIT IS CONTAMINATED ON PURPOSE, inherited from arm E. `valid`
# is MOT20 `val_half` and all 4,463 of those frames are also in `train`. Every
# `val/mAP_*` from this run is a TRAINING-SET FIT measurement. It is not
# comparable with arms A-D and must never share a table with them.
#
# This resumes rather than warm-starts: `resume` in the config is forwarded to
# Lightning's `trainer.fit(ckpt_path=...)`, so optimizer, EMA, LR-scheduler
# position and epoch counter continue from arm E's epoch 19. Training begins at
# epoch 20. See the config header for the six-GPU step-parity consequences and
# for why the LR schedule is cosine here rather than arm E's inherited step drop.
#
# Six GPUs, not arm E's seven. Devices 6 and 7 stay free for concurrent
# inference. This is a user-chosen deviation and it forfeits I4 step parity.
#
# Checkpoints: interval 10 (~16 GB over 80 epochs) plus a per-epoch `last.ckpt`
# and the always-on best checkpoints. Arm E's interval of 1 would have cost
# ~160 GB here.
#
# The console log is written by `tee` INSIDE the container. Teeing from the host
# into a container-created root-owned run directory fails silently and loses the
# log permanently, which is how the arm-A console log was lost.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
run_dir="finetuning/artifacts/rfdetr-2xl-arme2-continue-2026-09-09-r1"
config="finetuning/configs/rfdetr_2xl_i5-arm-e2-continue-100e-6gpu.toml"
dataset_root="datasets/finetuning/rfdetr-i5-arme-2026-09-08"
resume_ckpt="finetuning/artifacts/rfdetr-2xl-arme-diagnostic-2026-09-08-r1/checkpoint_19.ckpt"
gpus="0,1,2,3,4,5"
nproc=6

cd "$repo_root"
if [[ -e "$run_dir" ]]; then
    printf 'Refusing to overwrite existing run directory: %s\n' "$run_dir" >&2
    exit 1
fi

# The whole run is a continuation of this one file. If checkpoint pruning or a
# stray cleanup removed it, fail here rather than silently starting a fresh
# run from the pretrained weights at epoch 0 - which would look like progress
# for hours before anyone noticed the epoch counter.
if [[ ! -f "$resume_ckpt" ]]; then
    printf 'Refusing to start: resume checkpoint is missing: %s\n' "$resume_ckpt" >&2
    exit 1
fi
if ! grep -qF "resume = \"$resume_ckpt\"" "$config"; then
    printf 'Refusing to start: %s does not resume from %s\n' "$config" "$resume_ckpt" >&2
    exit 1
fi
printf 'Resume checkpoint present: %s\n' "$resume_ckpt"

# 8 interval checkpoints + last.ckpt at ~1.9 GB each, plus ~2.4 GB of best
# checkpoints and the 0.5 GB initialization, is about 21 GB. Refuse to start
# without real headroom rather than dying at epoch 70 with a full disk.
avail_gb=$(df -BG --output=avail "$repo_root" | tail -1 | tr -dc '0-9')
if (( avail_gb < 60 )); then
    printf 'Refusing to start: %sG free, need 60G headroom for ~21G of checkpoints.\n' "$avail_gb" >&2
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
