#!/usr/bin/env bash
# Launch the I4 competition build: ByteTrack parity, arm-D recipe, seven GPUs.
#
# THIS RUN TRAINS ON MOT20 `val_half`. Once its checkpoint exists, no figure in
# `docs/results-reference.md` can be re-measured against it, because the model
# has seen the evaluation split. The dataset carries a formally empty `valid`
# split so validation cannot drive checkpoint selection; the deliverable is the
# FINAL checkpoint, not `checkpoint_best_total.pth`.
#
# Seven GPUs, not eight: device 7 is left free for concurrent inference work
# (embedding exports, tracking runs). Effective batch is therefore 4 x 2 x 7 =
# 56 rather than arm D's 64.
#
# Weights start from the same base checkpoint as every other arm. Arm D is the
# ablation that selected this recipe and its epoch count; it is not the
# initialization. See the config header for the full deviation list.
#
# The console log is written by `tee` INSIDE the container. Teeing from the host
# into a container-created root-owned run directory fails silently and loses the
# log permanently, which is how the arm-A console log was lost.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
run_dir="finetuning/artifacts/rfdetr-2xl-i4-competition-2026-09-08-r1"
config="finetuning/configs/rfdetr_2xl_i4-competition-maxsize1600-7gpu.toml"
dataset_root="datasets/finetuning/rfdetr-mot20-fulltrain-crowdhuman-byte65-test-adapted-2026-09-07"
gpus="0,1,2,3,4,5,6"
nproc=7

cd "$repo_root"
if [[ -e "$run_dir" ]]; then
    printf 'Refusing to overwrite existing run directory: %s\n' "$run_dir" >&2
    exit 1
fi

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
