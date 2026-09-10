#!/usr/bin/env bash
# Launch one I5 CrowdHuman/MOT20 mix-ablation arm.
#
# Usage: run_rfdetr_2xl_i5_arm.sh {b|c}
#
# Arms differ only in the training mix; resolution, schedule shape, capacity,
# base checkpoint, and the held-out valid split are identical, so any difference
# in the result is attributable to the mix. Epoch counts are chosen to match
# optimisation steps against the 2026-09-04 arm-A run rather than to match
# epochs, because the arms have very different images per epoch.
#
# The console log is written by `tee` INSIDE the container. The arm-A launcher
# teed from the host into a run directory the container had created as root,
# which failed silently and lost that run's console log permanently.
set -euo pipefail

arm=${1:-}
case "$arm" in
    b) config_name="i5-arm-b-mot20-oversampled-50pct" ;;
    c) config_name="i5-arm-c-mot20-only" ;;
    *) printf 'Usage: %s {b|c}\n' "$0" >&2; exit 2 ;;
esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
run_dir="finetuning/artifacts/rfdetr-2xl-i5-arm${arm}-2026-09-07-r1"
config="finetuning/configs/rfdetr_2xl_${config_name}.toml"
dataset_root="datasets/finetuning/rfdetr-i5-arm${arm}-2026-09-07"

cd "$repo_root"
if [[ -e "$run_dir" ]]; then
    printf 'Refusing to overwrite existing run directory: %s\n' "$run_dir" >&2
    exit 1
fi

common_env=$(cat <<EOF
cd "$repo_root"
export LD_LIBRARY_PATH=/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=finetuning/src
EOF
)

docker exec nvpt-dm sh -lc "$common_env
.venv/bin/python finetuning/scripts/train_rfdetr_2xl.py --config \"$config\" --dataset-root \"$dataset_root\" --run-dir \"$run_dir\" --prepare-run"

docker exec nvpt-dm sh -lc "$common_env
.venv/bin/torchrun --standalone --nproc_per_node=8 finetuning/scripts/train_rfdetr_2xl.py --config \"$config\" --dataset-root \"$dataset_root\" --run-dir \"$run_dir\" 2>&1 | tee \"$run_dir/console.log\""
