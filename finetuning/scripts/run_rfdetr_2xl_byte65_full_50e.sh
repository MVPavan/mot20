#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
run_dir="finetuning/artifacts/rfdetr-2xl-byte65-test-adapted-ddp-batch8-lr5e5-aspect-full-50e-2026-09-04-r1"
config="finetuning/configs/rfdetr_2xl_byte65_test_adapted_ddp_batch8_lr5e5_aspect_full_50e.toml"
dataset_root="datasets/finetuning/rfdetr-mot20-crowdhuman-byte65-test-adapted-2026-09-04"

cd "$repo_root"
if [[ -e "$run_dir" ]]; then
    printf 'Refusing to overwrite existing run directory: %s\n' "$run_dir" >&2
    exit 1
fi

prepare_command=$(cat <<EOF
cd "$repo_root"
export LD_LIBRARY_PATH=/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=finetuning/src
.venv/bin/python finetuning/scripts/train_rfdetr_2xl.py --config "$config" --dataset-root "$dataset_root" --run-dir "$run_dir" --prepare-run
EOF
)

train_command=$(cat <<EOF
cd "$repo_root"
export LD_LIBRARY_PATH=/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=finetuning/src
.venv/bin/torchrun --standalone --nproc_per_node=8 finetuning/scripts/train_rfdetr_2xl.py --config "$config" --dataset-root "$dataset_root" --run-dir "$run_dir"
EOF
)

docker exec nvpt-dm sh -lc "$prepare_command"
docker exec nvpt-dm sh -lc "$train_command" 2>&1 | tee "$run_dir/console.log"